from typing import Union, List
import numpy as np
import torch
import torch.nn as nn
from utils import get_activation

## LAYER
class ReLUKANLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, base_activation: str, grid_size: int, k: int, train_boundaries: bool = True):
        super(ReLUKANLayer, self).__init__()
        assert base_activation == "relu"
        self.base_activation = get_activation(base_activation)
        self.grid_size, self.k, self.r = grid_size, k, 4*grid_size*grid_size / ((k+1)*(k+1))
        self.in_features, self.out_features = in_features, out_features
        phase_low = np.arange(-self.k, self.grid_size) / self.grid_size
        phase_height = phase_low + (self.k+1) / self.grid_size
        self.phase_low = nn.Parameter(torch.Tensor(np.array([phase_low for i in range(in_features)])),
                                      requires_grad=train_boundaries)
        self.phase_height = nn.Parameter(torch.Tensor(np.array([phase_height for i in range(in_features)])),
                                         requires_grad=train_boundaries)
        
        self.hidden_dim = 1
        self.equal_size_conv = nn.Conv2d(self.hidden_dim, out_features, (self.grid_size+self.k, in_features))
        
    def forward(self, x: torch.Tensor):
        x1 = self.base_activation(x - self.phase_low)
        x2 = self.base_activation(self.phase_height - x)
        x = x1 * x2 * self.r
        x = x * x
        x = x.reshape((len(x), 1, self.grid_size + self.k, self.in_features))
        x = self.equal_size_conv(x)
        x = x.reshape((len(x), self.out_features, 1))
        return x

## NETWORK
class ReLUKANNetwork(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, init_feature_extractor, layer_hidden: List[int], neurons_hidden: List[int], base_activation: Union[str, List[str]], relu_grid_size: Union[int, List[int]], relu_k: Union[int, List[int]], relu_train_boundary: Union[bool, List[bool]]):
        super(ReLUKANNetwork, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        if init_feature_extractor:
            self.conv_layers = self.make_conv_layers()
            if self.input_channels == 1:
                flat_features = 32 * 7 * 7
            elif self.input_channels == 3:
                flat_features = 32 * 8 * 8
        else:
            self.conv_layers = lambda x: x
            flat_features = input_channels
            if self.input_channels == 1:
                flat_features = 1 * 28 * 28
            elif self.input_channels == 3:
                flat_features = 3 * 32 * 32

        if issubclass(layer_hidden, ReLUKANLayer):
            layer_hidden = [layer_hidden for _ in range(len(neurons_hidden)+1)]
        else:
            assert len(layer_hidden) == len(neurons_hidden) and all([isinstance(l, ReLUKANLayer) for l in layer_hidden])
        if isinstance(base_activation, str):
            base_activation = [base_activation for _ in range(len(neurons_hidden)+1)]
        else:
            assert len(base_activation) == len(neurons_hidden)
        if isinstance(relu_grid_size, int):
            relu_grid_size = [relu_grid_size for _ in range((len(neurons_hidden)+1))]
        else:
            assert len(relu_grid_size) == len(neurons_hidden)
        if isinstance(relu_k, int):
            relu_k = [relu_k for _ in range((len(neurons_hidden)+1))]
        else:
            assert len(relu_k) == len(neurons_hidden)
        if isinstance(relu_train_boundary, bool):
            relu_train_boundary = [relu_train_boundary for _ in range(len(neurons_hidden)+1)]
        else:
            assert len(relu_train_boundary) == len(neurons_hidden)

        self.layers = nn.Sequential(*self.make_relu_layers(flat_features, output_channels, neurons_hidden, layer_hidden, base_activation, relu_grid_size, relu_k, relu_train_boundary))

    def make_relu_layers(self, input_channels, output_channels, neurons_hidden, layer_hiddens, base_activations, relu_grid_sizes, relu_ks, relu_train_boundaries):
        layers = []
        for i, (in_features, out_features) in enumerate(zip([input_channels] + neurons_hidden, neurons_hidden + [output_channels])):
            layers.append(layer_hiddens[i](in_features, out_features, base_activations[i], relu_grid_sizes[i], relu_ks[i], relu_train_boundaries[i]))
        return layers

    def forward(self, x: torch.Tensor):

        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x.squeeze(dim=-1)
