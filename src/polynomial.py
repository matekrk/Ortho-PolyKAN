from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache
from utils import get_activation

## LAYER
class PolynomialLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, base_activation: str, polynomial_order: int, layer_norm: bool = True):
        super(PolynomialLayer, self).__init__()

        # ParameterList for the base weights of each layer
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features))
        # ParameterList for the polynomial weights for chebyshev expansion
        self.poly_weight = nn.Parameter(torch.randn(out_features, in_features * (polynomial_order + 1)))
        # ModuleList for layer normalization for each layer's output
        if layer_norm:
            self.layer_norm = nn.LayerNorm(out_features)
        else:
            self.layer_norm = nn.BatchNorm1d(out_features)

        self.polynomial_order = polynomial_order
        self.base_activation = get_activation(base_activation)
        #self.init_weights()

    def init_weights(self):
        """ Initialize weights using Kaiming uniform distribution for better training start """
        for weight in self.base_weight:
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')
        for weight in self.poly_weight:
            nn.init.kaiming_uniform_(weight, nonlinearity='linear')

    def compute_polynomials(self, x):
        # return torch.eye(x.size(0))[:self.polynomial_order] @ x
        return self.compute_efficient_monomials(x)
        
    def compute_efficient_monomials(self, x):
        powers = torch.arange(self.polynomial_order + 1, device=x.device, dtype=x.dtype)
        x_expanded = x.unsqueeze(-1).repeat(1, 1, self.polynomial_order + 1)
        return torch.pow(x_expanded, powers)
    
    def forward(self, x):
        # Apply base activation to input and then linear transform with base weights
        base_output = F.linear(self.base_activation(x), self.base_weight)
        
        # Normalize x to the range [-1, 1] for stable Legendre polynomial computation
        x_normalized = 2 * (x - x.min()) / (x.max() - x.min()) - 1
        # Compute polynomials for the normalized x
        basis = self.compute_polynomials(x_normalized, self.polynomial_order)
        # Reshape to match the expected input dimensions for linear transformation
        basis = basis.view(x.size(0), -1)

        # Compute polynomial output using polynomial weights
        poly_output = F.linear(basis, self.poly_weight)
        # Combine base and polynomial outputs, normalize, and activate
        x = self.base_activation(self.layer_norm(base_output + poly_output))
        return x

## NETWORK
class PolynomialNetwork(nn.Module):
    def __init__(self, input_channels: int, 
                 output_channels: int, 
                 init_feature_extractor: bool, 
                 layer_hidden: Union[PolynomialLayer, List[PolynomialLayer]], 
                 neurons_hidden: List[int], 
                 base_activation: Union[str, List[str]], 
                 polynomial_order: Union[int, List[int]], 
                 layer_norm: bool = True):
        super(PolynomialNetwork, self).__init__()
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

        if issubclass(layer_hidden, PolynomialLayer):
            layer_hidden = [layer_hidden for _ in range(len(neurons_hidden)+1)]
        else:
            assert len(layer_hidden) == len(neurons_hidden) and all([isinstance(l, PolynomialLayer) for l in layer_hidden])
        if isinstance(base_activation, str):
            base_activation = [base_activation for _ in range(len(neurons_hidden)+1)]
        else:
            assert len(base_activation) == len(neurons_hidden)
        if isinstance(polynomial_order, int):
            polynomial_order = [polynomial_order for _ in range((len(neurons_hidden)+1))]
        else:
            assert len(polynomial_order) == len(neurons_hidden)

        self.layers = nn.Sequential(*self.make_poly_layers(flat_features, output_channels, neurons_hidden, layer_hidden, base_activation, polynomial_order, layer_norm))


    def make_conv_layers(self, output_channels: int = 32, hidden_channels: int = 16, kernel_size: int = 3, stride: int = 1, padding: int = 1, maxpoolsize: int = 2):
        # Convolutional encoder for initial feature extraction
        return nn.Sequential(
            nn.Conv2d(self.input_channels, hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(maxpoolsize, maxpoolsize),
            nn.Conv2d(hidden_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(maxpoolsize, maxpoolsize),
        )
    
    def make_poly_layers(self, input_channels, output_channels, neurons_hidden, layer_hiddens, base_activations, polynomial_orders, layer_norm):
        layers = []
        for i, (in_features, out_features) in enumerate(zip([input_channels] + neurons_hidden, neurons_hidden + [output_channels])):
            layers.append(layer_hiddens[i](in_features, out_features, base_activations[i], polynomial_orders[i], layer_norm))
        return layers

    def forward(self, x: torch.Tensor):
        # x = x.to(self.base_weights[0].device)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)

        x = self.layers(x)
        #for i, layer in enumerate(self.layers):
        #    x = layer(x)
        
        return x
