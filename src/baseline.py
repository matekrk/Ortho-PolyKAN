from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_activation, get_pooling

_CONV_OPTIONS = {"kernel_size": 3, "padding": 1, "stride": 1}

class LeNet(nn.Module):

    def __init__(self, input_shape: int = 1, output_shape: int = 10, activation: str = "relu", pooling: str = "max", softmax: bool = False):
        """
        LeNet(
        (conv1): Conv2d(input_shape[1], 6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
        (fc1): Linear(in_features=400, out_features=120, bias=True)
        (fc2): Linear(in_features=120, out_features=84, bias=True)
        (fc3): Linear(in_features=84, out_features=output_shape(10), bias=True)
        )
        ==========================================================================================
        Total params: 61706
        """
        super(LeNet, self).__init__()
        flat_features = 16 * 5 * 5 if input_shape == 1 else 16 * 6 * 6
        self.conv1 = nn.Conv2d(input_shape, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(flat_features, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, output_shape)

        self.activation = get_activation(activation)
        self.pooling = get_pooling(pooling)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input
        """
        x = self.pooling(self.activation(self.conv1(x)))
        x = self.pooling(self.activation(self.conv2(x)))
        x = x.flatten(start_dim=1) # x.view(-1, self.num_flat_features(x))
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x
    
class SkipSequential(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        for layer in self.layers:
            x = layer(x)
        return x + residual

def r9_conv_block(in_channels, out_channels, activation="relu", pool="max"):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              get_activation(activation)]
    if pool: layers.append(get_pooling(pool))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, input_shape: int, output_shape: int, softmax: bool = False):
        """
        ResNet9(
        (feature_extractor): Sequential(
            (0): Sequential(
            (0): Conv2d(input_shape[3], 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
            )
            (1): Sequential(
            (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
            )
            (2): SkipSequential(
            (layers): Sequential(
                (0): Sequential(
                (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU()
                (3): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
                )
                (1): Sequential(
                (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU()
                (3): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
                )
            )
            )
            (3): Sequential(
            (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
            )
            (4): Sequential(
            (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
            )
            (5): SkipSequential(
            (layers): Sequential(
                (0): Sequential(
                (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU()
                (3): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
                )
                (1): Sequential(
                (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU()
                (3): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
                )
            )
            )
        )
        (classifier): Sequential(
            (0): AdaptiveMaxPool2d(output_size=(1, 1))
            (1): Flatten(start_dim=1, end_dim=-1)
            (2): Dropout(p=0.2, inplace=False)
            (3): Linear(in_features=512, out_features=num_classes, bias=True)
        )
        )
        """
        super(ResNet9, self).__init__()
        assert input_shape == 3
        
        self.feature_extractor = nn.Sequential(
            r9_conv_block(input_shape, 64), # conv1
            r9_conv_block(64, 128, pool="max"), # conv2
            SkipSequential(r9_conv_block(128, 128), r9_conv_block(128, 128)), # res 1
            r9_conv_block(128, 256, pool="max"), # conv3
            r9_conv_block(256, 512, pool="max"), # conv4
            SkipSequential(r9_conv_block(512, 512), r9_conv_block(512, 512)) # res 2
        )
        
        classifier = [
            nn.AdaptiveMaxPool2d((1,1)), 
            nn.Flatten(), 
            nn.Dropout(0.2),
            nn.Linear(512, output_shape)
        ]
    
        if softmax:
            classifier.append(nn.Softmax(dim=-1))

        self.classifier = nn.Sequential(*classifier)

    def forward(self, x: torch.Tensor):
        f = self.feature_extractor(x)
        y = self.classifier(f)
        return y
        

def num_pixels(dataset_name):
    return num_channels(dataset_name) * (image_size(dataset_name)**2)
    
def num_channels(dataset_name):
    if dataset_name == "mnist":
        return 1
    elif dataset_name == "fashion":
        return 1
    elif dataset_name == "svhn":
        return 3
    elif dataset_name == "cifar10":
        return 3
    
def image_size(dataset_name):
    if dataset_name == "mnist":
        return 28
    elif dataset_name == "fashion":
        return 28
    elif dataset_name == "svhn":
        return 32
    elif dataset_name == "cifar10":
        return 32

class Fully_connected_net(nn.Module):
    def __init__(self, dataset_name: str, num_classes, widths: List[int], activation: str, bias: bool = True) -> None:
        super(Fully_connected_net, self).__init__()
        self.flatten = nn.Flatten()
        features = []
        for l in range(len(widths)):
            prev_width = widths[l - 1] if l > 0 else num_pixels(dataset_name)
            features.extend([
                nn.Linear(prev_width, widths[l], bias=bias),
                get_activation(activation),
            ])
        classifier = [nn.Linear(widths[-1], num_classes, bias=bias)]
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(*classifier)
        self.last = self.classifier[-1]
        self.gradcam = self.features[-2]

    def forward(self, x):
        x = self.flatten(x)
        x = self.features(x)
        x = self.classifier(x)
        return x

class Fully_connected_net_bn(nn.Module):
    def __init__(self, dataset_name: str, num_classes, widths: List[int], activation: str, bias: bool = True) -> None:
        super(Fully_connected_net_bn, self).__init__()
        self.flatten = nn.Flatten()
        features = []
        for l in range(len(widths)):
            prev_width = widths[l - 1] if l > 0 else num_pixels(dataset_name)
            features.extend([
                nn.Linear(prev_width, widths[l], bias=bias),
                get_activation(activation),
                nn.BatchNorm1d(widths[l])
            ])
        classifier = [nn.Linear(widths[-1], num_classes, bias=bias)]
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(*classifier)
        self.last = self.classifier[-1]
        self.gradcam = self.features[-3]

    def forward(self, x):
        x = self.flatten(x)
        x = self.features(x)
        x = self.classifier(x)
        return x

class Convnet(nn.Module):
    def __init__(self, dataset_name: str, num_classes, widths: List[int], activation: str, pooling: str, bias: bool = True) -> None:
        super(Convnet, self).__init__()
        size = image_size(dataset_name)
        input_width = num_channels(dataset_name)
        features = []
        for l in range(len(widths)):
            prev_width = widths[l - 1] if l > 0 else input_width
            features.extend([
                nn.Conv2d(prev_width, widths[l], bias=bias, **_CONV_OPTIONS),
                get_activation(activation),
                get_pooling(pooling),
            ])
            size //= 2
        self.flatten = nn.Flatten()
        classifier = [nn.Linear(widths[-1]*size*size, num_classes)]
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(*classifier)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

class Convnet_bn(nn.Module):
    def __init__(self, dataset_name: str, num_classes, widths: List[int], activation: str, pooling: str, bias: bool = True) -> None:
        super(Convnet_bn, self).__init__()
        size = image_size(dataset_name)
        input_width = num_channels(dataset_name)
        features = []
        for l in range(len(widths)):
            prev_width = widths[l - 1] if l > 0 else input_width
            features.extend([
                nn.Conv2d(prev_width, widths[l], bias=bias, **_CONV_OPTIONS),
                get_activation(activation),
                nn.BatchNorm2d(widths[l]),
                get_pooling(pooling),
            ])
            size //= 2
        self.flatten = nn.Flatten()
        classifier = [nn.Linear(widths[-1]*size*size, num_classes)]
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(*classifier)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
