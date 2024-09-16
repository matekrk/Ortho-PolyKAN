from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_activation, get_pooling

_CONV_OPTIONS = {"kernel_size": 3, "padding": 1, "stride": 1}

class LeNet(nn.Module):

    def __init__(self, n_classes = 10):
        """
        LeNet(
        (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
        (fc1): Linear(in_features=400, out_features=120, bias=True)
        (fc2): Linear(in_features=120, out_features=84, bias=True)
        (fc3): Linear(in_features=84, out_features=10, bias=True)
        )
        ==========================================================================================
        Total params: 61706
        """
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, n_classes)

    def forward(self, x):
        """
        Args:
            x: input
        """
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) #FIXME: Use get_activation, get_pooling
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.flatten(start_dim=1) # x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        f = self.fc2(x)
        x = F.relu(f)
        x = self.fc3(x)
        return x

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
