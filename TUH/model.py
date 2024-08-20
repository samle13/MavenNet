import torch
from torch import nn
import torch.nn.init as init

_BATCH_NORM_DECAY = 0.1
_BATCH_NORM_EPSILON = 1e-5


def batch_norm2d(num_features):
    return nn.BatchNorm2d(num_features, eps=_BATCH_NORM_EPSILON, momentum=_BATCH_NORM_DECAY)


def _weights_init(layer):
    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
        init.kaiming_normal_(layer.weight)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = batch_norm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = batch_norm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                batch_norm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = self.relu(out)
        return out


class CNN2DModel(nn.Module):
    def __init__(self, num_classes):
        super(CNN2DModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(19, 32, kernel_size=(7, 4), padding=(0, 0)),
            batch_norm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(5, 3)),
            # ResidualBlock(16, 32),  # Add a residual block
            # nn.MaxPool2d(kernel_size=(5, 3)),
            ResidualBlock(32, 64),  # Add a residual block
            nn.MaxPool2d(kernel_size=(5, 3)),
            ResidualBlock(64, 128),
            nn.AdaptiveAvgPool2d((1, 1))

        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, num_classes),  # Adjusted input size
        )
        self.apply(_weights_init)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
