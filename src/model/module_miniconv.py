import torch
import torch.nn as nn


class MiniConvNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_filters=16, kernel_size=3, num_layers=2):
        super(MiniConvNet, self).__init__()
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(nn.Conv2d(in_channels, num_filters, kernel_size, padding=kernel_size // 2))
        self.layers.append(nn.ReLU())
        
        # Intermediate layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Conv2d(num_filters, num_filters, kernel_size, padding=kernel_size // 2))
            self.layers.append(nn.ReLU())
        
        # Last layer
        self.layers.append(nn.Conv2d(num_filters, out_channels, kernel_size, padding=kernel_size // 2))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


