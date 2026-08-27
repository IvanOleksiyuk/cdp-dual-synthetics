import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")

import torch
import torch.nn as nn
from src.model.modules import IterativeNormLayer

class SimpleUNet(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 out_channels=1, 
                 features=[16, 32, 64, 128],
                 use_norm_in_conv_block="batch",
                 activation="relu"):  # Added activation parameter, default to ELU
        super(SimpleUNet, self).__init__()
        self.use_norm_in_conv_block = use_norm_in_conv_block
        self.activation = activation
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        for feature in features:
            self.encoder.append(self._conv_block(in_channels, feature))
            in_channels = feature

        # Decoder
        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(self._conv_block(feature * 2, feature))

        self.bottleneck = self._conv_block(features[-1], features[-1] * 2)
        self.final_layer = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx // 2]
            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](x)

        return self.final_layer(x)

    def _conv_block(self, in_channels, out_channels):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        ]
        if self.use_norm_in_conv_block == "batch":
            layers.append(nn.BatchNorm2d(out_channels))
        elif self.use_norm_in_conv_block == "layer":
            layers.append(nn.GroupNorm(1, out_channels))  # LayerNorm for CNNs
        elif self.use_norm_in_conv_block == "fastlearn_batch":
            layers.append(IterativeNormLayer(inpt_dim=(out_channels, -1, -1), dims=(-2, -1), max_n=10_000))
        # else: no normalization
        layers.append(self._get_activation())
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
        if self.use_norm_in_conv_block == "batch":
            layers.append(nn.BatchNorm2d(out_channels))
        elif self.use_norm_in_conv_block == "layer":
            layers.append(nn.GroupNorm(1, out_channels))
        elif self.use_norm_in_conv_block == "fastlearn_batch":
            layers.append(IterativeNormLayer(inpt_dim=(out_channels, -1, -1), dims=(-2, -1), max_n=10_000))
        layers.append(self._get_activation())
        return nn.Sequential(*layers)

    def _get_activation(self):
        if self.activation == "relu":
            return nn.ReLU(inplace=True)
        elif self.activation == "elu":
            return nn.ELU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}. Use 'relu' or 'elu'.")

class StackedUNet(nn.Module):
    def __init__(self, num_unets=2,  # Reduced number of stacked U-Nets
                 in_channels=1, 
                 out_channels=1, 
                 residual_weight=0.5,
                 trainable_residual=True,  # Optionally trainable residual weight
                 gap_channels_num=4,  # Intermediate channel size between U-Nets
                 use_sigmoid_on_residual_weight=False,
                 simple_unet_parameters=None):  # New dict argument

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual_channels = min(in_channels, out_channels)  # Ensure residual channels are not larger than input or output

        super(StackedUNet, self).__init__()
        self.unets = nn.ModuleList()
        self.residual_weight = nn.Parameter(torch.tensor(residual_weight, dtype=torch.float32)) if trainable_residual else torch.tensor(residual_weight, dtype=torch.float32)
        self.use_sigmoid_on_residual_weight = use_sigmoid_on_residual_weight

        # Default parameters for SimpleUNet
        if simple_unet_parameters is None:
            simple_unet_parameters = {
                "features": [16, 32, 64, 128],
                "use_norm_in_conv_block": "batch",
                "activation": "relu"
            }

        # Create stacked U-Nets with intermediate channels
        for i in range(num_unets):
            current_in_channels = in_channels if i == 0 else gap_channels_num
            current_out_channels = out_channels if i == num_unets - 1 else gap_channels_num
            self.unets.append(SimpleUNet(current_in_channels, current_out_channels,
                                        features=simple_unet_parameters.get("features", [16, 32, 64, 128]),
                                        use_norm_in_conv_block=simple_unet_parameters.get("use_norm_in_conv_block", "batch"),
                                        activation=simple_unet_parameters.get("activation", "relu")))

    def forward(self, x):
        residual = x
        for unet in self.unets:
            x = unet(x)
            # Apply residual connection only to the first in_channels channels
            weight = torch.sigmoid(self.residual_weight) if self.use_sigmoid_on_residual_weight else self.residual_weight
            x[:, :self.residual_channels] += weight * residual[:, :self.residual_channels]
            residual = x  # Update residual for next U-Net
        return x
