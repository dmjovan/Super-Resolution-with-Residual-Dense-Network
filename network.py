from typing import Dict, Union

import torch
import torch.nn as nn


class DenseLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Concatenating all features extracted for Contiguous Memory (CM)
        return torch.cat([x, self.relu(self.conv(x))], 1)


class ResidualDenseBlock(nn.Module):

    def __init__(self, in_channels: int, growth_rate: int, num_layers: int):
        super(ResidualDenseBlock, self).__init__()

        self.layers = nn.Sequential(
            *[DenseLayer(in_channels=in_channels + growth_rate * i, out_channels=growth_rate) for i in
              range(num_layers)])

        # Creating channel reducing convolution - Local-Feature-Fusion (LFF)
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=(1, 1))

    def forward(self, x):
        # Local-Residual-Learning
        return x + self.lff(self.layers(x))


class SuperResolutionNetwork(nn.Module):
    """ Implementation of Residual Dense Network - Super Resolution Network """

    def __init__(self, hyper_parameters: Dict[str, Union[int, str, float]]) -> None:
        super(SuperResolutionNetwork, self).__init__()

        self.num_channels = hyper_parameters["num_image_channels"]
        self.D = hyper_parameters["num_residual_dense_blocks"]
        self.G0 = hyper_parameters["num_features"]
        self.G = hyper_parameters["growth_rate"]
        self.c = hyper_parameters["num_layers"]
        self.scale = hyper_parameters["scale"]

        ############################################# LOW RESOLUTION SPACE #############################################

        # Creating SFENet (Shallow-Feature-Extraction Net)
        self.sfe1 = nn.Conv2d(in_channels=self.num_channels, out_channels=self.G0, kernel_size=(3, 3), padding=1)
        self.sfe2 = nn.Conv2d(in_channels=self.G0, out_channels=self.G0, kernel_size=(3, 3), padding=1)

        # Creating D RDBs (Residual Dense Blocks) for Local-Residual-Learning part
        self.residual_dense_blocks = nn.ModuleList(
            self.D * [ResidualDenseBlock(in_channels=self.G0, growth_rate=self.G, num_layers=self.c)])

        # Creating 1x1 convolution for channel-dimension reduction for Dense-Feature-Fusion part
        self.channel_reduction_conv = nn.Conv2d(in_channels=self.D * self.G0, out_channels=self.G0, kernel_size=(1, 1))

        # Creating 3x3 convolution for further extraction of features for Global-Feature-Fusion part
        self.global_feature_fusion_conv = nn.Conv2d(in_channels=self.G0, out_channels=self.G0, kernel_size=(3, 3),
                                                    padding=1)

        ############################################ HIGH RESOLUTION SPACE #############################################

        # Creating UpNet for upscaling
        self.up_net = nn.Sequential(nn.Conv2d(in_channels=self.G0, out_channels=self.G0,
                                              kernel_size=(3, 3), padding=1),
                                    nn.PixelShuffle(self.scale))

        # Final convolution
        self.output_conv = nn.Conv2d(in_channels=int(self.G0 / (self.scale ** 2)), out_channels=self.num_channels,
                                     kernel_size=(3, 3), padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output of the first Shallow Feature Extraction
        F_minus_1 = self.sfe1(x)

        # Output of the second Shallow Feature Extraction
        F_0 = self.sfe2(F_minus_1)

        # Stacking each output of each RDB - Extracting local dense features
        residual_block_outputs = []
        x = F_0
        for residual_block in self.residual_dense_blocks:
            x = residual_block(x)
            residual_block_outputs.append(x)

        # Concatenating RDB outputs by channel dimension
        concatenated_rdb_outputs = torch.cat(residual_block_outputs, dim=1)

        # Reducing number of channels - Dense Feature Fusion
        x = self.channel_reduction_conv(concatenated_rdb_outputs)

        # Further extraction of features - Global Feature Fusion
        x = self.global_feature_fusion_conv(x)

        # Final output of Dense Feature Fusion
        x = x + F_minus_1

        # Upscaling with sub-pixel convolution
        x = self.up_net(x)

        # Final convolution
        x = self.output_conv(x)

        return x
