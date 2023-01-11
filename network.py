from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShallowFeatureExtraction(nn.Module):

    def __init__(self, in_: int, out_: int) -> None:
        super(ShallowFeatureExtraction, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_, out_channels=out_, kernel_size=(3, 3), padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResidualDenseBlock(nn.Module):

    def __init__(self, G0: int = 64, G: int = 64, c: int = 6) -> None:

        super(ResidualDenseBlock, self).__init__()

        layer_list = []
        for i in range(c):
            in_dim = G0 + i * G
            layer_list.append(nn.Conv2d(in_channels=in_dim, out_channels=G, kernel_size=(3, 3), padding=1))

        self.layers = nn.ModuleList(layer_list)
        self.final_conv = nn.Conv2d(in_channels=(G0 + c * G), out_channels=G0, kernel_size=(1, 1))

    def forward(self, x_: torch.Tensor) -> torch.Tensor:

        ins = [x_]
        for layer in self.layers:
            x = F.relu(layer(torch.cat(ins, dim=1)))
            ins.append(x)

        x = torch.cat(ins, dim=1)
        x = self.final_conv(x)
        return x + x_


class SuperResolutionNetwork(nn.Module):

    def __init__(self, hyper_parameters: Dict[str, Union[int, str, float]]) -> None:

        super(SuperResolutionNetwork, self).__init__()

        self.G0 = hyper_parameters["G0"]
        self.G = hyper_parameters["G"]
        self.ratio = hyper_parameters["ratio"]
        self.d = hyper_parameters["d"]
        self.c = hyper_parameters["c"]

        if self.G0 % self.ratio ** 2:
            RuntimeError(f"Feature map count (G0) {self.G0} must be divisible by upscale ratio {self.ratio}")

        self.pixel_shuffle = nn.PixelShuffle(self.ratio)
        self.conv1 = ShallowFeatureExtraction(in_=3, out_=self.G0)
        self.conv2 = ShallowFeatureExtraction(in_=self.G0, out_=self.G0)

        block_list = self.d * [ResidualDenseBlock(G0=self.G0, G=self.G, c=self.c)]
        self.blocks = nn.ModuleList(block_list)
        self.conv3 = nn.Conv2d(in_channels=self.d * self.G0, out_channels=self.G0, kernel_size=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=self.G0, out_channels=self.G0, kernel_size=(3, 3), padding=1)
        self.upscale_conv = nn.Conv2d(in_channels=self.G0, out_channels=self.G0, kernel_size=(3, 3), padding=1)
        self.conv5 = nn.Conv2d(in_channels=int(self.G0 / (self.ratio ** 2)), out_channels=3, kernel_size=(3, 3),
                               padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        f_1 = self.conv1(x)
        x = self.conv2(f_1)

        block_outputs = []
        for block in self.blocks:
            x = block(x)
            block_outputs.append(x)

        x = self.conv3(torch.cat(block_outputs, dim=1))
        x = self.conv4(x)
        x = x + f_1

        x = self.upscale_conv(x)
        x = self.pixel_shuffle(x)
        x = self.conv5(x)

        return x
