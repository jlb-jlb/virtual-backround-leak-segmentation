# import torch
import torch.nn as nn



class ConvNetSimple(nn.Module):
    """
    ConvNetSimple: A simple convolutional neural network for single-channel output.
    Model from Class "Architecture of Machine Learning Systems - 2024"

    Args:
        channels (int): Number of input channels. Defaults to 3.

    Attributes:
        model (torch.nn.Sequential): A stack of layers (three 3×3 convolutions
            with ReLU activations, a 1×1 convolution to produce a single output
            channel, and a Sigmoid).

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Performs a forward pass through the network.
            Input:
                x (torch.Tensor): Tensor of shape (N, channels, H, W).
            Returns:
                torch.Tensor: Output tensor of shape (N, 1, H, W) with values in [0, 1].
    """

    def __init__(self, channels: int = 3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)
    


class UNetSimple(nn.Module):
    """
    UNetSimple

    A simplified U-Net-style convolutional neural network for image segmentation. The network consists of:
        - An encoder path that downsamples the input via conv→ReLU→conv→ReLU→max‐pool blocks.
        - A bottleneck convolutional layer.
        - A decoder path that upsamples via transposed convolutions followed by conv→ReLU blocks.
        - A final 1×1 convolution and sigmoid activation to produce a single‐channel segmentation map.

    Args:
            in_channels (int): Number of channels in the input images (e.g., 3 for RGB).

    Attributes:
            model (nn.Sequential): The full sequential model implementing the encoder, bottleneck, decoder, and output.

    Forward:
            x (torch.Tensor): Input tensor of shape (N, in_channels, H, W).

    Returns:
            torch.Tensor: Output segmentation map of shape (N, 1, H, W) with values in [0, 1].
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.model = nn.Sequential(
            # Encoder
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Bottleneck
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Decoder
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Output
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)