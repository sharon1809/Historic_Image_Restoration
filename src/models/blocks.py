import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual block:
        x -> Conv -> ReLU -> Conv -> x + result
    """

    def __init__(self, channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class EncoderBlock(nn.Module):
    """
    Encoder block:
    - convolution changes channel count
    - residual blocks refine features
    - stride=2 convolution downsamples the feature map
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_residual_blocks: int = 2,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

        self.residual = nn.Sequential(
            *[
                ResidualBlock(out_channels)
                for _ in range(num_residual_blocks)
            ]
        )

        self.downsample = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, x: torch.Tensor):
        features = self.conv(x)
        features = self.residual(features)
        downsampled = self.downsample(features)

        return features, downsampled


class DecoderBlock(nn.Module):
    """
    Decoder block:
    - upsamples the feature map
    - concatenates the corresponding encoder skip connection
    - reduces/refines the resulting channels
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        num_residual_blocks: int = 2,
    ):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.conv = nn.Sequential(
            nn.Conv2d(
                out_channels + skip_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

        self.residual = nn.Sequential(
            *[
                ResidualBlock(out_channels)
                for _ in range(num_residual_blocks)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:

        x = self.upsample(x)

        # Safety check for spatial dimensions.
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat([x, skip], dim=1)

        x = self.conv(x)
        x = self.residual(x)

        return x