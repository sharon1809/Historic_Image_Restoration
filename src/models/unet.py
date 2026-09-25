import torch
import torch.nn as nn

from .blocks import EncoderBlock, DecoderBlock, ResidualBlock


class RestorationUNet(nn.Module):
    """
    Residual U-Net for historical image restoration.

    Input:
        [B, 3, H, W]

    Output:
        [B, 3, H, W]
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
        num_levels: int = 4,
        residual_blocks_per_level: int = 2,
    ):
        super().__init__()

        if num_levels != 4:
            raise ValueError(
                "This implementation currently supports num_levels=4."
            )

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        # Encoder
        self.encoder1 = EncoderBlock(
            in_channels,
            c1,
            residual_blocks_per_level,
        )

        self.encoder2 = EncoderBlock(
            c1,
            c2,
            residual_blocks_per_level,
        )

        self.encoder3 = EncoderBlock(
            c2,
            c3,
            residual_blocks_per_level,
        )

        self.encoder4 = EncoderBlock(
            c3,
            c4,
            residual_blocks_per_level,
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(c4),
            ResidualBlock(c4),
        )

        # Decoder
        self.decoder4 = DecoderBlock(
            c4,
            c4,
            c4,
            residual_blocks_per_level,
        )

        self.decoder3 = DecoderBlock(
            c4,
            c3,
            c3,
            residual_blocks_per_level,
        )

        self.decoder2 = DecoderBlock(
            c3,
            c2,
            c2,
            residual_blocks_per_level,
        )

        self.decoder1 = DecoderBlock(
            c2,
            c1,
            c1,
            residual_blocks_per_level,
        )

        # Final reconstruction layer
        self.output_layer = nn.Conv2d(
            c1,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Encoder
        skip1, x = self.encoder1(x)
        skip2, x = self.encoder2(x)
        skip3, x = self.encoder3(x)
        skip4, x = self.encoder4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.decoder4(x, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)

        # Output constrained to [0, 1]
        x = torch.sigmoid(self.output_layer(x))

        return x