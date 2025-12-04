"""Prepare multiscale features for the transformer."""

from typing import List

import torch
from torch import Tensor, nn

from src.model.blocks.conv_blocks import TransposedLayerNorm

INIT_CONST: float = 0.01


class DWSCond1d(nn.Module):
    """DWS conv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
        bias: bool = False,
    ):
        """Initialize DWS block.

        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            kernel_size (int): kernel size. Defaults to 3.
            padding (int): padding size. Defaults to 1.
            stride (int): stride. Defaults to 1.
            bias (bool): whether to use bias or not. Defaults to False.
        """
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, emb: Tensor) -> Tensor:
        """Forward pass of the DWS block.

        Args:
            emb (Tensor): input tensor

        Returns:
            Tensor: output tensor
        """
        out = self.depthwise(emb)
        return self.pointwise(out)


class FPNSequence(nn.Module):
    """Feature Pyramid Network sequence."""

    def __init__(self, feature_dim: int) -> None:
        """Initialize a Feature Pyramid Network sequence.

        Args:
            feature_dim (int): Feature dimension.
        """
        super().__init__()
        self.upscale = nn.Sequential(
            nn.Upsample(scale_factor=2),
            DWSCond1d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=False),  # noqa: WPS204
            TransposedLayerNorm(feature_dim),  # noqa: WPS204
            nn.GELU(),
            DWSCond1d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=False),
            TransposedLayerNorm(feature_dim),
        )

        self.level0 = nn.Sequential(
            DWSCond1d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=False),
            TransposedLayerNorm(feature_dim),
            nn.GELU(),
            DWSCond1d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=False),
            TransposedLayerNorm(feature_dim),
        )

        self.level1 = nn.Sequential(
            DWSCond1d(feature_dim, feature_dim, kernel_size=3, stride=2, padding=1, bias=False),
            TransposedLayerNorm(feature_dim),
            nn.GELU(),
            DWSCond1d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=False),
            TransposedLayerNorm(feature_dim),
        )

        self.level2 = nn.Sequential(
            nn.GELU(),
            DWSCond1d(feature_dim, feature_dim, kernel_size=3, stride=2, padding=1, bias=False),
            TransposedLayerNorm(feature_dim),
            nn.GELU(),
            DWSCond1d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=False),
            TransposedLayerNorm(feature_dim),
        )
        self._init_params()

    def _init_params(self) -> None:
        """Init stem blocks."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                torch.nn.init.normal_(module.weight, std=INIT_CONST)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)  # type: ignore

    def forward(self, emb: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through the FPN sequence.

        Args:
            emb (torch.Tensor): Input tensor.

        Returns:
            List[torch.Tensor]: List of scaled features.
        """
        emb = emb.permute(0, 2, 1)
        levelu = self.upscale(emb).permute(0, 2, 1)  # 160
        level0 = self.level0(emb).permute(0, 2, 1)  # 80
        level1 = self.level1(emb)  # 40
        level2 = self.level2(level1).permute(0, 2, 1)  # 20
        return [levelu, level0, level1.permute(0, 2, 1), level2]
