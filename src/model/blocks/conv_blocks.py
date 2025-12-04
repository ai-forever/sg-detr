"""Convolutional blocks."""

from typing import List

from torch import Tensor, nn


class TransposedLayerNorm(nn.Module):
    """LayerNorm(emb.T).T."""

    def __init__(self, dim: int) -> None:
        """Initialize LayerNorm.

        Args:
            dim (int): embedding dim.
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, embedding: Tensor) -> Tensor:
        """Forward pass of the custom LayerNorm.

        Args:
            embedding (Tensor): input tensor

        Returns:
            Tensor: Normed tensor.
        """
        embedding = embedding.transpose(1, 2)
        embedding = self.norm(embedding)
        return embedding.transpose(1, 2)


class ConvBlock1D(nn.Module):
    """1D convolution block with optional upsampling."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        num_layers: int,
        kernel_size: int = 3,
        stride: int = 1,
        use_norm: bool = True,
        last_activate: bool = False,
        upscale: bool = False,
    ) -> None:
        """Intialize a 1D convolution block.

        Args:
            in_channels (int): Number of input channels.
            hidden_dim (int): Hidden dimension.
            out_channels (int): Number of output channels.
            num_layers (int): Number of layers.
            kernel_size (int): Kernel size of the convolution. Defaults to 3.
            stride (int): Stride of the convolution. Defaults to 1.
            use_norm (bool): Whether to use normalization or not.
            last_activate (bool): Whether to use activation on the last layer or not.
            upscale (bool): Whether to use upscale before convolution or not.
        """
        super().__init__()
        hiddens = [hidden_dim for _ in range(num_layers - 1)]
        self.num_layers = len(hiddens) + 1

        activations: List[nn.Module] = []
        for idx in range(self.num_layers):
            if idx < len(hiddens) or last_activate:
                activations.append(nn.ReLU(inplace=True))
            else:
                activations.append(nn.Identity())

        layers: List[nn.Module] = []
        input_channels = [in_channels] + hiddens
        output_channels = hiddens + [out_channels]
        for in_channel, out_channel, activate in zip(input_channels, output_channels, activations):
            conv = nn.Conv1d(
                in_channel,
                out_channel,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=not use_norm,
            )

            layer = [
                nn.Upsample(scale_factor=2, mode="linear") if upscale else nn.Identity(),
                conv,
                TransposedLayerNorm(out_channel) if use_norm else nn.Identity(),
                activate,
            ]
            layers.extend(layer)

        self.layers = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights of the convolution block."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_uniform_(module.weight, a=1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, embeds: Tensor) -> Tensor:
        """Forward Pass.

        Args:
            embeds (Tensor): Input Embeddings.

        Returns:
            Tensor: Output Embeddings.
        """
        embeds = embeds.permute(0, 2, 1)
        embeds = self.layers(embeds)
        return embeds.permute(0, 2, 1)
