"""Various positional encodings for the transformer."""

import math
from typing import Optional

import torch
from torch import Tensor, nn


class TrainablePositionalEncoding(nn.Module):
    """
    A module to add learnable positional encodings to the input features.

    Creates positional embeddings and applies layer norm and dropout to the sum of input features and these embeddings.

    Attributes:
        position_embeddings (nn.Embedding): Embedding layer for positional encodings.
        norm (nn.LayerNorm): Layer normalization.
        dropout (nn.Dropout): Dropout layer.
    """

    def __init__(self, dim, max_len: int, dropout: float = 0.1) -> None:
        """
        Initialize PositionalEncoding.

        Args:
            dim (int): The dimension of the input tensor.
            max_len (int): The maximum sequence length for positional encodings. Defaults to 5000.
            dropout (float): Dropout rate. Default: 0.1.
        """
        super().__init__()
        self.position_embeddings = nn.Embedding(max_len, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_feat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for adding positional encodings to input features.

        Args:
            input_feat (torch.Tensor): The input feature tensor of shape (N, L, D) where
                                    N is the batch size, L is the sequence length, and
                                    D is the feature dimension.

        Returns:
            torch.Tensor: The output tensor with positional encodings added to the input,
                        normalized and passed through dropout. Shape is (N, L, D).
        """
        bsz, seq_length = input_feat.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_feat.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = self.norm(input_feat + position_embeddings)
        return self.dropout(embeddings)


class PositionEmbeddingSine(nn.Module):
    """MRDETR implementation of PE."""

    def __init__(
        self,
        dim: int = 256,
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None,
    ) -> None:
        """Init of the PositionEmbeddingSine.

        Args:
            dim (int): features dim.
            temperature (int): Defaults to 10000.
            normalize (bool): Defaults to True.
            scale (Optional[float]): Defaults to None.

        Raises:
            ValueError: normalize should be True if scale is passed
        """
        super().__init__()
        self.dim = dim
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask: Tensor) -> Tensor:
        """
        Forward pass of the PositionEmbeddingSine.

        Args:
            mask (Tensor): embedding mask (batch_size, L), with 1 as valid

        Returns:
            Tensor: PE embedding.
        """
        assert mask is not None
        x_embed = mask.cumsum(1, dtype=torch.float32)  # (bsz, L)
        if self.normalize:
            eps = 1e-6
            x_embed = x_embed / (x_embed[:, -1:] + eps)
            x_embed = x_embed * self.scale

        dim_t = torch.arange(self.dim, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.dim)

        pos_x = x_embed[:, :, None] / dim_t
        sin_pos = pos_x[:, :, 0::2].sin()
        cos_pos = pos_x[:, :, 1::2].cos()
        return torch.stack((sin_pos, cos_pos), dim=3).flatten(2)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for sequence data as introduced in [1].

    This module adds positional encodings to the input tensor to provide
    positional information to the model.

    References:
        1. Vaswani et al. (https://arxiv.org/abs/1706.03762)
    """

    fixed_pe_constant: float = 10000.0

    def __init__(self, dim: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize the PositionalEncoding module.

        Args:
            dim (int): The dimension of the input tensor.
            max_len (int): The maximum sequence length for positional encodings. Defaults to 5000.
            dropout (float): The dropout probability. Defaults to 0.1.
        """
        super().__init__()
        self._create_fixed_pe(max_len, dim)
        self.dropout = nn.Dropout(p=dropout)

    def _create_fixed_pe(self, max_len: int, dim: int):
        """
        Create fixed positional encodings based on the given maximum sequence length and dimension.

        Args:
            max_len (int): The maximum sequence length.
            dim (int): The dimension of the input tensor.
        """
        position = torch.arange(max_len).unsqueeze(1)

        div_size = -math.log(self.fixed_pe_constant) / dim
        div_term = torch.exp(torch.arange(0, dim, 2) * div_size)

        pos_enc = torch.zeros(1, max_len, dim)
        pos_enc[0, :, 0::2] = torch.sin(position * div_term)
        pos_enc[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pos_enc", pos_enc)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PositionalEncoding module.

        Args:
            embedding (Tensor): The input tensor with shape (batch_size, sequence_length, dim).

        Returns:
            Tensor: The input tensor with positional encodings added.
        """
        embedding = embedding + self.pos_enc[:, : embedding.size(1)]  # type: ignore
        return self.dropout(embedding)


def gen_sineembed_for_position(pos_tensor: Tensor, d_model: int, temperature: int = 10000) -> Tensor:
    """Generate sine embeddings for position.

    Args:
        pos_tensor (Tensor): position tensor (anchor points)
        d_model (int): dimension of the model
        temperature (int): temperature of the pos emb.

    Returns:
        Tensor: sine embeddings for position
    """
    scale = 2 * math.pi
    dim_t = torch.arange(d_model // 2, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * (dim_t // 2) / (d_model // 2))

    # prepare PE embedding for center value
    center_embed = pos_tensor[:, :, 0] * scale
    pos_x = center_embed[:, :, None] / dim_t
    pos_x = torch.stack(  # noqa: WPS317
        (
            pos_x[:, :, 0::2].sin(),
            pos_x[:, :, 1::2].cos(),
        ),
        dim=3,
    ).flatten(2)

    # prepare PE embedding for width value
    span_embed = pos_tensor[:, :, 1] * scale
    pos_w = span_embed[:, :, None] / dim_t
    pos_w = torch.stack(  # noqa: WPS317
        (
            pos_w[:, :, 0::2].sin(),
            pos_w[:, :, 1::2].cos(),
        ),
        dim=3,
    ).flatten(2)

    return torch.cat((pos_x, pos_w), dim=2)
