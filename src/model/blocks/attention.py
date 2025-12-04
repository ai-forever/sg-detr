"""
DAB-DETR MultiheadAttention that support query, key, and value to have different dimensions.

Note: Query, key, and value projections are removed.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn.functional import dropout, linear, pad, softmax
from torch.nn.modules.linear import Linear
from torch.nn.modules.module import Module


class DABMultiheadAttention(Module):
    """MHA with no projection and dummy tokens support."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout_prob: float = 0.1,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        num_dummies: Optional[int] = None,
    ):
        """Initialize DABMultiheadAttention.

        Args:
            embed_dim (int): The embedding dimension.
            num_heads (int): The number of attention heads.
            dropout_prob (float): The dropout probability.
            add_zero_attn (bool): Whether to add a column of zeros to the attention weights. Defaults to False.
            kdim (Optional[int]): The dimension of the key. Defaults to None.
            vdim (Optional[int]): The dimension of the value. Defaults to None.
            num_dummies (Optional[int]): The number of dummy tokens. Defaults to None.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.num_dummies = num_dummies
        self.dropout_prob = dropout_prob
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.out_proj = Linear(self.vdim, self.vdim)

        self.add_zero_attn = add_zero_attn

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        dummy=True,
        saliency_scores: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass for DABMultiheadAttention.

        Args:
            query (Tensor): Query tensor of shape (L, N, E) where L is the target sequence length, N is the batch size,
            key (Tensor): Key tensor of shape (S, N, E) where S is the source sequence length, N is the batch size,
            value (Tensor): Value tensor of shape (S, N, E) where S is the source sequence length, N is the batch size,
            key_padding_mask (Optional[Tensor]): Key padding mask of shape (N, S) where N is the batch size,
            need_weights (bool): Whether to return attention weights. Defaults to True.
            attn_mask (Optional[Tensor]): Attention mask of shape (L, S) where L is the target sequence length
            dummy (bool): Whether to use dummy tokens. Defaults to True.

        Returns:
            Tuple[Tensor, Optional[Tensor]]: The output tensor and the attention weights
        """
        return multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            self.add_zero_attn,
            self.dropout_prob,
            self.out_proj.weight,
            self.out_proj.bias,
            num_dummies=self.num_dummies,
            out_dim=self.vdim,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            dummy=dummy,
            saliency_scores=saliency_scores,
        )


def prepare_attention_mask(attn_mask: Tensor, query: Tensor, key: Tensor, bsz: int, num_heads: int) -> Tensor:
    """Prepare attention mask for multi-head attention.

    Args:
        attn_mask (Tensor): Attention mask to be prepared
        query (Tensor): Query tensor of shape (L, N, E) where L is the target sequence length, N is the batch size
        key (Tensor): Key tensor of shape (S, N, E) where S is the source sequence length, N is the batch size
        bsz (int): The batch size
        num_heads (int): The number of attention heads

    Raises:
        RuntimeError: If the size of the 2D attn_mask is not correct.
        RuntimeError: If the size of the 3D attn_mask is not correct.
        RuntimeError: If attn_mask's dimension is not supported.

    Returns:
        Tensor: The prepared attention mask
    """
    if attn_mask.dtype == torch.uint8:
        attn_mask = attn_mask.to(torch.bool)

    if attn_mask.dim() == 2:
        attn_mask = attn_mask.unsqueeze(0)
        if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
            raise RuntimeError("The size of the 2D attn_mask is not correct.")
    elif attn_mask.dim() == 3:
        if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
            raise RuntimeError("The size of the 3D attn_mask is not correct.")
    else:
        raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    return attn_mask


def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    out_dim: int,
    num_dummies: Optional[int] = None,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    dummy=True,
    saliency_scores: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Forward pass for DABMultiheadAttention.

    Args:
        query (Tensor): Query tensor of shape (L, N, E) where L is the target sequence length, N is the batch size
        key (Tensor): Key tensor of shape (S, N, E) where S is the source sequence length, N is the batch size
        value (Tensor): Value tensor of shape (S, N, E) where S is the source sequence length, N is the batch size
        embed_dim_to_check (int): The embedding dimension to check for compatibility
        num_heads (int): The number of attention heads
        num_dummies (Optional[int]): The number of dummy tokens
        add_zero_attn (bool): Whether to add a column of zeros to the attention weights
        dropout_p (float): The dropout probability
        out_proj_weight (Tensor): The weight tensor for the output projection
        out_proj_bias (Tensor): The bias tensor for the output projection
        out_dim (int): The output dimension.
        training (bool): Whether to use dropout. Defaults to True.
        key_padding_mask (Optional[Tensor]): Key padding mask of shape (N, S) where N is the batch size
        need_weights (bool): Whether to return attention weights. Defaults to True.
        attn_mask (Optional[Tensor]): Attention mask of shape (L, S) where L is the target sequence length
        dummy (bool): Whether to use dummy tokens. Defaults to True.

    Returns:
        Tuple[Tensor, Optional[Tensor]]: The output tensor and the attention weights
    """
    tgt_len, bsz, embed_dim = query.size()
    head_dim = embed_dim // num_heads
    v_head_dim = out_dim // num_heads

    assert embed_dim == embed_dim_to_check
    assert key.size(0) == value.size(0)
    assert key.size(1) == value.size(1)
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

    scaling = float(head_dim) ** -0.5
    q = query * scaling
    k = key
    v = value

    if attn_mask is not None:
        attn_mask = prepare_attention_mask(attn_mask, query, key, bsz, num_heads)

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        key_padding_mask = key_padding_mask.to(torch.bool)

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, v_head_dim).transpose(0, 1)

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))  # pylint: disable=not-callable
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))  # pylint: disable=not-callable

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float("-inf"))
        else:
            attn_output_weights = attn_output_weights + attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float("-inf"),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    if attn_output_weights.size(-1) != 0:
        attn_output_weights_max = torch.max(attn_output_weights, dim=-1, keepdim=True)[0]
        attn_output_weights = softmax(attn_output_weights - attn_output_weights_max, dim=-1)
    else:
        attn_output_weights = softmax(attn_output_weights, dim=-1)

    if saliency_scores is not None:
        scores = torch.repeat_interleave(saliency_scores, num_heads, dim=0)[:, :, None]
        attn_output_weights = attn_output_weights * scores

    attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)

    if dummy:
        attn_output = torch.bmm(attn_output_weights[:, :, num_dummies:], v[:, num_dummies:, :])
    else:
        attn_output = torch.bmm(attn_output_weights, v)

    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, v_head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, out_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)  # pylint: disable=not-callable

    if need_weights:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    return attn_output, None
