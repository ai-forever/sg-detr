"""Module for the transformer layers."""

from typing import Optional

import torch
from torch import Tensor, nn

from src.model.blocks.attention import DABMultiheadAttention
from src.model.blocks.feed_forward import FeedForwardNetwork


class Scale(nn.Module):
    """Scale distinguisher."""

    def __init__(self, init_value: float = 1.0) -> None:
        """Initialize Scale.

        Args:
            init_value (float): Init value. Defaults to 1.0.
        """
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, emb: Tensor) -> Tensor:
        """Forward pass of the module.

        Args:
            emb (Tensor): Features from the certain scale.

        Returns:
            Tensor: Adjusted scale features.
        """
        return emb * self.scale


class DropPath(nn.Module):
    """Drop paths per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.1) -> None:
        """Initialize the DropPath module.

        Args:
            drop_prob (float): The dropout probability. Defaults to 0.1.
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, path_value: Tensor) -> Tensor:
        """Forward pass of the DropPath module.

        Args:
            path_value (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying drop path.
        """
        if not self.training:
            return path_value

        path_value = path_value.permute(1, 0, 2)
        keep_prob = 1 - self.drop_prob
        shape = (path_value.shape[0],) + (1,) * (path_value.ndim - 1)
        mask = keep_prob + torch.rand(shape, dtype=path_value.dtype, device=path_value.device)
        mask.floor_()
        path_value = path_value.div(keep_prob) * mask
        return path_value.permute(1, 0, 2)


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer."""

    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        expansion_ratio: int = 4,
        dropout: float = 0.1,
        droppath: float = 0.1,
    ):
        """Initialize the TransformerEncoderLayer.

        Args:
            d_model (int): The dimension of the input feature.
            nhead (int): The number of heads in the multihead attention.
            expansion_ratio (int): The expansion ratio for the hidden layer dimension of FFN. Defaults to 4.
            dropout (float): Dropout rate. Defaults to 0.1.
            droppath (float): Droppath rate. Defaults to 0.1.
        """
        super().__init__()
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self._init_parameters()

        self.ffn = FeedForwardNetwork(d_model, expansion_ratio, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = DropPath(droppath)
        self.dropout2 = DropPath(droppath)

    def _init_parameters(self) -> None:
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(
        self,
        embedding: Tensor,
        position_embedding: Tensor,
        embedding_mask: Optional[Tensor] = None,
        embedding_key_padding_mask: Optional[Tensor] = None,
    ):
        """Forward pass of the TransformerEncoderLayer.

        Args:
            embedding (Tensor): The input tensor.
            position_embedding (Tensor): The positional embedding for the input tensor.
            embedding_mask (Optional[Tensor]): The mask for the input tensor. Defaults to None.
            embedding_key_padding_mask (Optional[Tensor]):  The key padding mask for the input tensor. Defaults to None.

        Returns:
            _type_: _description_
        """
        query = embedding + position_embedding
        key = query

        # Attention part
        att_out, _ = self.self_attn(
            query=query,
            key=key,
            value=embedding,
            attn_mask=embedding_mask,
            key_padding_mask=embedding_key_padding_mask,
        )
        embedding = embedding + self.dropout1(att_out)
        embedding = self.norm1(embedding)

        # FFN part
        ffn_out = self.ffn(embedding)
        embedding = embedding + self.dropout2(ffn_out)
        return self.norm2(embedding)


class T2VTransformerEncoderLayer(nn.Module):
    """Text to video transformer encoder layer."""

    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        expansion_ratio: int = 4,
        dropout: float = 0.1,
        droppath: float = 0.1,
        num_dummies: int = 35,
        use_cross_attn_wo_dummy: bool = False,
        weight_attn_with_saliency: bool = False,
    ) -> None:
        """Initialize the T2V_TransformerEncoderLayer.

        Args:
            d_model (int): The dimension of the input feature.
            nhead (int): The number of heads in the multihead attention.
            expansion_ratio (int): The expansion ratio for the hidden layer dimension of FFN. Defaults to 4.
            dropout (float): Dropout rate. Defaults to 0.1.
            droppath (float): Droppath rate. Defaults to 0.1.
            num_dummies (int): The number of dummy tokens. Defaults to 35.
        """
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.use_cross_attn_wo_dummy = use_cross_attn_wo_dummy
        self.weight_attn_with_saliency = weight_attn_with_saliency

        if use_cross_attn_wo_dummy:
            self.self_attn = DABMultiheadAttention(d_model, nhead, dropout_prob=dropout)
        else:
            self.self_attn = DABMultiheadAttention(d_model, nhead, dropout_prob=dropout, num_dummies=num_dummies)

        self._init_parameters()

        self.ffn = FeedForwardNetwork(d_model, expansion_ratio, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = DropPath(droppath)
        self.dropout2 = DropPath(droppath)

    def _init_parameters(self) -> None:
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(
        self,
        emb: Tensor,
        pos_embed: Tensor,
        video_length: int,
        emb_key_padding_mask: Optional[Tensor] = None,
        dummy: bool = True,
        saliency_scores: Optional[Tensor] = None,
    ):
        """Forward pass of the T2VTransformerEncoderLayer.

        Args:
            emb (Tensor): The input tensor.
            pos_embed (Tensor): The positional embedding for the input tensor.
            video_length (int): The length of the video (#clips).
            emb_key_padding_mask (Optional[Tensor]): The key padding mask for the input tensor. Defaults to None.
            dummy (bool): Whether to use dummy tokens. Defaults to True.

        Returns:
            Tuple[Tensor, Tensor]: The output tensor and the attention weights.
        """
        pos_emb = emb + pos_embed
        query = pos_emb[:video_length]
        key = pos_emb[video_length:]
        value = emb[video_length:]

        attn_mask = None
        if emb_key_padding_mask is not None:
            query_mask = emb_key_padding_mask[:, :video_length].unsqueeze(2)
            key_mask = emb_key_padding_mask[:, video_length:].unsqueeze(1)
            attn_mask = torch.matmul(query_mask.float(), key_mask.float())
            attn_mask = torch.repeat_interleave(attn_mask.bool(), self.nhead, dim=0)
            emb_key_padding_mask = emb_key_padding_mask[:, video_length:]

        if self.use_cross_attn_wo_dummy:
            kwargs = {'saliency_scores': saliency_scores} if self.weight_attn_with_saliency else {}
            attn_out, attn_weights = self.self_attn(
                query,
                key,
                value=value,
                attn_mask=attn_mask,
                key_padding_mask=emb_key_padding_mask,
                dummy=False,
                **kwargs,
            )
            attn_weights = None
        else:
            attn_out, attn_weights = self.self_attn(
                query,
                key,
                value,
                attn_mask=attn_mask,
                key_padding_mask=emb_key_padding_mask,
                dummy=dummy,
            )

        emb_aug = emb[:video_length] + self.dropout1(attn_out)
        ffn_out = self.norm1(self.ffn(emb_aug))
        emb_aug = emb_aug + self.dropout2(ffn_out)
        emb_aug = self.norm2(emb_aug)
        src = torch.cat([emb_aug, emb[video_length:]])
        return src, attn_weights


class DecoderSelfAttention(nn.Module):
    """Self-attention layer for the decoder."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
        droppath: float = 0.1,
    ):
        """Initialize the DecoderSelfAttention.

        Args:
            d_model (int): The dimension of the input feature.
            nhead (int): The number of heads in the multihead attention.
            dropout (float): Dropout rate. Defaults to 0.1.
            droppath (float): Droppath rate. Defaults to 0.1.
        """
        super().__init__()
        self.query_content_proj = nn.Linear(d_model, d_model)
        self.query_pos_proj = nn.Linear(d_model, d_model)
        self.key_content_proj = nn.Linear(d_model, d_model)
        self.key_pos_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        self.self_attn = DABMultiheadAttention(d_model, nhead, dropout_prob=dropout, vdim=d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = DropPath(droppath)

    def forward(
        self,
        embedding: Tensor,
        embedding_mask: Optional[Tensor] = None,
        embedding_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass of the DecoderSelfAttention.

        Args:
            embedding (Tensor): The input tensor.
            embedding_mask (Optional[Tensor]): The mask for the input tensor. Defaults to None.
            embedding_key_padding_mask (Optional[Tensor]): The key padding mask for the input tensor. Defaults to None.
            query_pos (Optional[Tensor]): The positional embedding for the query.

        Returns:
            Tensor: The output tensor after applying self-attention.
        """
        # Apply projections here
        # shape: num_queries x batch_size x d_model
        query_content = self.query_content_proj(embedding)  # embedding is the input of the first decoder layer.
        query_pos = self.query_pos_proj(query_pos)
        key_content = self.key_content_proj(embedding)
        key_pos = self.key_pos_proj(query_pos)
        value = self.value_proj(embedding)

        query = query_content + query_pos
        key = key_content + key_pos

        att_out, _ = self.self_attn(
            query=query,
            key=key,
            value=value,
            attn_mask=embedding_mask,
            key_padding_mask=embedding_key_padding_mask,
            dummy=False,
        )

        embedding = embedding + self.dropout1(att_out)
        return self.norm1(embedding)


# pylint: disable=too-many-arguments,too-many-locals
class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer."""

    def __init__(
        self,
        d_model: int,
        cont_pos_tradeoff: int,
        nhead: int = 8,
        expansion_ratio: int = 4,
        dropout: float = 0.1,
        droppath: float = 0.1,
        rm_self_attn_decoder: bool = False,
    ) -> None:
        """
        Initialize the TransformerDecoderLayer.

        Args:
            d_model (int): The dimension of the input feature.
            cont_pos_tradeoff (int): Offset for the dim content/position dims.
            nhead (int): The number of heads in the multihead attention.
            expansion_ratio (int): The expansion ratio for the hidden layer dimension of FFN. Defaults to 4.
            dropout (float): Dropout rate. Defaults to 0.1.
            droppath (float): Droppath rate. Defaults to 0.1.
            rm_self_attn_decoder (bool): Whether to remove the self-attention layer in the decoder. Defaults to False.
        """
        super().__init__()
        self.nhead = nhead
        self.cont_pos_tradeoff = cont_pos_tradeoff
        self.rm_self_attn_decoder = rm_self_attn_decoder
        assert cont_pos_tradeoff % nhead == 0, "cont_pos_tradeoff should be divisible by nhead"

        # Decoder Self-Attention
        if not rm_self_attn_decoder:
            self.self_attention = DecoderSelfAttention(d_model, nhead, dropout=dropout, droppath=droppath)

        # Decoder Cross-Attention
        # query related mappers
        self.ca_qcontent_proj = nn.Linear(d_model, d_model + cont_pos_tradeoff)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model - cont_pos_tradeoff)

        # key related mappers
        self.ca_kcontent_proj = nn.Linear(d_model, d_model + cont_pos_tradeoff)
        self.ca_kpos_proj = nn.Linear(d_model, d_model - cont_pos_tradeoff)

        # value mapper
        self.ca_value_proj = nn.Linear(d_model, d_model)

        # cross attention module
        self.cross_attn = DABMultiheadAttention(d_model * 2, nhead, dropout_prob=dropout, vdim=d_model)

        # init params
        self._init_parameters()

        # Feedforward model
        self.ffn = FeedForwardNetwork(d_model, expansion_ratio, dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = DropPath(droppath)
        self.dropout3 = DropPath(droppath)

    def _init_parameters(self) -> None:
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(  # noqa: WPS211
        self,
        tgt: Tensor,
        src: Tensor,
        query_sine_embed: Tensor,
        query_pos: Tensor,
        src_pos: Tensor,
        tgt_mask: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ):
        """Forward pass of the TransformerDecoderLayer.

        Args:
            tgt (Tensor): Decoder content embedding from previous layer. Shape: [#Queries, batch_size, dim]
            src (Tensor): video content embedding from the encoder. Shape: [Lv, batch_size, dim]
            query_sine_embed (Tensor): The positional embedding for the query (tgt). Shape: [#Queries, batch_size, dim]
            query_pos (Tensor): The positional embedding for the query (Used in SelfAttnetion block).
            src_pos (Tensor): The positional embedding for the video.
            tgt_mask (Optional[Tensor]): The mask for the target tensor.
            src_mask (Optional[Tensor]): The mask for the video tensor.
            tgt_key_padding_mask (Optional[Tensor]): The key padding mask for the target tensor.
            src_key_padding_mask (Optional[Tensor]): The key padding mask for the video tensor.

        Returns:
            Tensor: The output tensor after applying the decoder layer.
        """
        # ========== Begin of Self-Attention =============
        if not self.rm_self_attn_decoder:
            tgt = self.self_attention(tgt, tgt_mask, tgt_key_padding_mask, query_pos)

        # ========== Begin of Cross-Attention =============
        # query related. shape: num_queries x batch_size x 256
        query = self.ca_qcontent_proj(tgt)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)

        # video related features
        key = self.ca_kcontent_proj(src)
        key_pos = self.ca_kpos_proj(src_pos)
        value = self.ca_value_proj(src)

        num_queries, batch_size, content_dim = query.shape
        _, _, pos_dim = query_sine_embed.shape
        length, _, _ = key.shape

        # concat query content and positional info
        query = query.view(num_queries, batch_size, self.nhead, content_dim // self.nhead)
        query_sine_embed = query_sine_embed.view(num_queries, batch_size, self.nhead, pos_dim // self.nhead)
        query = torch.cat([query, query_sine_embed], dim=3)
        query = query.view(num_queries, batch_size, content_dim + pos_dim)

        # concat video content and PE embedding
        key = key.view(length, batch_size, self.nhead, content_dim // self.nhead)
        key_pos = key_pos.view(length, batch_size, self.nhead, pos_dim // self.nhead)
        key = torch.cat([key, key_pos], dim=3)
        key = key.view(length, batch_size, content_dim + pos_dim)

        tgt2, _ = self.cross_attn(
            query=query,
            key=key,
            value=value,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            dummy=False,
        )
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.ffn(tgt)
        tgt = tgt + self.dropout3(tgt2)
        return self.norm3(tgt)
