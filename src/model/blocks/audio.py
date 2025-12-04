"""Audio module."""

from typing import Any, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from src.model.blocks.feed_forward import FeedForwardNetwork
from src.model.blocks.layers import DropPath
from src.model.utils.stacker import get_clones

EPS: float = 1e-6
INIT_CONST: float = 0.01


class FFNFuser(nn.Module):
    """FFNDuser Encoder."""

    def __init__(self, dim: int):
        """Initialize FFNFuser.

        Args:
            dim (int): model dim.
        """
        super().__init__()
        self.dim = dim
        self.projector = nn.Linear(dim + dim, dim)
        self.output_norm = nn.LayerNorm(dim)  # noqa: WPS204

    def forward(self, audio: Tensor, video: Tensor, **_: Any) -> Tensor:
        """Forward pass of the FFNFuser.

        Args:
            audio (Tensor): audio embeddings.
            video (Tensor): video embeddings.
            _ (Any): other kwargs.

        Returns:
            Tensor: merged features.
        """
        concated_features = torch.cat((video, audio), dim=2)
        merged = self.projector(concated_features)
        return self.output_norm(merged)


class CrossAttentionLayer(nn.Module):  # noqa: WPS230
    """Cross Attnetion."""

    def __init__(
        self,
        dim: int,
        expansion_ratio: int = 4,
        nhead: int = 8,
        dropout: float = 0.1,
        droppath: float = 0.1,
    ):
        """
        Initialize LearnedAggregation.

        Args:
            dim (int): Dimensionality of the input embeddings.
            expansion_ratio (int): Expansion ratio for the feedforward network. Default is 4.
            nhead (int): Number of attention heads. Default is 8.
            dropout (float): Dropout rate. Default is 0.1.
            droppath (float): Droppath rate. Defaults to 0.1.
        """
        super().__init__()
        self.dim = dim
        self.nhead = nhead
        self.attn = nn.MultiheadAttention(dim, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)  # noqa: WPS204
        self.norm2 = nn.LayerNorm(dim)
        self.dropout1 = DropPath(droppath)  # noqa: WPS204
        self.dropout2 = DropPath(droppath)
        self.ffn = FeedForwardNetwork(dim, expansion_ratio, dropout)
        self.apply(self._init_weights)

    @torch.no_grad()
    def _init_weights(self, module):
        """
        Initialize weights of the module.

        Args:
            module (nn.Module): A submodule of LearnedAggregation.
        """
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=INIT_CONST)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(
        self,
        query_emb: Tensor,
        key_emb: Tensor,
        pos_emb: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """
        Forward pass of the CrossAttentionLayer module.

        Args:
            query_emb (Tensor): query feature seqs. (batch_size, seq_len, dim).
            key_emb (Tensor): key feature seqs. (batch_size, seq_len, dim).
            pos_emb (Tensor): positional embedding. (batch_size, seq_len, dim).
            mask (Tensor): mask of shape (batch_size, seq_len).

        Returns:
            Tensor: Output tensor of shape (seq_len, batch_size, dim).
        """
        # prepare mask
        mask = mask.to(torch.bool)

        # prepare queries
        query_emb_pos = query_emb + pos_emb
        query_emb_pos = query_emb_pos.transpose(0, 1)
        query_emb = query_emb.transpose(0, 1)

        # prepare keys and values
        key_emb_pos = key_emb + pos_emb
        key_emb_pos = key_emb_pos.transpose(0, 1)
        value_emb = key_emb.transpose(0, 1)

        attn_out, _ = self.attn(
            query_emb_pos,
            key_emb_pos,
            value_emb,
            key_padding_mask=~mask,
        )

        output = query_emb + self.dropout1(attn_out)
        output = self.norm1(output)
        ffn_output = self.ffn(output)
        output = output + self.dropout2(ffn_output)
        output = self.norm2(output)
        return output.transpose(0, 1)


class CrossAttentionEncoder(nn.Module):
    """Co-Attnetion Encoder."""

    def __init__(self, num_layers: int, dim: int, dropout: float, droppath: float) -> None:
        """Initialize CrossAttention Encoder.

        Args:
            num_layers (int): num layers.
            dim (int): model dim.
            dropout (float): Dropout rate.
            droppath (float): Droppath rate.
        """
        super().__init__()
        self.num_layers = num_layers

        # video to audio cross attention
        vid_to_aud_cross = CrossAttentionLayer(dim=dim, dropout=dropout, droppath=droppath)
        self.vid_to_aud_cross_layers = get_clones(vid_to_aud_cross, num_layers)

    def forward(
        self,
        vid_emb: Tensor,
        aud_emb: Tensor,
        pos_emb: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """
        Forward pass of the CrossAttention Encoder.

        Args:
            vid_emb (Tensor): video feature seqs. (batch_size, seq_len, dim).
            aud_emb (Tensor): audio feature seqs. (batch_size, seq_len, dim).
            pos_emb (Tensor): positional embedding. (batch_size, seq_len, dim).
            mask (Tensor): mask of shape (batch_size, seq_len).

        Returns:
            Tensor: Output tensor of shape (seq_len, batch_size, dim).
        """
        for vid_aud_layer in self.vid_to_aud_cross_layers:
            vid_emb = vid_aud_layer(
                query_emb=vid_emb,
                key_emb=aud_emb,
                pos_emb=pos_emb,
                mask=mask,
            )
        return vid_emb


class CoAttentionEncoder(nn.Module):
    """Co-Attnetion Encoder."""

    def __init__(self, num_layers: int, dim: int, dropout: float, droppath: float) -> None:
        """Initialize CoAttention Encoder.

        Args:
            num_layers (int): num layers.
            dim (int): model dim.
            dropout (float): Dropout rate.
            droppath (float): Droppath rate.
        """
        super().__init__()
        self.num_layers = num_layers

        self.mapper = nn.Sequential(nn.Linear(dim + dim, dim), nn.LayerNorm(dim))  # noqa: WPS221

        # video to audio cross attention
        vid_to_aud_cross = CrossAttentionLayer(dim=dim, dropout=dropout, droppath=droppath)
        self.vid_to_aud_cross_layers = get_clones(vid_to_aud_cross, num_layers)

        # audio to video cross attention
        aud_to_vid_cross = CrossAttentionLayer(dim=dim, dropout=dropout, droppath=droppath)
        self.aud_to_vid_cross_layers = get_clones(aud_to_vid_cross, num_layers)

    def forward(
        self,
        vid_emb: Tensor,
        aud_emb: Tensor,
        pos_emb: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """
        Forward pass of the CoAttention Encoder.

        Args:
            vid_emb (Tensor): video feature seqs. (batch_size, seq_len, dim).
            aud_emb (Tensor): audio feature seqs. (batch_size, seq_len, dim).
            pos_emb (Tensor): positional embedding. (batch_size, seq_len, dim).
            mask (Tensor): mask of shape (batch_size, seq_len).

        Returns:
            Tensor: Output tensor of shape (seq_len, batch_size, dim).
        """
        for aud_vid_layer, vid_aud_layer in zip(self.aud_to_vid_cross_layers, self.vid_to_aud_cross_layers):
            updated_audio = aud_vid_layer(
                query_emb=aud_emb,
                key_emb=vid_emb,
                pos_emb=pos_emb,
                mask=mask,
            )
            updated_video = vid_aud_layer(
                query_emb=vid_emb,
                key_emb=aud_emb,
                pos_emb=pos_emb,
                mask=mask,
            )
            aud_emb = updated_audio
            vid_emb = updated_video
        return self.mapper(torch.cat((vid_emb, aud_emb), dim=2))


class BottleneckLayer(nn.Module):  # noqa: WPS230
    """Bottleneck Layer."""

    def __init__(self, dim: int, nhead: int = 8, expansion_ratio: int = 4, dropout: float = 0.1, droppath: float = 0.1):
        """Initialize the Bottleneck Layer.

        Args:
            dim (int): model dim.
            nhead (int): number of attn head. Defaults to 8.
            expansion_ratio (int): ffn expansion ratio. Defaults to 4.
            dropout (float): dropout prob. Defaults to 0.1.
            droppath (float): droppath prob. Defaults to 0.1.
        """
        super().__init__()
        self.dims = dim
        self.nhead = nhead
        self.expansion_ratio = expansion_ratio
        self.dropout = dropout
        self.droppath = droppath

        self.att1 = nn.MultiheadAttention(dim, nhead, dropout=dropout)
        self.att2 = nn.MultiheadAttention(dim, nhead, dropout=dropout)
        self.att3 = nn.MultiheadAttention(dim, nhead, dropout=dropout)
        self.att4 = nn.MultiheadAttention(dim, nhead, dropout=dropout)

        self.ffn1 = FeedForwardNetwork(dim, expansion_ratio, dropout)
        self.ffn2 = FeedForwardNetwork(dim, expansion_ratio, dropout)

        self.droppath1 = DropPath(droppath)  # noqa: WPS204
        self.droppath2 = DropPath(droppath)
        self.droppath3 = DropPath(droppath)
        self.droppath4 = DropPath(droppath)
        self.droppath5 = DropPath(droppath)
        self.droppath6 = DropPath(droppath)

        self.norm1 = nn.LayerNorm(dim)  # noqa: WPS204
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.norm5 = nn.LayerNorm(dim)
        self.norm6 = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    @torch.no_grad()
    def _init_weights(self, module):
        """
        Initialize weights of the module.

        Args:
            module (nn.Module): A submodule of LearnedAggregation.
        """
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=INIT_CONST)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(
        self,
        video: Tensor,
        audio: Tensor,
        bottle_token: Tensor,
        pos_emb: Tensor,
        mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass of the BottleneckLayer.

        Args:
            video (Tensor): video embeddings.
            audio (Tensor): audio embeddings.
            bottle_token (Tensor): bottlneck tokens.
            pos_emb (Tensor): positioanl embedding.
            mask (Tensor): key padding mask

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Updated video, audio and token embeddings.
        """
        mask = mask.to(torch.bool)
        normed_video = self.norm1(video)
        normed_audio = self.norm2(audio)
        normed_bottle_token = self.norm3(bottle_token)

        # token to audio-video
        key_video = normed_video + pos_emb
        key_audio = normed_audio + pos_emb
        token_video_attn, _ = self.att1(normed_bottle_token, key_video, normed_video, key_padding_mask=~mask)
        token_audio_attn, _ = self.att2(normed_bottle_token, key_audio, normed_audio, key_padding_mask=~mask)
        bottle_token = bottle_token + self.droppath1(token_video_attn)
        bottle_token = bottle_token + self.droppath2(token_audio_attn)

        # audio-video to token attn
        normed_bottle_token = self.norm4(bottle_token)
        query_video = normed_video + pos_emb
        query_audio = normed_audio + pos_emb
        video_token_attn, _ = self.att3(query_video, normed_bottle_token, normed_bottle_token)
        audio_token_attn, _ = self.att4(query_audio, normed_bottle_token, normed_bottle_token)
        video = video + self.droppath3(video_token_attn)
        audio = audio + self.droppath4(audio_token_attn)

        # apply ffn
        normed_video = self.norm5(video)
        normed_audio = self.norm6(audio)
        video = video + self.droppath5(self.ffn1(normed_video))
        audio = audio + self.droppath6(self.ffn2(normed_audio))

        return audio, video, bottle_token


class BottleneckEncoder(nn.Module):
    """Bottleneck Encoder."""

    def __init__(self, num_layers: int, dim: int, num_tokens: int, dropout: float, droppath: float) -> None:
        """Initialize Bottleneck Encoder.

        Args:
            num_layers (int): num layers.
            dim (int): model dim.
            num_tokens (int): number of bottleneck tokens.
            dropout (float): Dropout rate.
            droppath (float): Droppath rate.
        """
        super().__init__()
        self.num_layers = num_layers
        self.token = nn.Embedding(num_tokens, dim)
        self.mapper = nn.Sequential(nn.Linear(dim + dim, dim), nn.LayerNorm(dim))  # noqa: WPS221
        bottle_layer = BottleneckLayer(dim=dim, dropout=dropout, droppath=droppath)
        self.layers = get_clones(bottle_layer, num_layers)

    def forward(
        self,
        vid_emb: Tensor,
        aud_emb: Tensor,
        pos_emb: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """
        Forward pass of the CoAttention Encoder.

        Args:
            vid_emb (Tensor): video feature seqs. (batch_size, seq_len, dim).
            aud_emb (Tensor): audio feature seqs. (batch_size, seq_len, dim).
            pos_emb (Tensor): positional embedding. (batch_size, seq_len, dim).
            mask (Tensor): seq mask of shape (batch_size, seq_len).

        Returns:
            Tensor: Output tensor of shape (seq_len, batch_size, dim).
        """
        token = self.token.weight.expand(vid_emb.size(0), -1, -1)
        token = token.transpose(0, 1)
        vid_emb = vid_emb.transpose(0, 1)
        aud_emb = aud_emb.transpose(0, 1)
        pos_emb = pos_emb.transpose(0, 1)
        for layer in self.layers:
            vid_emb, aud_emb, token = layer(vid_emb, aud_emb, token, pos_emb, mask)
        projected_emb = self.mapper(torch.cat((vid_emb, aud_emb), dim=2))
        return projected_emb.transpose(0, 1)


class AudioMerger(nn.Module):
    """Class to merge video and audio features."""

    def __init__(  # noqa: WPS231
        self,
        num_layers: int,
        merge_type: Optional[str],
        num_tokens: int,
        model_dim: int,
        dropout: float,
        droppath: float,
    ) -> None:
        """Initialize AudioMerger class.

        Args:
            num_layers (int): num layers to use.
            merge_type (str): one of {concat, crossattn, coatt}
            num_tokens (int): number of bottleneck tokens to use.
            model_dim (int): model dim.
            dropout (float): Dropout rate.
            droppath (float): Droppath rate.

        Raises:
            NotImplementedError: merge_type should be one of {concat, crossattn, coatt}
        """
        super().__init__()
        self.dropout = dropout
        self.droppath = droppath
        self.merge_type = merge_type
        self.num_layers = num_layers
        if self.merge_type == "concat":  # noqa: WPS223
            self.audio_fuser: Optional[
                Union[FFNFuser, CrossAttentionEncoder, CoAttentionEncoder, BottleneckEncoder]
            ] = FFNFuser(model_dim)
        elif self.merge_type == "crossattn":

            self.audio_fuser = CrossAttentionEncoder(
                num_layers=num_layers,
                dim=model_dim,
                dropout=dropout,
                droppath=droppath,
            )
        elif self.merge_type == "coattn":
            self.audio_fuser = CoAttentionEncoder(
                num_layers=num_layers,
                dim=model_dim,
                dropout=dropout,
                droppath=droppath,
            )
        elif self.merge_type == "bottleneck":
            self.audio_fuser = BottleneckEncoder(
                num_layers=num_layers,
                dim=model_dim,
                num_tokens=num_tokens,
                dropout=dropout,
                droppath=droppath,
            )
        elif self.merge_type is None:
            self.audio_fuser = None
        else:
            raise NotImplementedError("merge_type should be one of {concat, crossattn, coatt, bottleneck}")

    def forward(
        self,
        audio: Tensor,
        video: Tensor,
        pos_emb: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Forward pass of the audio merger.

        Args:
            audio (Tensor): audio features
            video (Tensor): video features
            pos_emb (Tensor): pos embedding
            mask (Tensor): key padding mask

        Returns:
            Tensor: Merged features
        """
        if self.audio_fuser is not None:
            return self.audio_fuser(video, audio, pos_emb=pos_emb, mask=mask)
        return video
