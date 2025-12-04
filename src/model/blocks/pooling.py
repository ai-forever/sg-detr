"""Pooling modules."""

import torch
from torch import Tensor, nn

from src.model.blocks.attention import DABMultiheadAttention
from src.model.blocks.feed_forward import FeedForwardNetwork
from src.model.utils.stacker import get_clones

EPS: float = 1e-6
INIT_CONST: float = 0.02
GAMMA_CONST: float = 1e-4  # 1  #  A constant for the residual connection for pooling, since the query vector is
# initially random, it is better to add aggregation completely.


class AttentionPool2d(nn.Module):
    """Attention for Learned Aggregation."""

    def __init__(self, dim: int, bias: bool = True):
        """Initialize AttentionPool2d.

        Args:
            dim (int): Dimensionality of the input embeddings.
            bias (bool): If True, adds a learnable bias to the linear transformations. Default is True.
        """
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.vk_proj = nn.Linear(dim, dim * 2, bias=bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, emb: Tensor, cls_q: Tensor) -> Tensor:
        """
        Forward pass of the AttentionPool2d module.

        Args:
            emb (Tensor): Input embeddings of shape (batch_size, seq_len, dim).
            cls_q (Tensor): Query tensor of shape (batch_size, dim).

        Returns:
            Tensor: Projected and pooled output tensor of shape (batch_size, dim).
        """
        emb = emb.transpose(1, 2)  # swap seq_len and dim
        batch_size, seq_len, dim = emb.shape

        # Compute query vector
        query = self.q_proj(cls_q.expand(batch_size, -1, -1))
        # Compute key and value vectors
        key_value = self.vk_proj(emb).reshape(batch_size, seq_len, 2, dim)
        key, value = key_value.permute(2, 0, 1, 3).chunk(2, 0)

        # Compute attention scores
        attn = torch.matmul(query, key.transpose(-2, -1))
        attn = torch.softmax(attn, dim=-1)
        # Compute weighted sum of values
        emb = torch.matmul(attn, value).transpose(1, 2).reshape(batch_size, dim)

        # Project and return pooled output
        return self.proj(emb)


class LearnedAggregationLayer(nn.Module):
    """Learned Aggregation from https://arxiv.org/abs/2112.13692."""

    def __init__(
        self,
        dim: int,
        expansion_ratio: int = 4,
        nhead: int = 8,
        dropout: float = 0.1,
        use_projections: bool = False,
        use_gamma: bool = True,
    ):
        """
        Initialize LearnedAggregationLayer.

        Args:
            dim (int): Dimensionality of the input embeddings.
            expansion_ratio (int): Expansion ratio for the feedforward network. Default is 4.
            nhead (int): Number of attention heads. Default is 8.
            dropout (float): Dropout rate. Default is 0.1.
            use_projections (bool): Whether to use projections. Default is False.
        """
        super().__init__()
        self.use_gamma = use_gamma
        if use_gamma:
            self.gamma_1 = nn.Parameter(GAMMA_CONST * torch.ones(dim))  # noqa: WPS114
            self.gamma_2 = nn.Parameter(GAMMA_CONST * torch.ones(dim))  # noqa: WPS114
        else:
            self.gamma_1, self.gamma_2 = 1.0, 1.0  # type: ignore

        self.attn = DABMultiheadAttention(dim, nhead)
        self.attn_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForwardNetwork(dim, expansion_ratio, dropout)
        self.use_projections = use_projections
        if use_projections:
            self.k_proj = nn.Linear(dim, dim)
            self.v_proj = nn.Linear(dim, dim)

        self.apply(self._init_weights)

    def forward(self, emb: Tensor, query: Tensor, key_padding_mask: Tensor) -> Tensor:
        """
        Forward pass of the LearnedAggregationLayer module.

        Args:
            emb (Tensor): Input embeddings of shape (batch_size, seq_len, dim).
            query (Tensor): Content embeddings of shape (1, batch_size, dim).
            key_padding_mask (Tensor): Mask tensor of shape (batch_size, seq_len).

        Returns:
            Tensor: Output tensor of shape (1, batch_size, dim).
        """
        emb = emb.transpose(0, 1)
        emb = self.attn_norm(emb)
        if self.use_projections:
            k_emb = self.k_proj(emb)
            v_emb = self.v_proj(emb)
        else:
            k_emb, v_emb = emb, emb

        key_padding_mask = key_padding_mask.to(torch.bool)
        attn_out, _ = self.attn(query, k_emb, v_emb, key_padding_mask=~key_padding_mask, dummy=False)
        output = query + self.gamma_1 * attn_out
        ffn_output = self.ffn(self.ffn_norm(output))
        output = output + self.gamma_2 * ffn_output
        return output

    @torch.no_grad()
    def _init_weights(self, module):
        """
        Initialize weights of the module.

        Args:
            module (nn.Module): A submodule of LearnedAggregationLayer.
        """
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=INIT_CONST)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


class LearnedAggregation(nn.Module):
    """Learned Aggregation from https://arxiv.org/abs/2112.13692."""

    def __init__(self, dim, aggregation_layer: LearnedAggregationLayer, num_layers: int):
        """Initialize a LearnedAggregation module.

        Args:
            dim (int): Dimensionality of the input embeddings.
            aggregation_layer (LearnedAggregationLayer): An instance of the transformer aggregation layer.
            num_layers (int): The number of layers in the learned aggregation module.
        """
        super().__init__()
        self.layers = get_clones(aggregation_layer, num_layers)
        self.num_layers = num_layers
        self.cls_q = nn.Parameter(torch.zeros(dim))
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.cls_q, std=INIT_CONST)

    def forward(self, emb: Tensor, key_padding_mask: Tensor) -> Tensor:
        """
        Forward pass of the LearnedAggregationLay module.

        Args:
            emb (Tensor): Input embeddings of shape (batch_size, seq_len, dim).
            key_padding_mask (Tensor): Mask tensor of shape (batch_size, seq_len).

        Returns:
            Tensor: Output tensor of shape (batch_size, 1, dim).
        """
        # (1, batch_size, 256)
        content = self.cls_q[None, None].repeat(1, emb.size(0), 1)
        output = content
        layer_n = 0
        for layer in self.layers:
            output = layer(
                emb=emb,
                query=output,
                key_padding_mask=key_padding_mask,
            )
            if layer_n < self.num_layers - 1:
                output = self.norm(output)
            layer_n += 1
        return output.transpose(0, 1)


class GRUFeatureExtractor(nn.Module):
    """GRU-based Feature Extractor."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, bidirectional: bool = False) -> None:
        """Initialize GRUFeatureExtractor.

        Args:
            input_dim (int): Dimensionality of the input.
            hidden_dim (int): Dimensionality of the hidden state.
            num_layers (int): Number of recurrent layers. Default is 1.
            bidirectional (bool): If True, use bidirectional GRU. Default is False.
        """
        super().__init__()
        self.gru = nn.GRU(  # type: ignore
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(self, emb: Tensor) -> Tensor:
        """
        Forward pass of the GRUFeatureExtractor.

        Args:
            emb (Tensor): Input tensor of shape (batch_size, seq_length, input_dim).

        Returns:
            Tensor: Extracted features (hidden state)
        """
        # emb shape: (batch_size, seq_length, input_dim)
        _, hidden = self.gru(emb)

        # If bidirectional, concatenate the forward and backward hidden states
        if self.gru.bidirectional:
            # Combine the hidden states of both directions (forward and backward)
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)  # noqa:WPS221
        else:
            # If not bidirectional, use the hidden state of the last layer
            hidden = hidden[-1, :, :]

        # Output the hidden state(s) as the global feature(s)
        # For a single layer, hidden shape: (batch_size, hidden_dim)
        # For multiple layers, hidden shape: (batch_size, num_layers * hidden_dim)
        return hidden


class GlobalMaxPooling(nn.Module):
    """Max Pooling Layer."""

    def forward(self, extracted_intervals: Tensor, masks: Tensor) -> Tensor:
        """
        Perform forward pass of the GlobalMaxPooling layer.

        Args:
            extracted_intervals (Tensor): The input tensor with shape [batch_size, num_intervals, model_dim].
            masks (Tensor): The mask tensor with shape [batch_size, num_intervals].

        Returns:
            Tensor: The tensor after applying global max pooling with shape [batch_size, 1, model_dim].
        """
        # Apply mask to the extracted_intervals
        masked_intervals = extracted_intervals * masks.unsqueeze(-1).float()
        # Set masked elements to negative infinity
        masked_intervals[masks == 0] = float("-inf")
        # Calculate the max pooling
        max_intervals, _ = masked_intervals.max(dim=1)  # [batch_size, model_dim]
        # Add an extra dimension to match the desired output shape
        return max_intervals.unsqueeze(1)  # [batch_size, 1, model_dim]


class GlobalMeanPooling(nn.Module):
    """Mean Pooling Layer."""

    def forward(self, extracted_intervals: Tensor, masks: Tensor) -> Tensor:
        """
        Perform forward pass of the GlobalMeanPooling layer.

        Args:
            extracted_intervals (Tensor): The input tensor with shape [batch_size, num_intervals, model_dim].
            masks (Tensor): The mask tensor with shape [batch_size, num_intervals].

        Returns:
            Tensor: The tensor after applying global mean pooling with shape [batch_size, 1, model_dim].
        """
        # Apply mask to the extracted_intervals
        masked_intervals = extracted_intervals * masks.unsqueeze(-1).float()

        # Calculate the sum and count for mean pooling
        sum_intervals = masked_intervals.sum(dim=1)  # [batch_size, model_dim]
        count_intervals = masks.sum(dim=1).unsqueeze(-1).float()  # [batch_size, 1]

        # Avoid division by zero
        mean_intervals = sum_intervals / (count_intervals + EPS)  # [batch_size, model_dim]

        # Add an extra dimension to match the desired output shape
        return mean_intervals.unsqueeze(1)  # [batch_size, 1, model_dim]
