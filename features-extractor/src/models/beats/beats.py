"""BEATs feature extraction."""

from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.nn import LayerNorm
from torchaudio.compliance import kaldi as ta_kaldi

from src.models.beats.backbone import TransformerEncoder

N_MELS: int = 128
SAMPLE_FREQ: int = 16000
FRAME_LENGTH: int = 25
FRAME_SHIFT: int = 10


def prepare_audio_features(
    source: torch.Tensor,
    fbank_mean: float = 15.41663,
    fbank_std: float = 6.55582,
) -> torch.Tensor:
    """
    Prepare audio features by applying a ta_kaldi and normalizing them.

    Args:
        audio_tensor (Tensor): The input tensor containing the raw audio waveform.
        sample_rate (int): The sampling rate of the audio tensor. Defaults to 16000 Hz.

    Returns:
        Tensor: A tensor representing the log Mel spectrogram of the input audio. This output includes
                an additional dimension as a batch dimension, making it suitable for batch processing
                in machine learning models.
    """
    fbanks = []
    for waveform in source:
        waveform = waveform.unsqueeze(0) * 2**15
        fbank = ta_kaldi.fbank(
            waveform,
            num_mel_bins=N_MELS,
            sample_frequency=SAMPLE_FREQ,
            frame_length=FRAME_LENGTH,
            frame_shift=FRAME_SHIFT,
        )
        fbanks.append(fbank)
    fbank = torch.stack(fbanks, dim=0)
    return (fbank - fbank_mean) / (2 * fbank_std)


class BEATsConfig:
    """BEATs config."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        """Configuration class for BEATs model.

        Args:
            cfg (Optional[Dict[str, Any]]): Configuration dictionary to update default settings. Defaults to None.
        """
        self.input_patch_size: int = -1  # path size of patch embedding
        self.embed_dim: int = 512  # patch embedding dimension
        self.conv_bias: bool = False  # include bias in conv encoder

        self.encoder_layers: int = 12  # num encoder layers in the transformer
        self.encoder_embed_dim: int = 768  # encoder embedding dimension
        self.encoder_ffn_embed_dim: int = 3072  # encoder embedding dimension for FFN
        self.encoder_attention_heads: int = 12  # num encoder attention heads
        self.activation_fn: str = "gelu"  # activation function to use

        self.layer_wise_gradient_decay_ratio: float = 1.0  # ratio for layer-wise gradient decay
        self.layer_norm_first: bool = False  # apply layernorm first in the transformer
        self.deep_norm: bool = False  # apply deep_norm first in the transformer

        # dropouts
        self.dropout: float = 0.1  # dropout probability for the transformer
        self.attention_dropout: float = 0.1  # dropout probability for attention weights
        self.activation_dropout: float = 0.0  # dropout probability after activation in FFN
        self.encoder_layerdrop: float = 0.0  # probability of dropping a tarnsformer layer
        self.dropout_input: float = 0.0  # dropout to apply to the input (after feat extr)

        # positional embeddings
        self.conv_pos: int = 128  # number of filters for convolutional positional embeddings
        self.conv_pos_groups: int = 16  # number of groups for convolutional positional embedding

        # relative position embedding
        self.relative_position_embedding: bool = False  # apply relative position embedding
        self.num_buckets: int = 320  # number of buckets for relative position embedding
        self.max_distance: int = 1280  # maximum distance for relative position embedding
        self.gru_rel_pos: bool = False  # apply gated relative position embedding

        # label predictor
        self.finetuned_model: bool = False  # whether the model is a fine-tuned model.
        self.predictor_dropout: float = 0.1  # dropout probability for the predictor
        self.predictor_class: int = 527  # target class number for the predictor

        if cfg is not None:
            self.update(cfg)

    def update(self, cfg: Dict[str, Any]):
        """Update configuration parameters.

        Args:
            cfg (Dict): Configuration dictionary to update settings.
        """
        self.__dict__.update(cfg)


class BEATs(nn.Module):
    """BEATs model class."""

    def __init__(self, cfg: BEATsConfig) -> None:
        """BEATs model initialization.

        Args:
            cfg (BEATsConfig): Configuration object for BEATs model.
        """
        super().__init__()
        self.cfg = cfg

        self.post_extract_proj = nn.Linear(cfg.embed_dim, cfg.encoder_embed_dim)

        self.input_patch_size = cfg.input_patch_size
        self.patch_embedding = nn.Conv2d(
            1,
            cfg.embed_dim,
            kernel_size=self.input_patch_size,
            stride=self.input_patch_size,
            bias=cfg.conv_bias,
        )

        self.dropout_input = nn.Dropout(cfg.dropout_input)

        assert not cfg.deep_norm or not cfg.layer_norm_first
        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(cfg.embed_dim)

    def forward_padding_mask(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Generate padding mask for input features.

        Args:
            features (torch.Tensor): Input feature tensor.
            padding_mask (torch.Tensor): Padding mask tensor.

        Returns:
            torch.Tensor: Updated padding mask.
        """
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        return padding_mask.all(-1)

    def forward(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        fbank_mean: float = 15.41663,
        fbank_std: float = 6.55582,
    ) -> torch.Tensor:
        """Forward pass for the BEATs model.

        Args:
            source (torch.Tensor): Input tensor.
            padding_mask (Optional[torch.Tensor]): Padding mask tensor. Defaults to None.
            fbank_mean (float): Mean value for feature normalization. Defaults to 15.41663.
            fbank_std (float): Standard deviation for feature normalization. Defaults to 6.55582.

        Returns:
            torch.Tensor: Model output tensor.
        """
        fbank = prepare_audio_features(source, fbank_mean=fbank_mean, fbank_std=fbank_std)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(fbank, padding_mask)

        fbank = fbank.unsqueeze(1)
        features = self.patch_embedding(fbank)
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        features = self.post_extract_proj(features)
        x = self.dropout_input(features)
        x, _ = self.encoder(x, padding_mask=padding_mask)
        return x.mean(dim=1)
