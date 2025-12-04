"""Module for Transformer Encoders."""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import functional as func

from src.model.blocks.layers import T2VTransformerEncoderLayer, TransformerEncoderLayer
from src.model.blocks.pooling import (
    GlobalMaxPooling,
    GlobalMeanPooling,
    GRUFeatureExtractor,
    LearnedAggregation,
    LearnedAggregationLayer,
)
from src.model.utils.schemas import MomentEncoderOutput, SentenceEncoderOutput
from src.model.utils.stacker import get_clones
from src.utils.span_utils import span_cxw_to_xx

TEMP: float = 3.0
A_PARAM: float = 0.0  # noqa: WPS358
B_PARAM: float = 20.0


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder class that stacks multiple encoder layers.

    Attributes:
        num_layers (int): Number of encoder layers.
        layers (nn.ModuleList): List of duplicated encoder layers.
        return_intermediate (bool): Whether to return intermediate outputs from each layer.
    """

    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer,
        num_layers: int,
        return_intermediate: bool = False,
    ) -> None:
        """Initialize TransformerEncoder.

        Args:
            encoder_layer (TransformerEncoderLayer): An instance of the TransformerEncoderLayer to be duplicated.
            num_layers (int): Number of layers to be stacked.
            return_intermediate (bool): If set to True, the encoder will return all intermediate representations.
        """
        super().__init__()
        self.num_layers = num_layers
        self.layers = get_clones(encoder_layer, num_layers)
        self.return_intermediate = return_intermediate

    def forward(
        self,
        src: Tensor,
        src_pos: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Union[Tensor, List[Tensor]]:
        """
        Pass the input through the encoder layers in turn.

        Args:
            src (Tensor): The sequence to the encoder (required).
            src_pos (Tensor): The position of the sequence (required).
            mask (Optional[Tensor]): The mask for the src sequence (optional).
            src_key_padding_mask (Optional[Tensor]): The mask for the src keys per batch (optional).

        Returns:
            Union[Tensor, List[Tensor]]: Output of the last layer or intermediate outputs from all layers.
        """
        output = src

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                src_pos,
                embedding_mask=mask,
                embedding_key_padding_mask=src_key_padding_mask,
            )

            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerCATEEncoder(nn.Module):
    """
    A Transformer CATE Encoder module.

    This module applies a series of transformer encoder layers to the input data.

    Attributes:
        layers (nn.ModuleList): A list of identical transformer encoder layers.
        num_layers (int): The number of encoder layers.
        norm (nn.LayerNorm): A layer normalization module.
        return_intermediate (bool): If set to True, intermediate outputs of each layer will be returned.
    """

    def __init__(self, encoder_layer: T2VTransformerEncoderLayer, num_layers: int, return_intermediate: bool = False):
        """Initialize a TransformerCATEEncoder.

        Args:
            encoder_layer (T2VTransformerEncoderLayer): An instance of the transformer encoder layer.
            num_layers (int): The number of layers in the encoder.
            return_intermediate (bool): Whether to return intermediate outputs. Defaults to False.
        """
        super().__init__()
        self.layers = get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(encoder_layer.d_model)
        self.return_intermediate = return_intermediate

    def forward(
        self,
        src: Tensor,
        pos_embed: Tensor,
        video_length: int,
        src_key_padding_mask: Optional[Tensor] = None,
        dummy: bool = True,
        saliency_scores: Optional[Tensor] = None,
    ):
        """
        Pass the input (and mask) through each layer in turn.

        Args:
            src (Tensor): The sequence to the encoder (required). Shape: [Lv + Ld + Lt, Bs, dim]
            pos_embed (Tensor): The position of the sequence (required). Shape: [Lv + Ld + Lt, Bs, dim]
            video_length (int): The length of the video (#clips).
            src_key_padding_mask (Optional[Tensor]): The mask for the src keys per batch.  Shape: [Bs, Lv + Ld + Lt]
            dummy (bool): Whether to use dummy tokens.

        Returns:
            Tensor: The encoded output.
            Tensor: The attention weights.
        """
        output = src
        intermediate = []
        attn_weights = None
        for layer in self.layers:
            output, attn_weight = layer(
                output,
                pos_embed,
                video_length,
                emb_key_padding_mask=src_key_padding_mask,
                dummy=dummy,
                saliency_scores=saliency_scores,
            )
            output = self.norm(output)

            attn_weights = attn_weight if attn_weights is None else attn_weights + attn_weight

            if self.return_intermediate:
                intermediate.append(output)

        if attn_weights is not None:
            attn_weights /= self.num_layers  # type: ignore

        if self.norm is not None:
            output = self.norm(output)

        if self.return_intermediate:
            return torch.stack(intermediate), attn_weights

        return output, attn_weights


class DummyEncoder(nn.Module):
    """Encoder for dummy tokens."""

    def __init__(self, d_model: int, num_dummies: int, num_dummy_layers: int, dropout: float, droppath: float) -> None:
        """Initialize DummyEncoder.

        Args:
            d_model (int): dimension of the model
            num_dummies (int): number of dummy tokens
            num_dummy_layers (int): number of dummy layers
            dropout (float): dropout rate
            droppath (float): droppath rate
        """
        super().__init__()
        self.d_model = d_model
        self.num_dummies = num_dummies

        # define dummy tokens
        self.dummy_rep_token = torch.nn.Parameter(torch.randn(num_dummies, d_model))
        self.dummy_rep_pos = torch.nn.Parameter(torch.randn(num_dummies, d_model))

        # define self attention to set query-dummy tokens relations
        input_txt_sa_proj = TransformerEncoderLayer(d_model, dropout=dropout, droppath=droppath)
        self.txtproj_encoder = TransformerEncoder(input_txt_sa_proj, num_dummy_layers)

    def prepare_dummy_tokens(self, src_txt: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Prepare aux entities.

        Args:
            src_txt (Tensor): text features, (batch_size, L_txt, D_txt)

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - token: token repeated for batch_size times
                - mask: mask for token
                - position: position embedding for token repeated for batch_size times
        """
        # dummy or sentence tokens
        dummy_tokens = self.dummy_rep_token[None].repeat(src_txt.shape[0], 1, 1)

        # dummy or sentence mask
        dummy_mask = torch.tensor([True for _ in range(self.num_dummies)])
        dummy_mask = dummy_mask.to(src_txt.device)
        dummy_mask = dummy_mask.repeat(src_txt.shape[0], 1)

        # dummy positions
        dummy_position = self.dummy_rep_pos[None].repeat(src_txt.shape[0], 1, 1)
        return dummy_tokens, dummy_mask, dummy_position

    def forward(self, src_txt: Tensor, src_txt_mask: Tensor, pos_txt: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass.

        Create dummy tokens with query excluded content and concat them with text features.

        Args:
            src_txt (Tensor): intital text features, (batch_size, L_txt, D_txt)
            src_txt_mask (Tensor): mask for text features, (batch_size, L_txt)
            pos_txt (Tensor): position embedding for text features, (batch_size, L_txt, D_txt)

        Returns:
            Tuple[Tensor, Tensor, Tensor]: text, mask and position tensors concated with dummy tokens
        """
        # get dummy tokens
        dummy_tokens, dummy_mask, dummy_pos = self.prepare_dummy_tokens(src_txt)

        dummy_src_txt = torch.cat([dummy_tokens, src_txt], dim=1)
        dummy_src_txt_mask = torch.cat([dummy_mask, src_txt_mask], dim=1)
        dummy_pos_txt = torch.cat([dummy_pos, pos_txt], dim=1)

        # force dummy tokens to contain query excluding content
        dummy_src_txt = dummy_src_txt.permute(1, 0, 2)  # (L, batch_size, d)
        dummy_pos_txt = dummy_pos_txt.permute(1, 0, 2)  # (L, batch_size, d)
        memory = self.txtproj_encoder(
            dummy_src_txt,
            dummy_pos_txt,
            src_key_padding_mask=~(dummy_src_txt_mask.bool()),  # Should be True for padding tokens
        )  # (L, batch_size, d_model)

        # concat inhanced dummy tokens with text embeddings
        dummy_token = memory[: self.num_dummies].permute(1, 0, 2)  # (batch_size, L, d)
        dummy_src_txt = torch.cat([dummy_token, src_txt], dim=1)
        dummy_pos_txt = dummy_pos_txt.permute(1, 0, 2)  # (batch_size, L, d)

        return dummy_src_txt, dummy_src_txt_mask, dummy_pos_txt


class SentenceEncoder(nn.Module):
    """Encoder for sentence tokens."""

    def __init__(self, d_model: int, num_sentence_layers: int, dropout: float, droppath: float) -> None:
        """Initialize SentenceEncoder.

        Args:
            d_model (int): dimension of the model
            num_sentence_layers (int): number of sentence layers
            dropout (float): dropout rate
            droppath (float): droppath rate
        """
        super().__init__()
        self.d_model = d_model

        self.sent_rep_token = torch.nn.Parameter(torch.randn(1, d_model))
        self.sent_rep_pos = torch.nn.Parameter(torch.randn(1, d_model))

        # define sentence encoder to incorporate sentence representation
        scls_encoder_layer = TransformerEncoderLayer(d_model, dropout=dropout, droppath=droppath)
        self.sent_cls_encoder = TransformerEncoder(scls_encoder_layer, num_sentence_layers)

    def prepare_sentence_tokens(self, src_txt: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Prepare sentence tokens.

        Args:
            src_txt (Tensor): text features, (batch_size, L_txt, D_txt).

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - token: token repeated for batch_size times
                - mask: mask for token
                - position: position embedding for token repeated for batch_size times
        """
        # dummy or sentence tokens
        sent_rep_token = self.sent_rep_token[None].repeat(src_txt.shape[0], 1, 1)

        # dummy or sentence mask
        mask = torch.tensor([[True]])
        mask = mask.to(src_txt.device)
        mask = mask.repeat(src_txt.shape[0], 1)

        # dummy positions
        position = self.sent_rep_pos[None].repeat(src_txt.shape[0], 1, 1)
        return sent_rep_token, mask, position

    def run_encoder(self, src: Tensor, mask: Tensor, pos: Tensor) -> Tuple[Tensor, Tensor]:
        """Run self attention based encoder.

        Args:
            src (Tensor): input features, (batch_size, L_txt, D_txt)
            mask (Tensor): mask for input features, (batch_size, L_txt)
            pos (Tensor): position embedding for input features, (batch_size, L_txt, D_txt)

        Returns:
            Tuple[Tensor, Tensor]: token and memory tensors
        """
        src = src.permute(1, 0, 2)  # (L, batch_size, d)
        pos = pos.permute(1, 0, 2)  # (L, batch_size, d)
        memory = self.sent_cls_encoder(src, pos, src_key_padding_mask=~mask)  # True for padding tokens
        return memory[0], memory[1:]

    # pylint: disable=too-many-locals,too-many-arguments
    def forward(
        self,
        src_txt: Tensor,
        src_txt_mask: Tensor,
        pos_txt: Tensor,
        dummy_token: Optional[Tensor],
        dummy_mask: Optional[Tensor],
        dummy_pos: Optional[Tensor],
    ) -> SentenceEncoderOutput:
        """Forward pass.

        Args:
            src_txt (Tensor): intital text features, (batch_size, L_txt, D_txt)
            src_txt_mask (Tensor): mask for text features, (batch_size, L_txt)
            pos_txt (Tensor): position embedding for text features, (batch_size, L_txt, D_txt)
            dummy_token (Optional[Tensor]): dummy tokens, (batch_size, L_txt, D_txt)
            dummy_pos (Optional[Tensor]): dummy positions, (batch_size, L_txt, D_txt)
            dummy_mask (Optional[Tensor]): dummy mask, (batch_size, L_txt)

        Returns:
            SentenceEncoderOutput:
                - sent_txt_token: sentence tokens inhanced with query representation
                - sent_dummy_token: sentence tokens inhanced with dummy representation
                - sent_words_memory: memory of words inhanced with query representation
                - sent_dummy_memory: memory of words inhanced with dummy representation
        """
        sent_token, sent_mask, sent_pos = self.prepare_sentence_tokens(src_txt)

        # concat sentence token and text embeddings
        sent_txt = torch.cat([sent_token, src_txt], dim=1)
        sent_txt_mask = torch.cat([sent_mask, src_txt_mask.bool()], dim=1)
        sent_txt_pos = torch.cat([sent_pos, pos_txt], dim=1)

        # inhace sentence tokens with query representation
        sent_txt_token, sent_words_memory = self.run_encoder(sent_txt, sent_txt_mask, sent_txt_pos)

        if dummy_token is not None:
            assert dummy_mask is not None
            assert dummy_pos is not None
            # concat sentence token and dummy embeddings
            sent_dummy = torch.cat([sent_token, dummy_token], dim=1)
            sent_dummy_mask = torch.cat([sent_mask, dummy_mask.bool()], dim=1)
            sent_dummy_pos = torch.cat([sent_pos, dummy_pos], dim=1)

            # inhace sentence tokens with dummy representation (query excluding content)
            sent_dummy_token, sent_dummy_memory = self.run_encoder(sent_dummy, sent_dummy_mask, sent_dummy_pos)
        else:
            sent_dummy_token, sent_dummy_memory = None, None

        return SentenceEncoderOutput(
            sent_txt_token=sent_txt_token,
            sent_dummy_token=sent_dummy_token,
            sent_words_memory=sent_words_memory,
            sent_dummy_memory=sent_dummy_memory,
        )


class MomentEncoder(nn.Module):
    """Encoder for moment tokens."""

    def __init__(self, d_model: int, num_mcls_layers: int, dropout: float, droppath: float) -> None:
        """Initialize MomentEncoder.

        Args:
            d_model (int): dimension of the model
            num_mcls_layers (int): number of moment layers
            dropout (float): dropout rate
            droppath (float): droppath rate
        """
        super().__init__()
        # define moment tokens
        self.moment_rep_token = torch.nn.Parameter(torch.randn(1, d_model))
        self.moment_rep_pos = torch.nn.Parameter(torch.randn(1, d_model))

        # define moment encoder to incorporate moment representation
        mcls_encoder_layer = TransformerEncoderLayer(d_model, dropout=dropout, droppath=droppath)
        self.mcls_encoder = TransformerEncoder(mcls_encoder_layer, num_mcls_layers)

    def prepare_moment_tokens(
        self,
        src_vid: Tensor,
        pos_vid: Tensor,
        src_vid_mask: Tensor,
        targets: dict,
        is_positive: bool,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Prepare moment tokens.

        We will use them in SA block. The idea is to inhance the moment and non moment tokens. That is why we create
        masks based on clip relevance or irrelevance.

        Args:
            src_vid (Tensor): video features, (batch_size, L_vid, D_vid)
            pos_vid (Tensor): positional embedding for video
            src_vid_mask (Tensor): video mask, (batch_size, L_vid)
            targets (dict): targets dict
            is_positive (bool): True if we want to inhance moment tokens, False for non moment tokens

        Returns:
            Tuple[Tensor, Tensor, Tensor]: clip_mask, moment+vid emb, moment+vid position, moment+vid mask
        """
        # add moment token to video mask
        mom_mask = torch.tensor([[True]]).to(src_vid_mask.device)
        mom_mask = mom_mask.repeat(src_vid_mask.shape[0], 1)
        mom_vid_mask = torch.cat([mom_mask, src_vid_mask.bool()], dim=1)

        # add moment token to clips mask
        clips_mask = torch.clamp(targets["relevant_clips"], 0, 1).bool()
        clips_mask = clips_mask if is_positive else ~clips_mask
        clips_mask = clips_mask * src_vid_mask.bool()
        mom_clips_mask = torch.cat([mom_mask, clips_mask], dim=1)

        # ignore relevant or irrelevant clips based on 'is_positive' flag
        mom_vid_mask = mom_vid_mask * mom_clips_mask

        # Concat moment and vid embeddings
        moment_token = self.moment_rep_token[None].repeat(src_vid.shape[0], 1, 1)
        mom_vid_emb = torch.cat([moment_token, src_vid], dim=1)

        # concat moment and vid positions
        moment_token_pos = self.moment_rep_pos[None].repeat(pos_vid.shape[0], 1, 1)
        mom_vid_pos = torch.cat([moment_token_pos, pos_vid], dim=1)
        return clips_mask, mom_vid_emb, mom_vid_pos, mom_vid_mask

    # pylint: disable=too-many-locals
    def forward(
        self,
        src_vid: Tensor,
        src_vid_mask: Tensor,
        pos_vid: Tensor,
        targets: Dict[str, Any],
    ) -> MomentEncoderOutput:
        """Forward pass of the moment encoder.

        Args:
            src_vid (Tensor): video features, (batch_size, L_vid, D_vid)
            src_vid_mask (Tensor): video mask, (batch_size, L_vid)
            pos_vid (Tensor): positional embedding for video
            targets (Dict[str, Any]): targets dict contains 'relevant_clips' key

        Returns:
            MomentEncoderOutput: output of the encoder schema with the following attributes:
                - rel_clips_mask (Tensor): Mask for relevant clips
                - irrel_clips_mask (Tensor): Mask for irrelevant clips
                - moment_token (Tensor): Moment token inhanced with relevant clips representation
                - moment_memory (Tensor): Relevant clips representation (output from SA encoder)
                - non_moment_token (Tensor): Non-moment token inhanced with irrelevant clips representation
                - non_moment_memory (Tensor): Irrelevant clips representation (output from SA encoder)
        """
        rel_clips_mask, mom_vid_emb, mom_vid_pos, mom_vid_mask = self.prepare_moment_tokens(
            src_vid,
            pos_vid,
            src_vid_mask,
            targets,
            is_positive=True,
        )
        irrel_clips_mask, non_mom_vid_emb, non_mom_vid_pos, non_mom_vid_mask = self.prepare_moment_tokens(
            src_vid,
            pos_vid,
            src_vid_mask,
            targets,
            is_positive=False,
        )

        # moment token
        mom_vid_emb = mom_vid_emb.permute(1, 0, 2)  # (L, batch_size, dim)
        mom_vid_pos = mom_vid_pos.permute(1, 0, 2)  # (L, batch_size, dim)
        mmemory = self.mcls_encoder(mom_vid_emb, mom_vid_pos, src_key_padding_mask=~mom_vid_mask)  # True for padding
        moment_token, moment_memory = mmemory[0], mmemory[1:]

        non_mom_vid_emb = non_mom_vid_emb.permute(1, 0, 2)  # (L, batch_size, dim)
        non_mom_vid_pos = non_mom_vid_pos.permute(1, 0, 2)  # (L, batch_size, dim)
        nmmemory = self.mcls_encoder(non_mom_vid_emb, non_mom_vid_pos, src_key_padding_mask=~non_mom_vid_mask)
        non_moment_token, non_moment_memory = nmmemory[0], nmmemory[1:]

        return MomentEncoderOutput(
            relevant_clips_mask=rel_clips_mask,
            irrelevant_clips_mask=irrel_clips_mask,
            moment_token=moment_token,
            moment_memory=moment_memory,
            non_moment_token=non_moment_token,
            non_moment_memory=non_moment_memory,
        )


class Text2VisionEncoder(nn.Module):
    """Encoder to incorporate text representation into video."""

    def __init__(
        self,
        d_model: int,
        num_dummies: int,
        num_t2v_layers: int,
        dropout: float,
        droppath: float,
        remove_dummy: bool = True,
        use_cross_attn_wo_dummy: bool = False,
        weight_attn_with_saliency: bool = False,
    ) -> None:
        """Initialize Text2VisionEncoder.

        Args:
            d_model (int): dimension of the model
            num_dummies (int): number of dummy tokens
            num_t2v_layers (int): number of encoder layers
            dropout (float): dropout rate
            droppath (float): droppath rate
            remove_dummy (bool): wther to remove dummy or not.
        """
        super().__init__()
        self.d_model = d_model
        self.use_cross_attn_wo_dummy = use_cross_attn_wo_dummy
        self.num_dummies = 0 if use_cross_attn_wo_dummy else num_dummies

        # define text to vision encoder to incorporate text representation into video
        t2v_encoder_layer = T2VTransformerEncoderLayer(
            d_model,
            num_dummies=self.num_dummies,
            dropout=dropout,
            droppath=droppath,
            use_cross_attn_wo_dummy=use_cross_attn_wo_dummy,
            weight_attn_with_saliency=weight_attn_with_saliency,
        )
        self.t2v_encoder = TransformerCATEEncoder(t2v_encoder_layer, num_t2v_layers)
        self.remove_dummy = remove_dummy

    def forward(
        self,
        src: Tensor,
        mask: Tensor,
        pos: Tensor,
        batch_video_len: int,
        saliency_scores: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
        """Forward pass of the Text2VisionEncoder.

        Args:
            src (Tensor): video features, (batch_size, L_vid + L_dummy + L_txt, D_vid)
            mask (Tensor): video mask, (batch_size, L_vid + L_dummy + L_txt)
            pos (Tensor): position embedding for video, (batch_size, L_vid + L_dummy + L_txt, D_vid)
            batch_video_len (int): batch video length

        Returns:
            Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
                - video tensor concated with saliency token
                - mask tensor concated with saliency token
                - position tensors concated with saliency token
                - attention weights from video to text transformer
        """
        src = src.permute(1, 0, 2)  # (Lv + Ld + Lt, batch_size, dim)
        pos = pos.permute(1, 0, 2)  # (Lv + Ld + Lt, batch_size, dim)

        t2v_src, attn_weights = self.t2v_encoder(
            src,
            pos,
            video_length=batch_video_len,
            src_key_padding_mask=~mask,  # Should be True for padding tokens
            dummy=self.remove_dummy,
            saliency_scores=saliency_scores,
        )  # (L, batch_size, dim)
        return t2v_src, mask, pos, attn_weights  # type: ignore


class LocalSaliencyHead(nn.Module):
    """
    A neural network module to compute local saliency scores for video and text embeddings.

    Attributes:
        logit_mode (str): The mode for calculating logits. Must be one of {"linear", "exp", "exp_b"}.
        sentence_pooling (LearnedAggregation): An instance of LearnedAggregation for sentence pooling.
        temp (nn.Parameter): A temperature parameter used in "exp" and "exp_b" logit modes.
        a (nn.Parameter): A parameter used in "exp_b" and "linear" logit modes.
        b (nn.Parameter): A parameter used in "linear" logit mode.
    """

    allowed_logit_modes = {"linear", "exp", "exp_b"}

    def __init__(
        self,
        model_dim: int,
        use_projections: bool = True,
        logit_mode: str = "linear",
        use_gamma: bool = True,
        num_aggregation_layers: int = 1,
    ) -> None:
        """
        Initialize the LocalSaliencyHead module.

        Args:
            model_dim (int): Dimension of the model.
            use_projections (bool): Whether to use projections in LearnedAggregation. Defaults to True.
            logit_mode (str): Mode for calculating logits. Must be one of {"linear", "exp", "exp_b"}.
        """
        super().__init__()
        self._validate_logit_mode(logit_mode)
        self.logit_mode = logit_mode
        sentence_pooling_layer = LearnedAggregationLayer(
            dim=model_dim, use_projections=use_projections, use_gamma=use_gamma
        )
        self.sentence_pooling = LearnedAggregation(model_dim, sentence_pooling_layer, num_aggregation_layers)
        self._initialize_parameters()

    def _validate_logit_mode(self, logit_mode: str) -> None:
        """
        Validate the logit_mode.

        Args:
            logit_mode (str): The mode for calculating logits.

        Raises:
            ValueError: If logit_mode is not one of {"linear", "exp", "exp_b"}.
        """
        if logit_mode not in self.allowed_logit_modes:
            raise ValueError(f"logit_mode must be one of {self.allowed_logit_modes}")

    def _initialize_parameters(self) -> None:
        """Initialize the parameters based on the logit_mode."""
        self.temp: Optional[nn.Parameter] = None
        self.a_param: Optional[nn.Parameter] = None
        self.b_param: Optional[nn.Parameter] = None
        if self.logit_mode == "exp":
            self.temp = nn.Parameter(torch.tensor(TEMP))
        elif self.logit_mode == "exp_b":
            self.temp = nn.Parameter(torch.tensor(TEMP))
            self.a_param = nn.Parameter(torch.tensor(A_PARAM))
        else:
            self.a_param = nn.Parameter(torch.tensor(A_PARAM))
            self.b_param = nn.Parameter(torch.tensor(B_PARAM))

    def saliency_scores(self, vid_emb: Tensor, txt_emb: Tensor) -> Tensor:
        """
        Compute saliency scores for video and text embeddings.

        Args:
            vid_emb (Tensor): The video embeddings.
            txt_emb (Tensor): The text embeddings.

        Returns:
            Tensor: The computed saliency scores.
        """
        txt_emb = self._normalize_embedding(txt_emb)
        vid_emb = self._normalize_embedding(vid_emb)
        scores = torch.sum(vid_emb * txt_emb, dim=-1)
        return self._apply_logit_mode(scores)

    def _normalize_embedding(self, emb: Tensor) -> Tensor:
        """
        Normalize the embeddings.

        Args:
            emb (Tensor): The embeddings to normalize.

        Returns:
            Tensor: The normalized embeddings.
        """
        return nn.functional.normalize(emb, p=2, dim=-1)

    def _apply_logit_mode(self, scores: Tensor) -> Tensor:
        """
        Apply the logit mode to the scores.

        Args:
            scores (Tensor): The computed scores.

        Returns:
            Tensor: The scores after applying the logit mode.
        """
        if self.logit_mode == "exp":
            return scores * torch.exp(self.temp)  # type: ignore
        if self.logit_mode == "exp_b":
            return scores * torch.exp(self.temp) + self.a_param  # type: ignore
        return scores * self.b_param + self.a_param

    def forward(self, src_vid: Tensor, src_txt: Tensor, src_txt_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the LocalSaliencyHead module.

        Args:
            src_vid (Tensor): Source video embeddings.
            src_txt (Tensor): Source text embeddings.
            src_txt_mask (Tensor): Source text mask.

        Returns:
            Tuple[Tensor, Tensor]: The saliency scores and the pooled sentence embeddings.
        """
        src_sent = self.sentence_pooling(src_txt, key_padding_mask=src_txt_mask)
        saliency_scores = self.saliency_scores(src_vid, src_sent)
        return saliency_scores, src_sent


class SaliencyAmplifier(nn.Module):
    """
    A neural network module to amplify features based on saliency scores.

    Attributes:
        alpha (nn.Parameter): A learnable parameter to scale the new features.
        mode (str): The mode for calculating saliency. Must be one of {"sigmoid", "softmax"}.
        use_mha (bool): Whether to use multi-head attention (MHA).
        mha (TransformerEncoderLayer): An instance of TransformerEncoderLayer for multi-head attention.
    """

    allowed_modes = {"sigmoid", "softmax", "sin"}

    def __init__(
        self,
        d_model: int,
        mode: str = "sigmoid",
        use_mha: bool = True,
        use_norm: bool = False,
        temperature: int = 10000,
    ) -> None:
        """
        Initialize the SaliencyAmplifier module.

        Args:
            d_model (int): Dimension of the model.
            mode (str): Mode for calculating saliency. Must be one of {"sigmoid", "softmax"}. Defaults to "sigmoid".
            use_mha (bool): Whether to use multi-head attention (MHA). Defaults to True.
            use_norm (bool):  Whether to use norm or not. Defaults to False.
            temperature (int): temp for sin emb.

        Raises:
            ValueError: If mode is not one of {"sigmoid", "softmax"}.
        """
        super().__init__()
        if mode not in self.allowed_modes:
            raise ValueError(f"mode must be one of {self.allowed_modes}")

        self.mode = mode
        self.d_model = d_model
        self.use_mha = use_mha
        self.use_norm = use_norm
        self.temperature = temperature

        self.norm = nn.LayerNorm(d_model) if use_norm else None
        self.mha = TransformerEncoderLayer(d_model=d_model) if use_mha else None

    def gen_sin_emb(self, scores: Tensor) -> Tensor:
        """Generate sine embeddings for scores.

        Args:
            scores (Tensor): local scores. Shape: [seq_len, batch_size]

        Returns:
            Tensor: sine embeddings for local scores
        """
        scale = 2 * math.pi
        dim_t = torch.arange(self.d_model, dtype=torch.float32, device=scores.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.d_model)

        scaled_scores = scores * scale
        sin_emb = scaled_scores[:, :, None] / dim_t
        return torch.stack(  # noqa: WPS317
            (
                sin_emb[:, :, 0::2].sin(),
                sin_emb[:, :, 1::2].cos(),
            ),
            dim=3,
        ).flatten(2)

    # pylint: disable=not-callable
    def forward(
        self,
        features: Tensor,
        saliency_scores: Tensor,
        pos: Tensor,
        vid_mask: Tensor,
    ) -> Tensor:
        """
        Forward pass of the SaliencyAmplifier module.

        Args:
            features (Tensor): The input features.
            saliency_scores (Tensor): The saliency scores.
            pos (Tensor): Positional encodings.
            vid_mask (Tensor): The mask for the video sequence. Defaults to None.

        Returns:
            Tensor: The output features after saliency amplification.
        """
        if self.mode == "sin":
            saliency = saliency_scores.transpose(0, 1).detach()
            saliency_sine_embed = self.gen_sin_emb(saliency.sigmoid())
            output = features + saliency_sine_embed
        else:
            if self.mode == "sigmoid":
                saliency = torch.sigmoid(saliency_scores.transpose(0, 1))  # [seq_length, batch_size]
            elif self.mode == "softmax":
                masked_saliency_scores = saliency_scores.masked_fill(~vid_mask, float("-inf"))
                saliency = torch.softmax(masked_saliency_scores, dim=1)  # Check temperature
                saliency = saliency.transpose(0, 1)

            new_features = features * saliency[:, :, None]  # [seq_length, batch_size, model_dim]
            output = features + new_features

        if self.use_norm:
            assert self.norm is not None
            output = self.norm(output)

        if self.use_mha:
            assert self.mha is not None
            output = self.mha(output, pos)

        return output


class MR2HD(nn.Module):
    """
    A neural network module for moment retrieval to highlight detection amplification.

    Attributes:
        score_mode (str): The mode for calculating scores. Must be one of {"probs", "iou", "probs_iou"}.
        aggregation_mode (str): The mode for aggregating features. One of {"attention", "gru", "none", "mean", "max"}.
        learned_aggregation (Optional[nn.Module]): An instance of a learned aggregation module or None.
        linear (nn.Linear): A linear layer for generating logits.
        temperature (nn.Parameter): A temperature parameter for scaling the scores.
    """

    def __init__(
        self,
        model_dim: int,
        aggregation_mode: str = "attention",
        score_mode: str = "probs",
    ) -> None:
        """
        Initialize the MR2HD module.

        Args:
            model_dim (int): Dimension of the model.
            aggregation_mode (str): Mode for aggregating features. One of {"attention", "gru", "none", "mean", "max"}.
            score_mode (str): Mode for calculating scores. One of {"probs", "iou", "probs_iou"}.
        """
        super().__init__()
        # check score and aggregation modes
        assert aggregation_mode in {"attention", "gru", "none", "mean", "max"}
        assert score_mode in {"probs", "iou", "probs_iou"}
        self.score_mode = score_mode
        self.aggregation_mode = aggregation_mode
        self.temperature = nn.Parameter(torch.tensor(TEMP))

        # init aggregation class
        if aggregation_mode == "attention":
            self.learned_aggregation: Optional[nn.Module] = LearnedAggregation(
                model_dim, LearnedAggregationLayer(model_dim), 1
            )
        elif aggregation_mode == "gru":
            self.learned_aggregation = GRUFeatureExtractor(model_dim, model_dim)
        elif aggregation_mode == "none":
            self.learned_aggregation = None
        elif aggregation_mode == "mean":
            self.learned_aggregation = GlobalMeanPooling()
        else:
            self.learned_aggregation = GlobalMaxPooling()

    def extract_intervals(self, src_vid: Tensor, predicted_spans_idxes: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Extract specified intervals from a batch of video sequences and return these intervals along with masks.

        Args:
            src_vid (Tensor): A tensor of shape (bs, seq_length, model_dim) representing the input video sequences.
            predicted_spans_idxes (Tensor): represents [bs, num_spans, 2] tensor for the intervals to be extracted.

        Returns:
            Tuple[Tensor, Tensor]:
                - extracted_intervals (Tensor): A tensor of shape (batch_size, num_spans, seq_length, model_dim)
                    containing the extracted intervals. Each interval is padded to match the seq_length.
                - masks (Tensor): A tensor of shape (batch_size, num_spans, seq_length) containing the masks
                    corresponding to the extracted intervals, 1 indicates the interval and 0 indicates the padding.
        """
        batch_size, seq_length, model_dim = src_vid.shape
        _, num_spans, _ = predicted_spans_idxes.shape

        # Create a list to store the extracted intervals for each batch
        extracted_intervals = []
        masks = []

        for batch_idx in range(batch_size):
            batch_intervals = []
            batch_masks = []
            for span_idx in range(num_spans):
                start_idx, end_idx = predicted_spans_idxes[batch_idx, span_idx].tolist()

                # Correct intervals where start_idx equals end_idx
                if start_idx == end_idx and end_idx < seq_length:
                    end_idx += 1
                if start_idx == end_idx and end_idx == seq_length:
                    start_idx -= 1

                # Ensure indices are within valid range
                start_idx = max(0, start_idx)
                end_idx = min(seq_length, end_idx)

                # Extract the interval
                length = end_idx - start_idx
                mask = [1] * length + [0] * (seq_length - length)

                mask = torch.tensor(mask).bool().to(src_vid.device)  # type: ignore
                interval = src_vid[batch_idx, start_idx:end_idx]
                padding = torch.zeros(seq_length - length, model_dim).to(src_vid.device)
                interval = torch.concat([interval, padding])  # [seq_length, model_dim]
                batch_intervals.append(interval)
                batch_masks.append(mask)

            extracted_intervals.append(torch.stack(batch_intervals))
            masks.append(torch.stack(batch_masks))  # type: ignore
        return torch.stack(extracted_intervals), torch.stack(masks)

    def forward(
        self,
        video_features: Tensor,
        outputs_class: Tensor,
        quality_scores: Tensor,
        outputs_coord: Tensor,
        video_text_features: Tensor,
    ) -> Tensor:
        """
        Forward pass of the MR2HD class.

        Args:
            video_features (Tensor): shape [bs, seq_length, model_dim], video features before cross-attention with text.
            outputs_class (Tensor): shape [bs, num_queries, 2], class logits (0 idx is positive class).
            outputs_coord (Tensor): shape [bs, num_queries, 2], spans in normalized [0, 1] format (center and width).
            quality_scores (Tensor): shape [bs, num_queries], quality scores for each query.
            video_text_features (Tensor): shape [seq_length, bs, model_dim], video features after cross-attn with text.

        Returns:
            Tensor: updated saliency scores, shape [bs, seq_length]
        """
        if self.learned_aggregation is None:
            return video_text_features.transpose(0, 1)

        batch_size, video_length, _ = video_features.shape
        scores = torch.sigmoid(outputs_class)[..., 0]
        outputs_coord_xx = span_cxw_to_xx(outputs_coord)  # [batch_size, num_queries, 2]
        predicted_spans_idxes = torch.round(outputs_coord_xx * video_length).to(int)  # type: ignore
        # predicted_spans_idxes shape: [batch_size, num_queries, 2]

        extracted_intervals, masks = self.extract_intervals(video_features, predicted_spans_idxes)
        list_of_seqs: List[Tensor] = []
        for batch_idx in range(batch_size):
            batch_intervals = extracted_intervals[batch_idx]
            batch_masks = masks[batch_idx]

            # agregate features inside interval
            if self.aggregation_mode in {"attention", "max", "mean"}:
                aggregation_span_vectors = self.learned_aggregation(batch_intervals, batch_masks)[:, 0, :]
            else:  # only for test ancient evil
                span_vectors = [
                    self.learned_aggregation(interval[mask][None, :, :])[0]
                    for interval, mask in zip(batch_intervals, batch_masks)
                ]
                aggregation_span_vectors = torch.stack(span_vectors)

            # get interval scores
            if self.score_mode == "probs":
                item_scores = scores[batch_idx]
            elif self.score_mode == "iou":
                item_scores = quality_scores[batch_idx]
            else:
                item_scores = torch.sqrt(scores[batch_idx] * quality_scores[batch_idx])

            # agregate span features
            mult = torch.softmax(item_scores * torch.exp(self.temperature), dim=0)  # [seq_length]
            vercor_of_seq = torch.sum(aggregation_span_vectors * mult[:, None], dim=0)  # [model_dim]
            list_of_seqs.append(vercor_of_seq)

        interval_features = torch.stack(list_of_seqs).unsqueeze(1)  # [batch_size, 1, model_dim]
        cosine_similarities = func.cosine_similarity(video_features, interval_features, dim=-1)  # pylint: disable=E1102

        video_text_features = video_text_features.transpose(0, 1)
        cosine_similarities = cosine_similarities[:, :, None]
        return video_text_features + video_text_features * cosine_similarities


class CrossAttentionWithProbs(nn.Module):
    def __init__(self, model_dim: int, d_k: int, d_v: int) -> None:
        super(CrossAttentionWithProbs, self).__init__()
        self.w_query = nn.Linear(model_dim, d_k, bias=False)
        self.w_key = nn.Linear(model_dim, d_k, bias=False)
        self.w_value = nn.Linear(model_dim, d_v, bias=False)
        self.temperature = nn.Parameter(torch.tensor(3.0))

    def forward(self, aggregation_span_vectors: Tensor, video_text_features: Tensor, probs: Tensor) -> Tensor:
        # Conversion to Q, K, V
        query = self.w_query(video_text_features)  # shape: [batch_size, seq_length, d_k]
        key = self.w_key(aggregation_span_vectors)  # shape: [batch_size, n_spans, d_k]
        value = self.w_value(aggregation_span_vectors)  # shape: [batch_size, n_spans, d_v]

        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / torch.sqrt(torch.tensor(query.size(-1), dtype=torch.float32))
        # scores shape: [batch_size, seq_length, n_spans]

        scores_weighted = scores * torch.softmax(probs / self.temperature, dim=1).unsqueeze(1)
        # scores_weighted shape: [batch_size, seq_length, n_spans]

        attention_weights = F.softmax(scores_weighted, dim=-1)  # shape: [batch_size, seq_length, n_spans]

        output = torch.matmul(attention_weights, value)  # shape: [batch_size, seq_length, d_v]

        return output


class MR2HD_V2(nn.Module):
    """
    A neural network module for moment retrieval to highlight detection amplification.

    Attributes:
        score_mode (str): The mode for calculating scores. Must be one of {"probs", "iou", "probs_iou"}.
        aggregation_mode (str): The mode for aggregating features. One of {"attention", "gru", "none", "mean", "max"}.
        learned_aggregation (Optional[nn.Module]): An instance of a learned aggregation module or None.
        linear (nn.Linear): A linear layer for generating logits.
        temperature (nn.Parameter): A temperature parameter for scaling the scores.
    """

    def __init__(self, model_dim: int, aggregation_mode: str = "attention", score_mode: str = "probs") -> None:
        """
        Initialize the MR2HD module.

        Args:
            model_dim (int): Dimension of the model.
            aggregation_mode (str): Mode for aggregating features. One of {"attention", "gru", "none", "mean", "max"}.
            score_mode (str): Mode for calculating scores. One of {"probs", "iou", "probs_iou"}.
        """
        super().__init__()
        # check score and aggregation modes
        assert aggregation_mode in {"attention", "gru", "none", "mean", "max"}
        assert score_mode in {"probs", "iou", "probs_iou"}
        self.score_mode = score_mode
        self.aggregation_mode = aggregation_mode
        # self.temperature = nn.Parameter(torch.tensor(TEMP))

        # init aggregation class
        if aggregation_mode == "attention":
            self.learned_aggregation: Optional[nn.Module] = LearnedAggregation(
                model_dim, LearnedAggregationLayer(model_dim), 1
            )
        elif aggregation_mode == "gru":
            self.learned_aggregation = GRUFeatureExtractor(model_dim, model_dim)
        elif aggregation_mode == "none":
            self.learned_aggregation = None
        elif aggregation_mode == "mean":
            self.learned_aggregation = GlobalMeanPooling()
        else:
            self.learned_aggregation = GlobalMaxPooling()
        self.cross_attentio_with_scores = CrossAttentionWithProbs(model_dim, model_dim, model_dim)
        # init linear mapper
        self.linear = nn.Linear(model_dim, 1)
        nn.init.constant_(self.linear.weight.data, 0)  # noqa: WPS219
        nn.init.constant_(self.linear.bias.data, 0)  # noqa: WPS219

    def extract_intervals(self, src_vid: Tensor, predicted_spans_idxes: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, seq_length, model_dim = src_vid.shape
        _, num_spans, _ = predicted_spans_idxes.shape

        # Create a list to store the extracted intervals for each batch
        extracted_intervals = []
        masks = []

        for batch_idx in range(batch_size):
            batch_intervals = []
            batch_masks = []
            for span_idx in range(num_spans):
                start_idx, end_idx = predicted_spans_idxes[batch_idx, span_idx].tolist()

                # Correct intervals where start_idx equals end_idx
                if start_idx == end_idx and end_idx < seq_length:
                    end_idx += 1
                if start_idx == end_idx and end_idx == seq_length:
                    start_idx -= 1

                # Ensure indices are within valid range
                start_idx = max(0, start_idx)
                end_idx = min(seq_length, end_idx)

                # Extract the interval
                length = end_idx - start_idx
                mask = [1] * length + [0] * (seq_length - length)

                mask = torch.tensor(mask).bool().to(src_vid.device)  # type: ignore
                interval = src_vid[batch_idx, start_idx:end_idx]
                padding = torch.zeros(seq_length - length, model_dim).to(src_vid.device)
                interval = torch.concat([interval, padding])  # [seq_length, model_dim]
                batch_intervals.append(interval)
                batch_masks.append(mask)

            extracted_intervals.append(torch.stack(batch_intervals))
            masks.append(torch.stack(batch_masks))  # type: ignore
        return torch.stack(extracted_intervals), torch.stack(masks)

    def forward(
        self,
        saliency_scores: Tensor,
        video_features: Tensor,
        outputs_class: Tensor,
        quality_scores: Tensor,
        outputs_coord: Tensor,
        video_text_features: Tensor,
    ) -> Tensor:
        """
        Forward pass of the MR2HD class.

        Args:
            saliency_scores (Tensor): shape [bs, seq_length], initial saliency scores.
            video_features (Tensor): shape [bs, seq_length, model_dim], video features before cross-attention with text.
            outputs_class (Tensor): shape [bs, num_queries, 2], class logits (0 idx is positive class).
            outputs_coord (Tensor): shape [bs, num_queries, 2], spans in normalized [0, 1] format (center and width).
            quality_scores (Tensor): shape [bs, num_queries], quality scores for each query.
            video_text_features (Tensor): shape [seq_length, bs, model_dim], video features after cross-attn with text.

        Returns:
            Tensor: updated saliency scores, shape [bs, seq_length]
        """
        if self.learned_aggregation is None:
            logits = self.linear(video_text_features.transpose(0, 1))[:, :, 0]  # [batch_size, seq_length]
            return saliency_scores + logits  # [batch_size, seq_length]

        batch_size, video_length, _ = video_features.shape
        scores = torch.sigmoid(outputs_class)[..., 0]
        outputs_coord_xx = span_cxw_to_xx(outputs_coord)  # [batch_size, num_queries, 2]
        predicted_spans_idxes = torch.round(outputs_coord_xx * video_length).to(int)  # type: ignore
        # predicted_spans_idxes shape [batch_size, num_queries, 2]

        extracted_intervals, masks = self.extract_intervals(video_features, predicted_spans_idxes)
        # list_of_seqs: List[Tensor] = []
        list_of_aggregation_span_vectors = []
        for batch_idx in range(batch_size):
            batch_intervals = extracted_intervals[batch_idx]
            batch_masks = masks[batch_idx]

            # agregate features inside interval
            if self.aggregation_mode in {"attention", "max", "mean"}:
                aggregation_span_vectors = self.learned_aggregation(batch_intervals, batch_masks)[:, 0, :]
            else:  # only for test ancient evil
                span_vectors = [
                    self.learned_aggregation(interval[mask][None, :, :])[0]
                    for interval, mask in zip(batch_intervals, batch_masks)
                ]
                aggregation_span_vectors = torch.stack(span_vectors)

            list_of_aggregation_span_vectors.append(aggregation_span_vectors)
        aggregation_span_vectors = torch.stack(list_of_aggregation_span_vectors)

        video_text_features = video_text_features.transpose(0, 1)
        video_text_features_updated = self.cross_attentio_with_scores(
            aggregation_span_vectors, video_text_features, scores
        )
        logits = self.linear(video_text_features_updated)[:, :, 0]  # [batch_size, seq_length]
        return saliency_scores + logits  # [batch_size, seq_length]
