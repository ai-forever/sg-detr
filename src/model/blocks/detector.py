"""DETR Transformer class."""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as func

from src.model.blocks.decoder import TransformerDecoder
from src.model.blocks.encoder import TransformerEncoder
from src.model.blocks.feed_forward import SlimMLP
from src.model.blocks.layers import TransformerDecoderLayer, TransformerEncoderLayer
from src.model.utils.aux_anchors import prepare_anchors_codetr, prepare_anchors_dn
from src.model.utils.model_utils import gen_encoder_output_proposals, inverse_sigmoid
from src.model.utils.schemas import (
    DetectorOutput,
    DetEncoderOutput,
    QueryProposalsOutput,
)

EPS: float = 0.01
MIN_CONST: float = -1e7


class QuerySelector(nn.Module):  # noqa: WPS230
    """Class to select suqries for decoder."""

    def __init__(
        self,
        model_dim: int,
        num_queries: int,
        prior_prob: float = 0.35,
        default_widths: List[float] = [0.05, 0.2, 0.4, 0.85],
        init_spans_with_zeros: bool = True,
    ):
        """Initialize QuerySelector.

        Args:
            model_dim (int): Hidden dimension of the model.
            num_queries (int): number of queries to predict.
            prior_prob (float): prior foreground prob.
            default_widths (float): default width of encoder's anchors.
            init_spans_with_zeros (bool): whether to init last mlp layer with zeros or not
        """
        super().__init__()
        self.model_dim = model_dim
        self.num_queries = num_queries
        self.prior_prob = prior_prob
        self.default_widths = default_widths
        self.init_spans_with_zeros = init_spans_with_zeros

        self.enc_output = nn.Linear(model_dim, model_dim)
        self.enc_output_norm = nn.LayerNorm(model_dim)
        self.enc_out_span_embed = SlimMLP(input_dim=model_dim, hidden_dim=model_dim, output_dim=2, num_layers=3)
        self.enc_out_class_embed = nn.Linear(model_dim, 1)
        self.enc_out_iou_embed = nn.Linear(model_dim, 1)
        self._init_parameters(prior_prob)

    def _init_parameters(self, prior_prob: float) -> None:
        """Init parameters.

        Args:
            prior_prob (float): prior foreground prob.
        """
        # init cls layer with prior prob
        bias_value = math.log(prior_prob / (1 - prior_prob))
        torch.nn.init.normal_(self.enc_out_class_embed.weight, std=EPS)  # noqa: WPS432
        torch.nn.init.normal_(self.enc_out_iou_embed.weight, std=EPS)  # noqa: WPS432
        self.enc_out_class_embed.bias.data.fill_(bias_value)
        self.enc_out_iou_embed.bias.data.fill_(bias_value * 2)

        # init last reg layer with zeros
        for module in self.enc_out_span_embed.linear_mapper.modules():
            if isinstance(module, nn.Linear) and module.out_features == 2 and self.init_spans_with_zeros:
                nn.init.constant_(module.weight.data, 0)  # noqa: WPS219
                nn.init.constant_(module.bias.data, 0)  # noqa: WPS219

        # init mapper
        nn.init.xavier_uniform_(self.enc_output.weight.data)
        nn.init.constant_(self.enc_output.bias.data, 0)

    def get_query_proposals(
        self,
        memory_aux: Tensor,
        output_proposals: Tensor,
        mask: Tensor,
    ) -> QueryProposalsOutput:
        """Prepare query proposals.

        Args:
            memory_aux (Tensor): Query memory tensor from the encoder. Shape: [bs, seq, dim]
            output_proposals (Tensor): Tensor containing the initial proposals. Shape: [bs, seq, 2]
            mask (Tensor): mask for irrelevant embs.

        Returns:
            QueryProposalsOutput: Query proposal schema.
        """
        # map memory features
        memory_aux = self.enc_output_norm(self.enc_output(memory_aux))

        # predict objectness and iou scores
        enc_outputs_class_unselected = self.enc_out_class_embed(memory_aux)
        enc_outputs_iou_unselected = self.enc_out_iou_embed(memory_aux)

        # combine them
        enc_outputs_combo_unselected = torch.sqrt(
            enc_outputs_class_unselected.sigmoid() * enc_outputs_iou_unselected.sigmoid(),
        )
        enc_outputs_combo_unselected = enc_outputs_combo_unselected.masked_fill(~mask, float("-inf"))

        # predict reference points
        enc_outputs_offsets_unselected = self.enc_out_span_embed(memory_aux)
        enc_outputs_coord_unselected = output_proposals + enc_outputs_offsets_unselected

        # Find the most relevant indices
        topk_proposals = torch.topk(enc_outputs_combo_unselected[..., 0], self.num_queries, dim=1)[1]  # noqa: WPS221
        topk_proposals = topk_proposals.unsqueeze(-1)

        # gather features
        query_topk = topk_proposals.repeat(1, 1, self.model_dim)
        query_embs = torch.gather(memory_aux, 1, query_topk).detach()

        # gather spans
        spans_topk = topk_proposals.repeat(1, 1, 2)
        refpoint_embed_undetach = torch.gather(enc_outputs_coord_unselected, 1, spans_topk)  # unsigmoid
        refpoint_embed_detach = refpoint_embed_undetach.detach()

        # gather logits
        class_logit_enc = torch.gather(enc_outputs_class_unselected, 1, topk_proposals[..., [0]])
        iou_logit_enc = torch.gather(enc_outputs_iou_unselected, 1, topk_proposals[..., [0]])

        return QueryProposalsOutput(
            query_embs=query_embs,
            refpoint_embed_detach=refpoint_embed_detach,
            refpoint_embed_enc=refpoint_embed_undetach.sigmoid(),
            class_logit_enc=class_logit_enc,
            iou_logit_enc=iou_logit_enc,
        )

    @staticmethod
    def _prepare_mask(fpn_features: List[Tensor], mask: Tensor) -> List[Tensor]:  # noqa: WPS602
        spatial_shapes = [seq.size(1) for seq in fpn_features]
        original_seq_length = mask.shape[1]

        updated_masks = []
        for seq_length in spatial_shapes:
            if seq_length > original_seq_length:
                scale = seq_length // original_seq_length
                updated_mask = mask.repeat_interleave(scale, dim=1)
                updated_masks.append(updated_mask)
            elif seq_length < original_seq_length:
                scale = original_seq_length // seq_length
                updated_mask = func.max_pool1d(mask.float(), kernel_size=scale, stride=scale)  # type: ignore
                updated_masks.append(updated_mask.bool())
            else:
                updated_masks.append(mask)
        return updated_masks

    def forward(self, multiscale: List[Tensor], vid_mask: Tensor) -> QueryProposalsOutput:
        """Forward pass of the QuerySelector.

        Args:
            multiscale (List[Tensor]): multiscale features.
            vid_mask (Tensor): mask for the source sequence. Shape: [batch_size, Lv]

        Returns:
            QueryProposalsOutput: query proposals.
        """
        masks = self._prepare_mask(multiscale, vid_mask)

        # compute reference points
        memory_aux, output_proposals, mask = gen_encoder_output_proposals(
            fpn_features=multiscale,
            memory_padding_masks=masks,
            default_widths=self.default_widths,
        )

        # get proposal outputs
        return self.get_query_proposals(memory_aux, output_proposals, mask)


class DetectorEncoder(nn.Module):
    """Stack of the encoder layers."""

    def __init__(
        self,
        model_dim: int,
        num_encoder_layers: int = 3,
        dropout: float = 0.1,
        droppath: float = 0.1,
    ) -> None:
        """Initialize Detector Encoder.

        Args:
            model_dim (int): Hidden dimension of the model
            num_encoder_layers (int): Number of encoder layers.
            dropout (float): Dropout rate
            droppath (float): Droppath rate
        """
        super().__init__()
        self.model_dim = model_dim
        self.enc_layers = num_encoder_layers

        # Init Encoder
        general_encoder_layer = TransformerEncoderLayer(model_dim, dropout=dropout, droppath=droppath)
        self.encoder = TransformerEncoder(general_encoder_layer, num_encoder_layers)

    def forward(
        self,
        src: Tensor,
        mask: Tensor,
        pos: Tensor,
        video_length: Tensor,
    ) -> DetEncoderOutput:
        """Forward pass of the Transformer.

        Args:
            src (Tensor): source sequence. Shape: [Lv, batch_size, dim]
            mask (Tensor): mask for the source sequence. Shape: [batch_size, Lv]
            pos (Tensor): positional embedding. Shape: [Lv, batch_size, dim]
            video_length (Tensor): length of the video. Shape: [batch_size]

        Returns:
            DetEncoderOutput: output of the encoder
        """
        vid_src = src[:video_length]  # (L_video, batch_size, dim)
        vid_mask = mask[:, :video_length]  # (batch_size, L_video)
        vid_pos = pos[:video_length]  # (L_video, batch_size, dim)

        # encoder forward pass
        memory = self.encoder(vid_src, vid_pos, src_key_padding_mask=~vid_mask)

        return DetEncoderOutput(memory=memory, vid_pos=vid_pos, vid_mask=vid_mask)


class MomentDetector(nn.Module):  # noqa: WPS230
    """Transformer module from DETR."""

    def __init__(  # noqa: WPS211
        self,
        reference: Optional[nn.Module],
        model_dim: int = 512,
        cont_pos_tradeoff: int = 0,
        num_queries: int = 25,
        use_rpn: bool = True,
        use_encoder_features: bool = True,
        num_decoder_layers: int = 3,
        dropout: float = 0.1,
        droppath: float = 0.1,
        temperature: int = 10000,
        prior_prob: float = 0.35,
        unique_content_queries: bool = True,
        init_spans_with_zeros: bool = True,
        return_intermediate_dec: bool = True,
        predict_quality_score: bool = True,
        num_groups: int = 5,
        span_noise_scale: float = 0.4,
        negative_offset: float = 1.0,
        look_at_target: bool = False,
        aux_anchors_type: Tuple[str, ...] = (),
    ):
        """
        Initialize a Transformer module.

        Args:
            reference (nn.Module): anchors generator.
            model_dim (int): hidden dimension of the model
            cont_pos_tradeoff (int): Offset for the dim content/position dims.
            num_queries (int): number of queries
            use_rpn (bool): whether to use RPN as anchor generator or reference.
            use_encoder_features (bool): whether to use encoder features as content queries or not.
            num_decoder_layers (int): number of decoder layers
            dropout (float): dropout rate
            droppath (float): droppath rate
            temperature (int): temperature of the pos emb.
            prior_prob (float): prior foreground prob.
            unique_content_queries (bool): unique content embeddings.
            init_spans_with_zeros (bool): whether to init last mlp layer with zeros or not
            return_intermediate_dec (bool): whether to return intermediate results of the decoder
            predict_quality_score (bool): predict iou of the predicted interval
            num_groups (int): number of noised gt groups.
            span_noise_scale (float): noise scale for the bbox
            negative_offset (float): offset for negative samples
            look_at_target (bool): if true aux anchors can see main queries
            aux_anchors_type (Tuple[str]): aux anchors preparation method. Could be of {"collab", "denoise"}
        """
        super().__init__()
        self.model_dim = model_dim
        self.cont_pos_tradeoff = cont_pos_tradeoff
        self.dec_layers = num_decoder_layers
        self.num_groups = num_groups
        self.span_noise_scale = span_noise_scale
        self.use_rpn = use_rpn
        self.use_encoder_features = use_encoder_features
        self.look_at_target = look_at_target
        self.aux_anchors_type = aux_anchors_type
        self.unique_content_queries = unique_content_queries
        self.init_spans_with_zeros = init_spans_with_zeros
        self.predict_quality_score = predict_quality_score
        self.negative_offset = negative_offset

        # defince query tokens for decoder layers
        self.num_queries: int = num_queries
        self.refpoint_embed = None if use_rpn else reference

        # content query encoder
        self.init_content_queries(num_queries, model_dim, aux_anchors_type)

        decoder_layer = TransformerDecoderLayer(
            d_model=model_dim,
            cont_pos_tradeoff=cont_pos_tradeoff,
            dropout=dropout,
            droppath=droppath,
        )

        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            return_intermediate=return_intermediate_dec,
            d_model=model_dim,
            temperature=temperature,
            predict_quality_score=predict_quality_score,
            init_spans_with_zeros=init_spans_with_zeros,
        )
        # define heads for classification and box regression
        self.span_embed = SlimMLP(input_dim=model_dim, hidden_dim=model_dim, output_dim=2, num_layers=3)
        self.class_embed = nn.Linear(model_dim, 1)
        self._init_parameters(prior_prob)

    def init_content_queries(self, num_queries: int, model_dim: int, aux_anchors_type: Tuple[str, ...]) -> None:
        """Init contnent queries.

        Args:
            num_queries (int): number of queries to use.
            model_dim (int): model dim.
            aux_anchors_type (Tuple[str, ...]): list of auxiliary anchors.
        """
        if self.use_rpn and self.use_encoder_features:
            self.label_enc = None
        else:
            if self.unique_content_queries:
                self.label_enc = nn.Embedding(num_queries, model_dim)
            else:
                self.label_enc = nn.Embedding(1, model_dim)
            nn.init.normal_(self.label_enc.weight.data, std=EPS)
        if "collab" in aux_anchors_type:
            self.co_mapper = nn.Sequential(nn.Linear(model_dim, model_dim), nn.LayerNorm(model_dim))
            nn.init.normal_(self.co_mapper[0].weight.data, std=EPS)
        if "denoise" in aux_anchors_type:
            self.dn_label_enc = nn.Embedding(1, model_dim)
            nn.init.normal_(self.dn_label_enc.weight.data, std=EPS)

    def _init_parameters(self, prior_prob: float) -> None:
        """Init parameters.

        Args:
            prior_prob (float): prior foreground prob.
        """
        # init cls layer with prior prob
        bias_value = math.log(prior_prob / (1 - prior_prob))
        nn.init.normal_(self.class_embed.weight, std=EPS)  # noqa: WPS432
        self.class_embed.bias.data.fill_(bias_value)

        # init last reg layer with zeros
        for module in self.span_embed.linear_mapper.modules():
            if isinstance(module, nn.Linear) and module.out_features == 2 and self.init_spans_with_zeros:
                nn.init.constant_(module.weight.data, 0)  # noqa: WPS219
                nn.init.constant_(module.bias.data, 0)  # noqa: WPS219

    def prepare_regular_detr(
        self,
        proposals: Optional[QueryProposalsOutput],
        refpoint_emb: Tensor,
        batch_size: int = 512,
    ) -> Tuple[Tensor, Tensor]:
        """
        Prepare reference points for detr decoder.

        Args:
            proposals (QueryProposalsOutput): query selector output.
            refpoint_emb (Tensor): positional queries as anchor points
            batch_size (int): batch size

        Returns:
            Tuple[Tensor, Tensor]:
                - input_query_label: label query embedding for detr decoder
                - input_query_spans: reference points for detr decoder
        """
        # prepare content queries
        if self.use_rpn and self.use_encoder_features:
            assert proposals is not None
            input_query_label = proposals.query_embs.transpose(0, 1)
        else:
            if self.label_enc is None:
                raise ValueError("label_enc cannot be None when using anchors")
            input_query_label = self.label_enc.weight[:, None, :]
            if self.unique_content_queries:
                input_query_label = input_query_label.repeat(1, batch_size, 1)
            else:
                input_query_label = input_query_label.repeat(self.num_queries, batch_size, 1)

        # prepare pos queries
        input_query_span = refpoint_emb if self.use_rpn else refpoint_emb.repeat(batch_size, 1, 1)
        input_query_span = input_query_span.transpose(0, 1)
        return input_query_label, input_query_span

    def get_collab_queries(  # noqa: WPS234
        self,
        matched_gts: Optional[Tensor],
        anchors_spans: Optional[Tensor],
        encoder_features: Optional[Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Dict[str, Any]]]:  # noqa: WPS221
        """Get coolab queries.

        Args:
            memory_local (Tensor): Output from the encoder. Shape: [Lv, batch_size, dim]
            matched_gts (Optional[Tensor]): gt spans mathced to selected anchors.
            anchors_spans (Optional[Tensor]): selected anchors.
            encoder_features (Optional[Tensor]): selected encoder features.

        Returns:
            Tuple[Optional[Tensor], Optional[Tensor], Optional[Dict[str, Any]]]: prepared collab info
        """
        if (
            "collab" in self.aux_anchors_type
            and self.training
            and matched_gts is not None
            and anchors_spans is not None
        ):
            co_query_label, co_query_span, co_info = prepare_anchors_codetr(
                linear_mapper=self.co_mapper,  # type: ignore
                matched_gts=matched_gts,  # type: ignore
                anchors_per_seq=anchors_spans,  # type: ignore
                encoder_features_per_seq=encoder_features,  # type: ignore
            )
        else:
            co_query_label = torch.empty((0, batch_size, self.model_dim), device=device)
            co_query_span = torch.empty((0, batch_size, 2), device=device)
            co_info = None
        return co_query_label, co_query_span, co_info

    def get_denoise_queries(  # noqa: WPS234
        self,
        targets: Optional[Dict[str, Any]],
        batch_size: int,
        device: torch.device,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Dict[str, Any]]]:  # noqa: WPS221
        """Get denoise queries.

        Args:
            targets (Optional[Dict[str, Any]]): gt spans mathced to selected anchors.
            batch_size (int): batch size
            device (torch.device): device

        Returns:
            Tuple[Optional[Tensor], Optional[Tensor], Optional[Dict[str, Any]]]: prepared denoise info
        """
        if "denoise" in self.aux_anchors_type and self.training and self.num_groups > 0 and targets is not None:
            dn_query_label, dn_query_span, dn_info = prepare_anchors_dn(
                label_enc=self.dn_label_enc,
                targets=targets,
                num_groups=self.num_groups,
                span_noise_scale=self.span_noise_scale,
                negative_offset=self.negative_offset,
                batch_size=batch_size,
            )
        else:
            dn_query_label = torch.empty((0, batch_size, self.model_dim), device=device)
            dn_query_span = torch.empty((0, batch_size, 2), device=device)
            dn_info = None
        return dn_query_label, dn_query_span, dn_info

    def _get_attention_mask(
        self,
        co_info: Optional[Dict[str, Any]],
        dn_info: Optional[Dict[str, Any]],
    ) -> Optional[Tensor]:
        """Compute attention mask.

        Args:
            co_info (Optional[Dict[str, Any]]): collaborative anchors info
            dn_info (Optional[Dict[str, Any]]): denoise anchors info

        Returns:
            Optional[Tensor]: computed attention mask
        """
        if co_info is None and dn_info is None:
            return None

        tgt_size = self.num_queries
        colab_size = co_info["pad_size"] if co_info is not None else 0
        denoise_size = dn_info["pad_size"] if dn_info is not None else 0
        attn_mask = torch.zeros(
            tgt_size + colab_size + denoise_size,
            tgt_size + colab_size + denoise_size,
            device=self.class_embed.weight.device,
        ).bool()

        total_mask = colab_size + denoise_size
        total_mask = total_mask if self.look_at_target else total_mask + tgt_size

        if dn_info is not None:
            num_groups = dn_info["num_groups"]
            double_pad = denoise_size / num_groups
            # match query cannot see the reconstruct
            attn_mask[denoise_size:, :denoise_size] = True
            # reconstruct cannot see each other
            for idx in range(num_groups):
                double_idx = int(double_pad * idx)
                double_idx_p = int(double_pad * (idx + 1))
                attn_mask[double_idx:double_idx_p, double_idx_p:total_mask] = True
                attn_mask[double_idx:double_idx_p, :double_idx] = True

        if co_info is not None:
            attn_mask[denoise_size + colab_size :, : denoise_size + colab_size] = True
            attn_mask[denoise_size : denoise_size + colab_size, denoise_size + colab_size : total_mask] = True

        return attn_mask

    def _predict_spans(self, output: Tensor, reference: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Predict spans.

        Args:
            output (Tensor): content vector from CA block
            reference (Tensor): anchor points

        Returns:
            Tuple[Tensor, Tensor]: predicter spans, shape [batch_size, quary_num, 2] and offsets
        """
        offset = self.span_embed(output)
        reference_before_sigmoid = inverse_sigmoid(reference)
        outputs_coord = offset + reference_before_sigmoid
        outputs_coord = outputs_coord.sigmoid()
        # due to the fact that the usual offset that is calculated above adds up before sigmoids
        final_offset = outputs_coord - reference
        return outputs_coord, final_offset

    # pylint: disable=too-many-locals
    def forward(  # noqa: R0913 C901
        self,
        memory_local: Tensor,
        vid_mask: Tensor,
        vid_pos: Tensor,
        matched_gts: Optional[List[Tensor]],
        anchors_spans: Optional[List[Tensor]],
        encoder_features: Optional[List[Tensor]],
        proposals: Optional[QueryProposalsOutput],
        targets: Optional[Dict[str, Any]] = None,
    ) -> DetectorOutput:
        """Forward pass of the Transformer.

        Args:
            memory_local (Tensor): Output from the encoder. Shape: [Lv, batch_size, dim]
            vid_mask (Tensor): mask for the source sequence. Shape: [batch_size, Lv]
            vid_pos (Tensor): positional embedding. Shape: [Lv, batch_size, dim]
            matched_gts (Optional[List[Tensor]]): gt spans mathced to selected anchors.
            anchors_spans (Optional[List[Tensor]]): selected anchors.
            encoder_features (Optional[List[Tensor]]): selected encoder features.
            proposals (Optional[QueryProposalsOutput]): proposals info.
            targets (Optional[Dict[str, Any]]): target meta information. Defaults to None.

        Returns:
            DetectorOutput: detector output schema.
        """
        batch_size = memory_local.size(1)
        device = memory_local.device
        if self.use_rpn:
            assert proposals is not None
            ref_points = proposals.refpoint_embed_detach
        else:
            ref_points = self.refpoint_embed.get_reference_points()  # type: ignore

        input_query_label, input_query_span = self.prepare_regular_detr(
            proposals=proposals,
            refpoint_emb=ref_points,
            batch_size=batch_size,
        )
        co_query_label, co_query_span, co_info = self.get_collab_queries(
            matched_gts,  # type: ignore
            anchors_spans,  # type: ignore
            encoder_features,  # type: ignore
            batch_size,
            device,
        )
        dn_query_label, dn_query_span, dn_info = self.get_denoise_queries(targets, batch_size, device)
        attn_mask = self._get_attention_mask(co_info, dn_info)
        input_query_label = torch.cat([dn_query_label, co_query_label, input_query_label], dim=0)  # type: ignore
        input_query_span = torch.cat([dn_query_span, co_query_span, input_query_span], dim=0)  # type: ignore

        hs, reference_points, quality_score = self.decoder(  # noqa: WPS111
            src=memory_local,
            src_key_padding_mask=~vid_mask,
            src_pos=vid_pos,
            content=input_query_label,
            content_mask=attn_mask,
            refpoints_unsigmoid=input_query_span,
        )  # (#layers, #queries, batch_size, dim)

        # get positive class and coords
        outputs_class: Tensor = self.class_embed(hs)  # (#layers, batch_size, #queries, 1)
        outputs_coord, offset = self._predict_spans(hs, reference_points)

        return DetectorOutput(
            outputs_class=outputs_class,
            outputs_coord=outputs_coord,
            offsets=offset,
            quality_scores=quality_score,
            co_info=co_info,
            dn_info=dn_info,
        )
