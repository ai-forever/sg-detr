"""MRDETR model."""

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from src.model.blocks.atss import ATSSHead
from src.model.blocks.audio import AudioMerger
from src.model.blocks.detector import DetectorEncoder, MomentDetector, QuerySelector
from src.model.blocks.encoder import (
    DummyEncoder,
    LocalSaliencyHead,
    MomentEncoder,
    SaliencyAmplifier,
    SentenceEncoder,
    Text2VisionEncoder,
)
from src.model.blocks.feed_forward import MLP
from src.model.blocks.multiscale import FPNSequence
from src.model.blocks.position_encoding import (
    PositionEmbeddingSine,
    TrainablePositionalEncoding,
)
from src.model.misc import (  # moment_txt_similarity,
    prepare_negative_tensors,
    prepare_real_neg_mask,
)
from src.model.utils.aux_anchors import aux_post_process
from src.model.utils.model_utils import init_weights
from src.model.utils.schemas import (
    AuxDetectorOutput,
    MomentEncoderOutput,
    SentenceEncoderOutput,
)

EPS: float = 0.01


# pylint: disable=too-many-instance-attributes,too-many-branches
class MRDETR(nn.Module):
    """Saliency Guided Hybrid DETR."""

    # pylint: disable=too-many-arguments,too-many-locals
    def __init__(
        self,
        audio_merger: AudioMerger,
        txt2vis_encoder: Text2VisionEncoder,
        query_selector: Optional[QuerySelector],
        det_encoder: DetectorEncoder,
        detr_detector: MomentDetector,
        aux_det_head: ATSSHead,
        saliency_amplifier: SaliencyAmplifier,
        local_saliency_head: LocalSaliencyHead,
        use_global_saliency_head: bool,
        batch_size: int,
        model_dim: int,
        aud_dim: int,
        vid_dim: int,
        txt_dim: int,
        max_video_length: int,
        num_input_proj_layers: int,
        num_dummy_layers: int,
        num_sentence_layers: int,
        num_moment_layers: int,
        dropout: float,
        proj_dropout: float,
        droppath: float,
        pos_temp: int,
        aux_loss: bool = True,
        use_txt_pos: bool = False,
        saliency_video_features_mode: str = "src",
        saliency_video_text_features_mode: str = "memory",
    ) -> None:
        """Initialize MRDETR.

        Args:
            audio_merger (AudioMerger): module to merge audio and video features
            txt2vis_encoder (Text2VisionEncoder): text to vision encoder
            query_selector (Optional[QuerySelector]): RPN
            det_encoder (DetectorEncoder): detection encoder
            detr_detector (MomentDetector): detr head
            aux_det_head (ATSSHead): auxiliary detection head
            saliency_amplifier (SaliencyAmplifier): saliency amplifier module
            local_saliency_head (LocalSaliencyHead): local saliency head module
            use_global_saliency_head (bool): whether to use global saliency head or not
            batch_size (int): batch size
            aud_dim (int): audio feature input dimension
            vid_dim (int): video feature input dimension
            txt_dim (int): text query input dimension
            model_dim (int): hidden model dimension
            max_video_length (int): maximum #clips in videos
            num_input_proj_layers (int): number of input projection layers
            num_dummy_layers (int): number of dummy layers
            num_sentence_layers (int): number of sentence layers
            num_moment_layers (int): number of moment layers
            dropout (float): dropout probability
            proj_dropout (float): projection dropout probability
            droppath (float): drop path probability
            pos_temp (int): temperature of the pos emb
            aux_loss (bool): If true, auxiliary decoding losses (loss at each decoder layer) are to be used.
            use_txt_pos (bool): If true, use trainable positional embeddings for text
            saliency_video_features_mode (str): mode of features for saliency generator
            saliency_video_text_features_mode (str): mode of video-text features for saliency generator
        """
        super().__init__()
        self.batch_size = batch_size  # used by pl.Trainer
        self.num_sentence_layers = num_sentence_layers
        self.num_moment_layers = num_moment_layers
        self.max_video_length = max_video_length
        self.model_dim = model_dim
        self.num_input_proj_layers = num_input_proj_layers
        self.saliency_video_features_mode = saliency_video_features_mode
        self.saliency_video_text_features_mode = saliency_video_text_features_mode

        # define input projectors
        self.input_txt_proj = MLP(
            input_dim=txt_dim,
            hidden_dim=model_dim,
            output_dim=model_dim,
            num_layers=num_input_proj_layers,
            dropout=proj_dropout,
        )

        self.input_aud_proj = (
            MLP(
                input_dim=aud_dim,
                hidden_dim=model_dim,
                output_dim=model_dim,
                num_layers=num_input_proj_layers,
                dropout=proj_dropout,
            )
            if audio_merger.merge_type is not None
            else None
        )

        self.input_vid_proj = MLP(
            input_dim=vid_dim,
            hidden_dim=model_dim,
            output_dim=model_dim,
            num_layers=num_input_proj_layers,
            dropout=proj_dropout,
        )

        # define modality embeddings
        self.modality_embeddings = nn.Embedding(2, model_dim)
        self.modality_embeddings.apply(init_weights)

        # define position embeddings builder
        self.vid_position_embed = PositionEmbeddingSine(model_dim, temperature=pos_temp)
        self.txt_position_embed = TrainablePositionalEncoding(model_dim, max_len=100)
        self.use_txt_pos = use_txt_pos

        # define audio merger
        self.audio_merger = audio_merger

        # define text to vision encoder
        self.txt2vis_sal_encoder = txt2vis_encoder
        self.use_cross_attn_wo_dummy = self.txt2vis_sal_encoder.use_cross_attn_wo_dummy
        self.num_dummies = self.txt2vis_sal_encoder.num_dummies

        # define dummy encoder
        if self.num_dummies == 0:
            self.dummy_encoder = None
            self.num_dummy_layers = None
        else:
            self.num_dummy_layers = num_dummy_layers
            self.dummy_encoder = DummyEncoder(model_dim, self.num_dummies, num_dummy_layers, dropout, droppath)

        # define sentence encoder
        self.sent_encoder = SentenceEncoder(model_dim, num_sentence_layers, dropout, droppath)

        # define moment encoder
        self.moment_encoder = MomentEncoder(model_dim, num_moment_layers, dropout, droppath)

        self.fpn = FPNSequence(model_dim)

        # define query selector
        self.query_selector = query_selector

        # define detector encoder
        self.det_encoder = det_encoder

        # define moment detector
        self.main_det_head = detr_detector
        if self.main_det_head.use_rpn and query_selector is None:
            raise RuntimeError("If use_rpn is true query_selector should be defined.")
        if not self.main_det_head.use_rpn and query_selector is not None:
            raise RuntimeError("If use_rpn is false query_selector should not be defined.")

        # define auxiliary head
        self.aux_det_head = aux_det_head

        # define saliency amplifier
        self.saliency_amplifier = saliency_amplifier

        self.local_saliency_head = local_saliency_head
        self.use_global_saliency_head = use_global_saliency_head
        if use_global_saliency_head:
            self.global_saliency_head = nn.Linear(model_dim, 1)
            nn.init.constant_(self.global_saliency_head.weight.data, 0)  # noqa: WPS219
            nn.init.constant_(self.global_saliency_head.bias.data, 0)  # noqa: WPS219

        # Compute loss per each decoder layer
        self.aux_loss = aux_loss

    def get_mr2hd_features(
        self,
        src_vid: Tensor,
        memory_origins: Tensor,
        memory: Tensor,
        pos_emb: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Prepare input features for mr2hd module.

        Args:
            src_vid (Tensor): intitial video features
            memory_origins (Tensor): encoder features
            memory (Tensor): encoder features after amplified module
            pos_emb (Tensor): positional embs.

        Raises:
            ValueError: Unsupported saliency_video_features_mode or saliency_video_text_features_mode

        Returns:
            Tuple[Tensor, Tensor]: local and enhanced features
        """
        if self.saliency_video_features_mode == "src":
            video_features = src_vid + pos_emb
        elif self.saliency_video_features_mode == "memory":
            video_features = memory_origins.transpose(0, 1) + pos_emb
        elif self.saliency_video_features_mode == "memory_amplified":
            video_features = memory.transpose(0, 1) + pos_emb
        else:
            raise ValueError(f"unsupported saliency_video_features_mode: {self.saliency_video_features_mode}")

        if self.saliency_video_text_features_mode == "memory":
            video_text_features = memory_origins + pos_emb.transpose(0, 1)
        elif self.saliency_video_text_features_mode == "memory_amplified":
            video_text_features = memory + pos_emb.transpose(0, 1)
        else:
            raise ValueError(f"unsupported saliency_video_text_features_mode: {self.saliency_video_text_features_mode}")
        return video_features, video_text_features

    # pylint: disable=R0913,R0914,R0915,E1130
    def forward(  # noqa: WPS234,WPS231
        self,
        src_txt: Tensor,
        src_txt_mask: Tensor,
        src_vid: Tensor,
        src_vid_mask: Tensor,
        vid: Optional[List[str]],
        src_aud: Optional[Tensor] = None,
        targets: Optional[Dict[str, Any]] = None,
        meta: Optional[List[Dict[str, Any]]] = None,
        **_,
    ) -> Dict[str, Any]:
        """Forward pass of the MRDETR.

        Args:
            src_txt (Tensor): Query embedding, shape: [batch_size, L_txt, D_txt]
            src_txt_mask (Tensor): Query embedding mask, containing 0 on padded pixels. Shape: [batch_size, L_vid]
            src_vid (Tensor): Video embedding, shape: [batch_size, L_vid, D_vid]
            src_vid_mask (Tensor): Video embedding mask, containing 0 on padded pixels. Shape: [batch_size, L_vid]
            vid (Optional[List[str]]): Names of the videos in the batch
            src_aud (Optional[Tensor]): Audio embedding, shape: [batch_size, L_vid, D_vid]
            targets (Optional[Dict[str, Any]]): target information. Defaults to None.
            meta (Optional[List[Dict[str, Any]]]): meta information. Defaults to None.
            _ (Dict[str, Any]): additional useless arguments

        Returns:
            Dict[str, Any]: output dict with the following entries:
                - "pred_logits": the classification logits (including no-object) for all queries.
                - "pred_boxes": The normalized boxes coordinates for all queries, represented as (center_x, w) in [0, 1]
                - "aux_outputs": Only returned when auxilary losses are activated. Logits and boxes for each dec. layer
        """
        # get video length
        video_length = src_vid.shape[1]
        real_video_len = src_vid_mask.sum(1).long()

        # project input features to same dimension
        src_aud = self.input_aud_proj(src_aud) if self.input_aud_proj is not None else src_aud
        src_vid = self.input_vid_proj(src_vid)
        src_txt = self.input_txt_proj(src_txt)

        # get saliency tokens
        saliency_scores, src_sent = self.local_saliency_head(src_vid, src_txt, src_txt_mask)

        # generate positional embeddings for text and video
        pos_vid = self.vid_position_embed(src_vid_mask)  # (bsz, L_vid, d)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)  # (bsz, L_txt, d)

        # merge video and audio
        src_vid = self.audio_merger(
            audio=src_aud,
            video=src_vid,
            pos_emb=pos_vid,
            mask=src_vid_mask,
        )

        # add modality type embeddings
        src_txt = src_txt + self.modality_embeddings(torch.zeros_like(src_txt_mask.long()))
        src_vid = src_vid + self.modality_embeddings(torch.ones_like(src_vid_mask.long()))

        if self.dummy_encoder is None:
            # concat: video, txt
            src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)
            mask = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()  # (bsz, L_vid+L_txt)
            pos = torch.cat([pos_vid, pos_txt], dim=1)
            dummy_token, dummy_mask, dummy_pos = None, None, None
        else:
            # insert query excluded dummy token in front of txt
            dummy_src_txt, dummy_src_txt_mask, dummy_src_txt_pos = self.dummy_encoder(src_txt, src_txt_mask, pos_txt)

            # slice dummy entities
            dummy_token = dummy_src_txt[:, : self.dummy_encoder.num_dummies]
            dummy_mask = dummy_src_txt_mask[:, : self.dummy_encoder.num_dummies]
            dummy_pos = dummy_src_txt_pos[:, : self.dummy_encoder.num_dummies]

            # concat: video, dummy, txt
            src = torch.cat([src_vid, dummy_src_txt], dim=1)  # (bsz, L_vid+L_txt, d)
            mask = torch.cat([src_vid_mask, dummy_src_txt_mask], dim=1).bool()  # (bsz, L_vid+L_txt)
            pos = torch.cat([pos_vid, dummy_src_txt_pos], dim=1)

        if targets is not None:
            sents_schema = self.sent_encoder(src_txt, src_txt_mask, pos_txt, dummy_token, dummy_mask, dummy_pos)
            moments_schema = self.moment_encoder(src_vid, src_vid_mask, pos_vid, targets)
        else:
            sents_schema = SentenceEncoderOutput()  # type: ignore
            moments_schema = MomentEncoderOutput()  # type: ignore
        src_updated, mask_updated, pos_updated, attn_weights = self.txt2vis_sal_encoder(
            src=src,
            mask=mask,
            pos=pos,
            batch_video_len=video_length,
            saliency_scores=torch.sigmoid(saliency_scores),
        )

        encoder_output = self.det_encoder(
            src=src_updated,
            mask=mask_updated,
            pos=pos_updated,
            video_length=video_length,
        )

        memory = (
            self.saliency_amplifier(
                encoder_output.memory,
                saliency_scores,
                encoder_output.vid_pos,
                encoder_output.vid_mask,
            )
            if self.saliency_amplifier is not None
            else encoder_output.memory
        )

        if self.use_global_saliency_head:
            saliency_scores_offset = self.global_saliency_head(memory)[:, :, 0]
            saliency_scores_refined = saliency_scores + saliency_scores_offset.transpose(0, 1)

            memory = (
                self.saliency_amplifier(
                    encoder_output.memory,
                    saliency_scores_refined,
                    encoder_output.vid_pos,
                    encoder_output.vid_mask,
                )
                if self.saliency_amplifier is not None
                else encoder_output.memory
            )
        else:
            saliency_scores_refined = None

        multiscale = self.fpn(memory.transpose(0, 1))

        proposals_output = (
            self.query_selector(multiscale, encoder_output.vid_mask) if self.query_selector is not None else None
        )

        # aux head reg
        if meta is not None:
            aux_head_schema = self.aux_det_head(
                fpn_features=multiscale,
                real_video_len=real_video_len,
                targets=targets,
                meta=meta,
            )
        else:
            aux_head_schema = AuxDetectorOutput()  # type: ignore

        # main head reg
        det_output = self.main_det_head(  # noqa: WPS236
            memory_local=memory,
            vid_mask=encoder_output.vid_mask,
            vid_pos=encoder_output.vid_pos,
            matched_gts=aux_head_schema.matched_gts,
            anchors_spans=aux_head_schema.anchors_spans,
            encoder_features=aux_head_schema.selected_features,
            proposals=proposals_output,
            targets=targets,
        )

        # aux anchors postprocessing
        outputs_class, outputs_coord, quality_scores, offsets = aux_post_process(
            det_output.outputs_class,
            det_output.outputs_coord,
            det_output.quality_scores,
            det_output.offsets,
            det_output.co_info,
            det_output.dn_info,
        )

        out = {
            "local_saliency_scores": saliency_scores,
            "saliency_scores": saliency_scores_refined,
            "pred_logits_aux": aux_head_schema.cls_logits,
            "pred_spans_aux": aux_head_schema.bbox_regression,
            "pred_cntrness_aux": aux_head_schema.bbox_ctrness,
            "locations_aux": aux_head_schema.anchors,
            "pred_logits": outputs_class[-1],
            "pred_spans": outputs_coord[-1],
            "offset": offsets[-1],
            "denoise_ref_dict": det_output.dn_info,
            "collab_ref_dict": det_output.co_info,
            "pred_quality_scores": (
                quality_scores[-1] if self.main_det_head.predict_quality_score else outputs_class[-1]
            ),
        }

        # interim encoder outputs
        if proposals_output is not None:
            encoder_outputs = {
                "pred_logits": proposals_output.class_logit_enc,
                "pred_spans": proposals_output.refpoint_embed_enc,
                "pred_quality_scores": proposals_output.iou_logit_enc,
                "ref_points": proposals_output.refpoint_embed_detach,
            }
            out["encoder_outputs"] = encoder_outputs

        # moments-sentence alignment
        out["moment_token"] = moments_schema.moment_token
        out["non_moment_token"] = moments_schema.non_moment_token
        out["sent_txt_token"] = sents_schema.sent_txt_token
        out["sent_dummy_token"] = sents_schema.sent_dummy_token
        out["moment_mask"] = moments_schema.relevant_clips_mask

        # Prepare attention values (batch_size, L_vid, L_txt) / (batch_size, L_txt)
        out["dummy_tokens"] = dummy_token
        if self.num_dummies == 0:
            out["t2vattnvalues"] = None
        else:
            # Prepare attention values (batch_size, L_vid, L_txt) / (batch_size, L_txt)
            unsq_mask = src_txt_mask.unsqueeze(1).repeat(1, video_length, 1)
            out["t2vattnvalues"] = (attn_weights[:, :, self.num_dummies :] * unsq_mask).sum(2)
            out["t2vattnvalues"] = torch.clamp(out["t2vattnvalues"], 0, 1)

        # moment classification
        if targets is not None:
            out["src_vid"] = (
                moments_schema.moment_memory.permute(1, 0, 2) * moments_schema.relevant_clips_mask.unsqueeze(2)
            ) + (moments_schema.non_moment_memory.permute(1, 0, 2) * moments_schema.irrelevant_clips_mask.unsqueeze(2))
        else:
            out["src_vid"] = None

        if self.aux_loss:
            # assert proj_queries and proj_txt_mem
            out["aux_outputs"] = [
                {"pred_logits": logits, "pred_spans": spans, "offset": offset}
                for logits, spans, offset in zip(outputs_class[:-1], outputs_coord[:-1], offsets[:-1])
            ]
            if self.main_det_head.predict_quality_score:
                for idx, scores in enumerate(quality_scores[:-1]):
                    out["aux_outputs"][idx]["pred_quality_scores"] = scores
            else:
                for idx, scores in enumerate(outputs_class[:-1]):
                    out["aux_outputs"][idx]["pred_quality_scores"] = scores

        input_src_txt = src_txt if self.num_dummies == 0 else dummy_src_txt
        input_src_txt_mask = src_txt_mask if self.num_dummies == 0 else dummy_src_txt_mask
        out["txt_mask"] = input_src_txt_mask
        out["video_mask"] = src_vid_mask

        ################################################################################################
        ################################################################
        ################################
        # artificial negative inference
        if vid is not None:
            real_neg_mask = prepare_real_neg_mask(vid, input_src_txt.device)
            out["real_neg_mask"] = real_neg_mask
            if real_neg_mask.sum() == 0:
                out["saliency_scores_neg"] = None
                out["t2vattnvalues_neg"] = None
                out["local_saliency_scores_neg"] = None
            else:
                src_dummy_neg, mask_dummy_neg, pos_neg, input_src_txt_mask_neg = prepare_negative_tensors(
                    src_vid,
                    src_vid_mask,
                    input_src_txt,
                    input_src_txt_mask,
                    pos,
                    real_neg_mask,
                )
                src_sent = torch.cat([src_sent[1:], src_sent[:1]])
                saliency_scores_neg = self.local_saliency_head.saliency_scores(
                    src_vid[real_neg_mask],
                    src_sent[real_neg_mask],
                )

                src_updated_neg, mask_updated_neg, pos_neg, attn_weights_neg = self.txt2vis_sal_encoder(
                    src=src_dummy_neg,
                    mask=mask_dummy_neg,
                    pos=pos_neg,
                    batch_video_len=video_length,
                    saliency_scores=torch.sigmoid(saliency_scores_neg),
                )

                encoder_output_neg = self.det_encoder(
                    src=src_updated_neg,
                    mask=mask_updated_neg,
                    pos=pos_neg,
                    video_length=video_length,
                )

                memory_neg = (
                    self.saliency_amplifier(
                        encoder_output_neg.memory,
                        saliency_scores_neg,
                        encoder_output_neg.vid_pos,
                        encoder_output_neg.vid_mask,
                    )
                    if self.saliency_amplifier is not None
                    else encoder_output_neg.memory
                )

                out["local_saliency_scores_neg"] = saliency_scores_neg
                if self.use_global_saliency_head:
                    saliency_scores_neg_offset = self.global_saliency_head(memory_neg)[:, :, 0]
                    out["saliency_scores_neg"] = saliency_scores_neg + saliency_scores_neg_offset.transpose(0, 1)
                else:
                    out["saliency_scores_neg"] = None

                # Prepare negative attention values (batch_size, L_vid, L_txt) / (batch_size, L_txt)
                if self.num_dummies == 0:
                    out["t2vattnvalues_neg"] = None
                else:
                    src_txt_mask_neg = input_src_txt_mask_neg[:, self.num_dummies :]
                    mask_unsq_neg = src_txt_mask_neg.unsqueeze(1).repeat(1, video_length, 1)
                    out["t2vattnvalues_neg"] = (attn_weights_neg[:, :, self.num_dummies :] * mask_unsq_neg).sum(2)
                    out["t2vattnvalues_neg"] = torch.clamp(out["t2vattnvalues_neg"], 0, 1)
        else:
            out["local_saliency_scores_neg"] = None
            out["saliency_scores_neg"] = None
            out["t2vattnvalues_neg"] = None
            out["real_neg_mask"] = None
        return out
