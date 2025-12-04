"""ATSS head implementation."""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from src.model.blocks.anchors import AnchorGenerator
from src.model.blocks.conv_blocks import ConvBlock1D
from src.model.blocks.layers import Scale
from src.model.utils.schemas import AuxDetectorOutput
from src.utils.codetr_utils import prepare_matched_gt
from src.utils.span_utils import SpanList, span_cxw_to_xx, span_xx_to_cxw

INIT_CONST: float = 0.01


class ATSSClassificationHead(nn.Module):
    """A classification head of the ATSS."""

    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        num_convs: int = 3,
        prior_probability: float = 0.3,
    ) -> None:
        """Initialize classification head.

        Args:
            in_channels (int): number of channels of the input feature.
            num_anchors (int): number of anchors.
            num_convs (int): number of conv layer. Default: 3.
            prior_probability (float): probability of prior. Default: 0.3.
        """
        super().__init__()
        self.num_anchors = num_anchors
        self.stem = ConvBlock1D(
            in_channels=in_channels,
            hidden_dim=in_channels,
            out_channels=in_channels,
            num_layers=num_convs,
            kernel_size=3,
            last_activate=True,
        )
        self._stem_init()

        self.cls_logits = ConvBlock1D(
            in_channels=in_channels,
            hidden_dim=in_channels,
            out_channels=num_anchors,
            num_layers=1,
            kernel_size=3,
            use_norm=False,
        )
        self.cls_init(prior_probability)

    def _stem_init(self) -> None:
        """Init stem blocks."""
        for module in self.stem.modules():
            if isinstance(module, nn.Conv1d):
                torch.nn.init.normal_(module.weight, std=INIT_CONST)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)  # type: ignore

    def cls_init(self, prior_probability: float) -> None:
        """Init classification layer.

        Args:
            prior_probability (float): Prior prob.
        """
        for module in self.cls_logits.modules():
            if isinstance(module, nn.Conv1d):
                torch.nn.init.normal_(module.weight, std=INIT_CONST)
                if module.bias is not None:
                    prior_prob_logit = -math.log((1 - prior_probability) / prior_probability)
                    torch.nn.init.constant_(module.bias, prior_prob_logit)

    def forward(self, features: List[Tensor]) -> List[Tensor]:
        """Forward pass of the classification head.

        Args:
            features (List[Tensor]): FPN features.

        Returns:
            List[Tensor]: Predicted logits for each scale.
        """
        all_cls_logits = []
        for feature in features:
            feature = self.stem(feature)
            all_cls_logits.append(self.cls_logits(feature))
        return all_cls_logits


class ATSSRegressionHead(nn.Module):
    """A regression head to use in ATSS, which combines regression branch and center-ness branch."""

    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        num_convs: int = 3,
        fpn_strides: Tuple[float, ...] = (0.5, 1, 2, 4),
    ) -> None:
        """Initialize aux regression head.

        Args:
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors.
            num_convs (int): number of conv layer. Default: 2.
            fpn_strides (Tuple[float, ...]): fpn strides.
        """
        super().__init__()
        self.num_anchors = num_anchors
        self.stem = ConvBlock1D(
            in_channels=in_channels,
            hidden_dim=in_channels,
            out_channels=in_channels,
            num_layers=num_convs,
            kernel_size=3,
            last_activate=True,
        )

        self.bbox_reg = ConvBlock1D(
            in_channels=in_channels,
            hidden_dim=in_channels,
            out_channels=2 * num_anchors,
            num_layers=1,
            kernel_size=3,
            use_norm=False,
        )
        self.bbox_ctrness = ConvBlock1D(
            in_channels=in_channels,
            hidden_dim=in_channels,
            out_channels=num_anchors,
            num_layers=1,
            kernel_size=3,
            use_norm=False,
        )

        # init modules
        self._init_modules()

        # scale adjuster
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(len(fpn_strides))])  # noqa: WPS221
        self.fpn_strides = fpn_strides

    def _init_modules(self) -> None:
        """Init weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                torch.nn.init.normal_(module.weight, std=INIT_CONST)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, features: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """Forwar pass of the regression head.

        Args:
            features (List[Tensor]): input FPN embeddings.

        Returns:
            Tuple[List[Tensor], List[Tensor]]: [Offsets, Centerness]
        """
        all_bbox_regression = []
        all_bbox_ctrness = []

        for idx, feature in enumerate(features):
            bbox_feature = self.stem(feature)

            # predict centerness
            bbox_ctrness: Tensor = self.bbox_ctrness(bbox_feature)
            all_bbox_ctrness.append(bbox_ctrness)

            # predict spans
            bbox_regression: Tensor = self.bbox_reg(bbox_feature)
            bbox_regression = self.scales[idx](bbox_regression)
            all_bbox_regression.append(bbox_regression)

        return all_bbox_regression, all_bbox_ctrness


class ATSSHead(nn.Module):
    """A regression and classification head of the ATSS."""

    def __init__(
        self,
        in_channels: int,
        num_convs: int = 3,
        top_k_positive_anchors: int = 9,
        prior_probability: float = 0.3,
        fpn_strides: Tuple[float, ...] = (0.5, 1, 2, 4),
        anchor_sizes: Tuple[int, ...] = (4, 16, 32, 64),
    ) -> None:
        """Initialize ATSS module.

        Args:
            in_channels (int): number of channels of the input feature
            num_convs (int): number of conv layer of head. Default: 3.
            top_k_positive_anchors (int): num anchors to select as positive ones. Default: 9.
            prior_probability (float): probability of prior. Default: 0.3.
            fpn_strides (Tuple[float, ...]): fpn strides.
            anchor_sizes (Tuple[int, ...]): size of anchors.
        """
        super().__init__()
        self.fpn_strides = fpn_strides
        self.num_anchors = 1
        self.top_k_positive_anchors = top_k_positive_anchors
        self.classification_head = ATSSClassificationHead(in_channels, self.num_anchors, num_convs, prior_probability)
        self.regression_head = ATSSRegressionHead(in_channels, self.num_anchors, num_convs, fpn_strides)
        self.anchor_generator = AnchorGenerator(anchor_sizes, fpn_strides)

    def prepare_positive_locations(  # noqa: WPS210
        self,
        encoder_features: List[Tensor],
        anchors: List[List[SpanList]],
        targets: Dict[str, Any],
        meta: List[Dict[str, Any]],
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """Prepare positive locations for anchors based on targets.

        Args:
            encoder_features (List[Tensor]): Encoder features for each feature map.
            anchors (List[List[SpanList]]): List of anchor spans for each seq and level.
            targets (Dict[str, Any]): Dictionary containing target information.
            meta (List[Dict[str, Any]]): List of metadata dictionaries for each sequence.

        Returns:
            Tuple[List[Tensor], List[Tensor], List[Tensor]]:
                - List of matched ground truth spans for each seq.
                - List of anchor spans for each seq.
                - List of selected features for each seq.
        """
        targets_xx = [
            (span_cxw_to_xx(target["spans"]) * sample_meta["duration"]).type(torch.int) / 2
            for target, sample_meta in zip(targets["span_labels"], meta)
        ]

        cls_labels, matched_gts, anchors_all_lvls = prepare_matched_gt(targets_xx, anchors, self.top_k_positive_anchors)

        max_labels_num: int = 0
        for labels in cls_labels:
            cur_labels_sum = sum(labels)
            max_labels_num = max_labels_num if max_labels_num > cur_labels_sum else cur_labels_sum

        gt_all_seqs = []
        anchors_all_seqs = []
        selected_features = []
        stacked_encoder_features = torch.cat(encoder_features, 1)
        device = stacked_encoder_features.device
        for idx, (cls_labels_per_seq, matched_gts_per_seq, anchors_per_seq, feature_map) in enumerate(  # noqa: WPS352
            zip(cls_labels, matched_gts, anchors_all_lvls, stacked_encoder_features),
        ):
            # get padding size
            negative_padding_size = max_labels_num - cls_labels_per_seq.sum()

            # select positive and negative encoder features
            selected_features_seq = feature_map[cls_labels_per_seq.bool()]
            neg_inds = torch.nonzero(~cls_labels_per_seq.bool())
            indices = torch.randperm(neg_inds.shape[0], device=device)[:negative_padding_size]
            neg_inds = neg_inds[indices, 0]
            neg_selected_features_seq = feature_map[neg_inds]
            selected_features_seq = torch.cat((selected_features_seq, neg_selected_features_seq))

            # select positive gts and anchors
            matched_gts_per_seq = matched_gts_per_seq[cls_labels_per_seq.bool()]
            pos_anchors_per_seq = anchors_per_seq[cls_labels_per_seq.bool()]
            neg_anchors_per_seq = anchors_per_seq[neg_inds]
            anchors_per_seq = torch.cat((pos_anchors_per_seq, neg_anchors_per_seq))

            # convert positive anchors and gts
            matched_gts_per_seq = span_xx_to_cxw(matched_gts_per_seq)
            anchors_per_seq = span_xx_to_cxw(anchors_per_seq)
            matched_gts_per_seq = (matched_gts_per_seq / meta[idx]["duration"]) * 2
            anchors_per_seq = (anchors_per_seq / meta[idx]["duration"]) * 2

            gt_all_seqs.append(matched_gts_per_seq)
            anchors_all_seqs.append(anchors_per_seq)
            selected_features.append(selected_features_seq)

        return gt_all_seqs, anchors_all_seqs, selected_features

    def forward(
        self,
        fpn_features: List[Tensor],
        real_video_len: Tensor,
        targets: Optional[Dict[str, Any]] = None,
        meta: Optional[List[Dict[str, Any]]] = None,
    ) -> AuxDetectorOutput:
        """Forward pass of the model.

        Args:
            fpn_features (List[Tensor]): input FPN embeddings.
            real_video_len (Tensor): real video length.
            targets (Optional[Dict[str, Any]]): target information.
            meta (Optional[List[Dict[str, Any]]]): meta information.

        Returns:
            AuxDetectorOutput: Auxiliary head output schema.
        """
        cls_logits = self.classification_head(fpn_features)
        bbox_regression, bbox_ctrness = self.regression_head(fpn_features)
        anchors = self.anchor_generator(fpn_features, real_video_len)
        if targets is not None and meta is not None:
            matched_gts, anchors_spans, selected_features = self.prepare_positive_locations(
                fpn_features,
                anchors,
                targets,
                meta,
            )
        else:
            matched_gts, anchors_spans, selected_features = None, None, None
        return AuxDetectorOutput(
            cls_logits=cls_logits,
            bbox_regression=bbox_regression,
            bbox_ctrness=bbox_ctrness,
            anchors=anchors,
            matched_gts=matched_gts,
            anchors_spans=anchors_spans,
            selected_features=selected_features,
        )
