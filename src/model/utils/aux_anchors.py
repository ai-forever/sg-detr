"""Aux anchors utils."""

from typing import Any, Dict, List, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import Embedding

from src.model.utils.model_utils import inverse_sigmoid
from src.utils.span_utils import span_cxw_to_xx, span_xx_to_cxw


# pylint: disable=R0913,R0914
def prepare_anchors_codetr(  # noqa: WPS210, WPS234
    linear_mapper: Union[nn.Linear, nn.Sequential],
    matched_gts: List[torch.Tensor],
    anchors_per_seq: List[torch.Tensor],
    encoder_features_per_seq: List[torch.Tensor],
) -> Tuple[Tensor, Tensor, Dict[str, Any]]:  # noqa: WPS221
    """
    Add anchors from aux head as support reference points for detr during training.

    Args:
        linear_mapper (nn.Linear): linear mapper.
        matched_gts (List[torch.Tensor]): gt spans mathced to selected anchors.
        anchors_per_seq (List[torch.Tensor]): selected anchors from aux head.
        encoder_features_per_seq (List[torch.Tensor]): selected encoder features.

    Returns:
        Tuple[Tensor, Tensor, Dict[str, Any]]:
            - input_query_label: label query embedding for detr decoder
            - input_query_spans: reference points for detr decoder
            - mask_dict: aux information for support reference points. None if no support spans
    """
    device = encoder_features_per_seq[0].device
    pad_size = len(anchors_per_seq[0])

    # prepare gt spans and labels
    known = [torch.nonzero(torch.ones(len(span))) for span in matched_gts]  # noqa: WPS221
    positive_inds = torch.cat([ones + idx * pad_size for idx, ones in enumerate(known)]).to(device)  # noqa: WPS221

    gt_labels = torch.zeros(len(anchors_per_seq) * pad_size, device=device)
    gt_labels[positive_inds[..., 0]] = 1
    gt_spans = torch.cat(matched_gts)  # flatten the batch bboxes

    # get content query
    encoder_features = torch.stack(encoder_features_per_seq)
    input_query_label = linear_mapper(encoder_features)

    # get pos queries
    anchors_spans = torch.stack(anchors_per_seq)
    input_query_span = inverse_sigmoid(anchors_spans).to(torch.float32)

    input_query_label = input_query_label.transpose(0, 1)
    input_query_span = input_query_span.transpose(0, 1)

    mask_dict = {
        "known_lbs_bboxes": (gt_labels, gt_spans),
        "pad_size": pad_size,
    }
    return input_query_label, input_query_span, mask_dict


# pylint: disable=R0913,R0914
def prepare_anchors_codetr_legacy(  # noqa: WPS210, WPS234
    linear_mapper: nn.Linear,
    matched_gts: List[torch.Tensor],
    anchors_per_seq: List[torch.Tensor],
    encoder_features_per_seq: List[torch.Tensor],
    batch_size: int = 512,
) -> Tuple[Tensor, Tensor, Dict[str, Any]]:  # noqa: WPS221
    """
    Add anchors from aux head as support reference points for detr during training.

    Args:
        linear_mapper (nn.Linear): linear mapper.
        matched_gts (List[torch.Tensor]): gt spans mathced to selected anchors.
        anchors_per_seq (List[torch.Tensor]): selected anchors from aux head.
        encoder_features_per_seq (List[torch.Tensor]): selected encoder features.
        batch_size (int): batch size

    Returns:
        Tuple[Tensor, Tensor, Dict[str, Any]]:
            - input_query_label: label query embedding for detr decoder
            - input_query_spans: reference points for detr decoder
            - mask_dict: aux information for support reference points. None if no support spans
    """
    device = encoder_features_per_seq[0].device

    known = [torch.ones(len(span)) for span in matched_gts]  # replace bboxes with ones in batch
    know_idx = [torch.nonzero(idxs) for idxs in known]  # enumerate bboxes of each object in the batch
    known_num = [sum(idxs) for idxs in known]  # count bboxes in the batch

    # prepare indices
    tmp = torch.cat(known)  # flatten the indices
    known_indice = torch.nonzero(tmp).view(-1)  # enumerate all bboxes
    known_indice = known_indice.to(device)

    # prepare spans and labels
    gt_spans = torch.cat(matched_gts)  # flatten the batch bboxes
    gt_labels = torch.cat(known).long().to(device)
    anchors_spans = torch.cat(anchors_per_seq)

    # get the batch index for each span
    batch_idx = torch.cat([torch.full_like(ones, idx) for idx, ones in enumerate(known)])  # noqa: WPS221
    batch_idx = batch_idx.to(device)

    # padding shapes
    pad_size = int(max(known_num))  # max anchors per seq

    # add padding to labels
    encoder_features = torch.cat(encoder_features_per_seq)
    input_label_embed = linear_mapper(encoder_features)
    padding_label = torch.zeros(pad_size, input_label_embed.size(1)).to(device)
    input_query_label = padding_label.repeat(batch_size, 1, 1)

    # add padding to spans
    input_bbox_embed = inverse_sigmoid(anchors_spans).to(torch.float32)
    padding_bbox = torch.zeros(pad_size, 2, device=device)
    input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

    # map in order
    map_known_indice = torch.cat([torch.tensor(range(int(num))) for num in known_num]).long()  # noqa: WPS221
    input_query_label[(batch_idx.long(), map_known_indice)] = input_label_embed
    input_query_bbox[(batch_idx.long(), map_known_indice)] = input_bbox_embed

    input_query_label = input_query_label.transpose(0, 1)
    input_query_bbox = input_query_bbox.transpose(0, 1)

    mask_dict = {
        "known_indice": torch.as_tensor(known_indice).long(),
        "batch_idx": torch.as_tensor(batch_idx).long(),
        "map_known_indice": torch.as_tensor(map_known_indice).long(),
        "known_lbs_bboxes": (gt_labels, gt_spans),
        "know_idx": know_idx,
        "pad_size": pad_size,
    }

    return input_query_label, input_query_bbox, mask_dict


def recalculate_num_groups(num_groups: int, known_num: List[int]) -> int:
    """Recalculate num groups based on max number of spans.

    Args:
        num_groups (int): initial num of groups
        known_num (List[int]): number of gt spans in each sample

    Returns:
        int: recalulated num of groups
    """
    if int(max(known_num)) == 0:
        num_groups = 1
    else:
        num_groups = num_groups // (int(max(known_num) * 2))
    if num_groups < 1:
        num_groups = 1
    return num_groups


# pylint: disable=R0915
def prepare_anchors_dn(  # noqa: WPS210
    label_enc: Embedding,
    targets: Dict[str, Any],
    num_groups: int = 5,
    span_noise_scale: float = 0.4,
    negative_offset: float = 1.0,
    batch_size: int = 512,
) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
    """
    Add noise to gt spans and use them as support regerence points during training.

    Args:
        label_enc (Embedding): target embeddings.
        targets (Dict[str, Any]): target dict contains "span_labels"
        num_groups (int): number of denoise groups
        span_noise_scale (float): noise scale for span coords
        negative_offset (float): offset for negative samples
        batch_size (int): batch size

    Returns:
        Tuple[Tensor, Tensor, Dict[str, Any]]:
            - input_query_label: label query embedding for detr decoder
            - input_query_spans: reference points for detr decoder
            - mask_dict: aux information for support reference points. None if no support spans
    """
    device = label_enc.weight.device
    tg_spans = targets["span_labels"]
    known = [torch.ones(len(span["spans"])) for span in tg_spans]  # replace spans with ones in batch
    known_num = [sum(idxs) for idxs in known]  # count spans in the batch

    num_groups = recalculate_num_groups(num_groups, known_num)

    # prepare for the dn part
    tmp = torch.cat(known)  # flatten the indices
    known_indice = torch.nonzero(tmp).view(-1)  # enumerate all spans

    # get the batch index for each span
    spans = torch.cat([span["spans"] for span in tg_spans])  # flatten the batch spans
    labels = torch.cat(known).long()
    known_indice = known_indice.to(device)
    labels = labels.to(device)
    batch_idx = torch.cat([torch.full_like(ones, idx) for idx, ones in enumerate(known)])  # noqa: WPS221
    batch_idx = batch_idx.to(device)

    # add noise
    double_groups = 2 * num_groups
    known_indice = known_indice.repeat(double_groups, 1).view(-1)
    known_labels = labels.repeat(double_groups, 1).view(-1)
    known_bid = batch_idx.repeat(double_groups, 1).view(-1)
    known_spans = spans.repeat(double_groups, 1)
    known_labels_expaned = known_labels.clone()
    known_spans_expand = known_spans.clone()

    # padding shapes
    single_pad = int(max(known_num))
    pad_size = int(single_pad * double_groups)

    # positive/negative indices
    positive_idx = torch.tensor(range(len(spans)), dtype=torch.long, device=device)
    positive_idx = positive_idx[None].repeat(num_groups, 1)
    pos_shift = torch.tensor(range(num_groups), dtype=torch.long, device=device) * len(spans) * 2  # noqa: WPS221
    positive_idx += pos_shift.unsqueeze(1)
    positive_idx = positive_idx.flatten()
    negative_idx = positive_idx + len(spans)

    # apply noise on the box
    known_spans_expand_xx = span_cxw_to_xx(known_spans_expand)
    diff = torch.zeros_like(known_spans_expand, device=device)
    diff[:, :1] = known_spans_expand[:, 1:] / 2  # center diff
    diff[:, 1:] = known_spans_expand[:, 1:] / 2  # width diff

    rand_sign = torch.randint_like(known_spans_expand, low=0, high=2, dtype=torch.float32)  # exclusive
    rand_sign = rand_sign * 2.0 - 1.0  # noqa: WPS432
    rand_part = torch.rand_like(known_spans_expand)
    rand_part[negative_idx] += negative_offset
    modulation = rand_part * rand_sign
    modulated_diff = torch.mul(modulation, diff) * span_noise_scale

    # make diff scale dependent
    known_spans_expand_xx = known_spans_expand_xx + modulated_diff
    known_spans_expand_xx = known_spans_expand_xx.clamp(min=0.0, max=1.0)
    known_spans_expand = span_xx_to_cxw(known_spans_expand_xx)

    # prepare labels
    input_label_embed = label_enc(known_labels_expaned - 1)
    padding_label = torch.zeros(pad_size, batch_size, input_label_embed.size(1)).to(device)
    input_query_label = padding_label.transpose(0, 1)

    # prepare spans
    input_span_embed = inverse_sigmoid(known_spans_expand)
    input_query_spans = torch.zeros(batch_size, pad_size, 2, device=device)

    # map in order
    map_known_indice = torch.cat([torch.tensor(range(int(num))) for num in known_num])  # noqa: WPS221
    map_known_indice = torch.cat([map_known_indice + single_pad * idx for idx in range(double_groups)])  # noqa: WPS221
    map_known_indice = map_known_indice.long()
    input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
    input_query_spans[(known_bid.long(), map_known_indice)] = input_span_embed

    mask_dict = {
        "pad_size": pad_size,
        "num_groups": num_groups,
    }

    input_query_label = input_query_label.transpose(0, 1)
    input_query_spans = input_query_spans.transpose(0, 1)

    return input_query_label, input_query_spans, mask_dict


def aux_post_process(
    outputs_class: torch.Tensor,
    outputs_coord: torch.Tensor,
    quality_scores: torch.Tensor,
    offsets: torch.Tensor,
    co_dict: Dict[str, Any],
    dn_dict: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Separate denoise part from the output and put it in the mask_dict.

    Args:
        outputs_class (torch.Tensor): known and predicted classes.
        outputs_coord (torch.Tensor): known and predicted spans.
        quality_scores (torch.Tensor): known and predicted iou scores.
        offsets (torch.Tensor): calculated offsets.
        co_dict (Dict[str, Any]): collab paddings.
        dn_dict (Dict[str, Any]): dn paddings.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Predicted classes and spans, etc.
    """
    co_pad_size = co_dict["pad_size"] if co_dict is not None else 0
    dn_pad_size = dn_dict["pad_size"] if dn_dict is not None else 0
    tgt_pad_size = dn_pad_size + co_pad_size

    tgt_outputs_class = outputs_class[:, :, tgt_pad_size:, :]
    tgt_outputs_coord = outputs_coord[:, :, tgt_pad_size:, :]
    if quality_scores is not None:
        quality_scores = quality_scores[:, :, tgt_pad_size:, :]
    if offsets is not None:
        offsets = offsets[:, :, tgt_pad_size:, :]

    if dn_pad_size > 0:
        output_dn_class = outputs_class[:, :, :dn_pad_size, :]
        output_dn_coord = outputs_coord[:, :, :dn_pad_size, :]
        dn_dict["output_known_lbs_bboxes"] = (output_dn_class, output_dn_coord)

    if co_pad_size > 0:
        output_co_class = outputs_class[:, :, dn_pad_size:tgt_pad_size, :]
        output_co_coord = outputs_coord[:, :, dn_pad_size:tgt_pad_size, :]
        co_dict["output_known_lbs_bboxes"] = (output_co_class, output_co_coord)
    return tgt_outputs_class, tgt_outputs_coord, quality_scores, offsets
