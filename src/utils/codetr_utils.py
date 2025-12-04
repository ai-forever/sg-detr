"""Co-DETR utils."""

from typing import List, Tuple

import torch
from torch import Tensor

from src.utils.span_utils import SpanList, cat_boxlist, encode_spans, temporal_iou

INF: int = 100000000
EPS: float = 1e-7


def initialize_labels_targets(
    seq_idx: int,
    tgt_spans_per_seq: Tensor,
    anchors: List[List[SpanList]],
) -> Tuple[Tensor, SpanList, Tensor, Tensor, Tensor]:
    """Initialize labels and targets for a single seq.

    Args:
        seq_idx (int): Index of the seq in the batch.
        tgt_spans_per_seq (Tensor): Target spans for the seq.
        anchors (List[List[SpanList]]): List of anchor spans for each level.

    Returns:
        Tuple[Tensor, SpanList, Tensor, Tensor, Tensor]:
            - Labels per seq (Tensor).
            - Concatenated anchor spans per seq (SpanList).
            - Anchor points (Tensor).
            - IoU values (Tensor).
            - Distances between anchor points and target points (Tensor).
    """
    labels_per_seq = torch.ones(len(tgt_spans_per_seq), dtype=torch.int64, device=tgt_spans_per_seq.device)
    anchors_per_seq = cat_boxlist(anchors[seq_idx])
    ious, _ = temporal_iou(anchors_per_seq.spans, tgt_spans_per_seq)

    gt_points = (tgt_spans_per_seq[:, 0] + tgt_spans_per_seq[:, 1]) / 2
    anchor_points = (anchors_per_seq.spans[:, 0] + anchors_per_seq.spans[:, 1]) / 2  # noqa: WPS221
    distances = torch.abs(anchor_points[:, None] - gt_points[None, :])  # noqa: WPS221

    return labels_per_seq, anchors_per_seq, anchor_points, ious, distances


def select_candidate_anchors(
    distances: Tensor,
    num_anchors_per_level: List[int],
    top_k_positive_anchors: int,
) -> Tensor:
    """Select candidate anchors based on distances.

    Args:
        distances (Tensor): Distances between anchor points and target points.
        num_anchors_per_level (List[int]): Number of anchors at each level.
        top_k_positive_anchors (int): Number of top anchors to select per level.

    Returns:
        Tensor: Indices of the candidate anchors.
    """
    candidate_idxs = []
    star_idx = 0
    for level, _ in enumerate(num_anchors_per_level):
        end_idx = star_idx + num_anchors_per_level[level]
        distances_per_level = distances[star_idx:end_idx, :]
        topk = min(top_k_positive_anchors, num_anchors_per_level[level])
        _, topk_idxs_per_level = distances_per_level.topk(topk, dim=0, largest=False)
        candidate_idxs.append(topk_idxs_per_level + star_idx)
        star_idx = end_idx
    return torch.cat(candidate_idxs, dim=0)


def get_pos_based_on_iou_thresh(ious: Tensor, candidate_idxs: Tensor, num_gt: int) -> Tensor:
    """Determine positive samples based on IoU threshold.

    Args:
        ious (Tensor): IoU values between anchors and targets.
        candidate_idxs (Tensor): Indices of candidate anchors.
        num_gt (int): Number of ground truth targets.

    Returns:
        Tensor: Boolean tensor indicating positive samples.
    """
    candidate_ious = ious[candidate_idxs, torch.arange(num_gt)]
    iou_mean_per_gt = candidate_ious.mean(0)
    iou_std_per_gt = candidate_ious.std(0)
    iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt
    return candidate_ious >= iou_thresh_per_gt[None, :]


def finalize_positive_samples(
    is_pos: Tensor,
    candidate_idxs: Tensor,
    num_gt: int,
    anchor_points: Tensor,
    tgt_spans_per_seq: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Finalize positive samples by checking anchor points.

    Args:
        is_pos (Tensor): Boolean tensor indicating initial positive samples.
        candidate_idxs (Tensor): Indices of candidate anchors.
        num_gt (int): Number of ground truth targets.
        anchor_points (Tensor): Anchor points.
        tgt_spans_per_seq (Tensor): Target spans for the seq.

    Returns:
        Tuple[Tensor, Tensor]:
            - Updated boolean tensor indicating positive samples.
            - Concatenated candidate indices.
    """
    anchor_num = anchor_points.shape[0]
    for gt_idx in range(num_gt):
        candidate_idxs[:, gt_idx] += gt_idx * anchor_num
    cat_candidate_idxs = candidate_idxs.view(-1)
    expanded_anchors_cx = anchor_points.view(1, -1).expand(num_gt, anchor_num)
    expanded_anchors_cx = expanded_anchors_cx.contiguous().view(-1)

    # get candidate anchors
    expanded_anchors_cx = expanded_anchors_cx[cat_candidate_idxs].view(-1, num_gt)
    left = expanded_anchors_cx - tgt_spans_per_seq[:, 0]
    right = tgt_spans_per_seq[:, 1] - expanded_anchors_cx
    left_right = torch.stack([left, right], dim=1)
    is_in_gts = left_right.min(dim=1)[0] > EPS
    is_pos = is_pos & is_in_gts
    return is_pos, cat_candidate_idxs


def assign_cls_and_reg_targets(
    ious: Tensor,
    labels_per_seq: Tensor,
    tgt_spans_per_seq: Tensor,
    cat_candidate_idxs: Tensor,
    is_pos: Tensor,
    num_gt: int,
) -> Tuple[Tensor, Tensor]:
    """Assign classification and regression targets.

    Args:
        ious (Tensor): IoU values between anchors and targets.
        labels_per_seq (Tensor): Initial labels for the seq.
        tgt_spans_per_seq (Tensor): Target spans for the seq.
        cat_candidate_idxs (Tensor): Concatenated candidate indices.
        is_pos (Tensor): Boolean tensor indicating positive samples.
        num_gt (int): Number of ground truth targets.

    Returns:
        Tuple[Tensor, Tensor]:
            - Updated labels per seq.
            - Matched ground truth spans.
    """
    ious_inf = torch.full_like(ious, -INF)
    ious_inf = ious_inf.t().contiguous().view(-1)
    index = cat_candidate_idxs.view(-1)[is_pos.view(-1)]
    ious_inf[index] = ious.t().contiguous().view(-1)[index]
    ious_inf = ious_inf.view(num_gt, -1).t()

    anchors_to_gt_values, anchors_to_gt_indexs = ious_inf.max(dim=1)
    labels_per_seq = labels_per_seq[anchors_to_gt_indexs]
    labels_per_seq[anchors_to_gt_values == -INF] = 0
    matched_gts = tgt_spans_per_seq[anchors_to_gt_indexs]

    return labels_per_seq, matched_gts


def prepare_matched_gt(
    targets: List[Tensor],
    anchors: List[List[SpanList]],
    top_k_positive_anchors: int,
) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
    """Prepare matched gt spans.

    Args:
        targets (List[Tensor]): List of target spans.
        anchors (List[List[SpanList]]): List of anchor spans.
        top_k_positive_anchors (int): Number of top anchors to select per level.

    Returns:
        Tuple[List[Tensor], List[Tensor], List[Tensor]]:
            - List of classification labels.
            - List of regression targets.
            - List of anchors.
    """
    cls_labels: List[Tensor] = []
    matched_gts: List[Tensor] = []
    anchors_all_lvls: List[Tensor] = []

    for seq_idx, tgt_spans_per_seq in enumerate(targets):
        num_gt = tgt_spans_per_seq.shape[0]
        num_anchors_per_level = [len(anchors_per_level.spans) for anchors_per_level in anchors[seq_idx]]

        labels_per_seq, anchors_per_seq, anchor_points, ious, distances = initialize_labels_targets(  # noqa: WPS236
            seq_idx,
            tgt_spans_per_seq,
            anchors,
        )

        # Selecting candidates based on the center distance between anchor box and object
        cat_candidate_idxs = select_candidate_anchors(distances, num_anchors_per_level, top_k_positive_anchors)

        # Using the sum of mean and standard deviation as the IoU threshold to select final positive samples
        is_pos = get_pos_based_on_iou_thresh(ious, cat_candidate_idxs, num_gt)

        # Limiting the final positive samples’ center to object
        is_pos, cat_candidate_idxs = finalize_positive_samples(
            is_pos=is_pos,
            candidate_idxs=cat_candidate_idxs,
            num_gt=num_gt,
            anchor_points=anchor_points,
            tgt_spans_per_seq=tgt_spans_per_seq,
        )

        # if an anchor box is assigned to multiple gts, the one with the highest IoU will be selected.
        cls_labels_per_seq, matched_gts_per_seq = assign_cls_and_reg_targets(
            ious=ious,
            labels_per_seq=labels_per_seq,
            tgt_spans_per_seq=tgt_spans_per_seq,
            cat_candidate_idxs=cat_candidate_idxs,
            is_pos=is_pos,
            num_gt=num_gt,
        )
        cls_labels.append(cls_labels_per_seq)
        matched_gts.append(matched_gts_per_seq)
        anchors_all_lvls.append(anchors_per_seq.spans)

    return cls_labels, matched_gts, anchors_all_lvls


def prepare_targets(targets: List[Tensor], anchors: List[List[SpanList]], top_k_positive_anchors: int):
    """Prepare targets for ATSS loss computation.

    Args:
        targets (List[Tensor]): List of target spans.
        anchors (List[List[SpanList]]): List of anchor spans.
        top_k_positive_anchors (int): Number of top anchors to select per level.

    Returns:
        Tuple[List[Tensor], List[Tensor]]:
            - List of classification labels.
            - List of regression targets.
    """
    cls_labels, matched_gts, anchors_all_lvls = prepare_matched_gt(targets, anchors, top_k_positive_anchors)

    reg_targets = []
    for matched_gts_per_seq, anchors_per_seq in zip(matched_gts, anchors_all_lvls):
        reg_targets_per_im = encode_spans(matched_gts_per_seq, anchors_per_seq)
        reg_targets.append(reg_targets_per_im)
    return cls_labels, reg_targets
