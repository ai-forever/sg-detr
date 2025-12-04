"""Modules to compute the matching cost and solve the corresponding LSAP."""

from typing import Dict, List, Optional, Tuple

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

from src.utils.span_utils import generalized_temporal_iou, span_cxw_to_xx


# pylint: disable=too-many-locals
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    foreground_label = 0

    def __init__(
        self,
        cost_iou: float = 1,
        cost_class: float = 1,
        cost_span: float = 1,
        cost_giou: float = 1,
        cost_reference: float = 1,
    ):
        """Create the matcher.

        Args:
            cost_iou (float): Weight of the iou error in the matching cost
            cost_class (float): Weight of the classification error in the matching cost
            cost_span (float): Weight of the L1 error of the span coordinates in the matching cost
            cost_giou (float): Weight of the giou loss of the spans in the matching cost
            cost_reference (float): Weight of the reference distance cost.
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_span = cost_span
        self.cost_giou = cost_giou
        self.cost_iou = cost_iou
        self.cost_reference = cost_reference
        assert cost_class != 0 or cost_span != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(
        self,
        outputs: Dict[str, Tensor],
        targets: List[Dict[str, Tensor]],
        ref_points: Optional[Tensor],
    ):
        """
        Perform matching between predicted spans and target spans.

        It computes a cost matrix based on classification logits, L1 distance, and generalized IoU btwn the spans.
        It then applies a linear sum assignment algorithm to determine the best match between predictions and targets.

        Args:
            outputs (Dict[str, Tensor]): A dictionary containing "pred_spans" and "pred_logits" spans
            targets (List[Dict[str, Tensor]]): A list of target dicts #batch_size.
            ref_points (Optional[Tensor]): reference points.

        Returns:
            Tuple[List[Tuple[Tensor, Tensor]], Dict[str, Any]]:
                - matched indices: (selected predictions, corresponding selected targets)
                - cost statistics
        """
        batch_size, num_queries = outputs["pred_spans"].shape[:2]
        targets = targets["span_labels"]  # type: ignore

        matched_total_cost = torch.tensor(0, dtype=torch.float)  # noqa: WPS204
        matched_span_cost = torch.tensor(0, dtype=torch.float)
        matched_giou_cost = torch.tensor(0, dtype=torch.float)
        matched_iou_cost = torch.tensor(0, dtype=torch.float)
        matched_class_cost = torch.tensor(0, dtype=torch.float)
        matched_reference_cost = torch.tensor(0, dtype=torch.float)

        # calculate number of spans
        n_gt_spans = sum([len(item["spans"]) for item in targets])

        # Prepare target labels and spans
        tgt_spans = torch.cat([target["spans"] for target in targets])  # [num_target_spans in batch, 2]
        tgt_ids = torch.full([len(tgt_spans)], self.foreground_label)  # [total #spans in the batch]

        # We flatten to compute the cost matrices in a batch
        if ref_points is not None:
            pred_diffs = torch.abs(span_cxw_to_xx(outputs["pred_spans"]).detach() - ref_points)
            pred_diffs = torch.sqrt(pred_diffs[:, :, 0] ** 2 + pred_diffs[:, :, 1] ** 2)  # noqa: WPS221
            pred_diffs = pred_diffs.flatten()
            cost_reference = pred_diffs.unsqueeze(1).repeat(1, len(tgt_spans))
        else:
            cost_reference = torch.zeros_like(outputs["pred_spans"][..., 0].flatten())
            cost_reference = cost_reference.unsqueeze(1).repeat(1, len(tgt_spans))

        # Compute the L1 cost between spans
        out_spans = outputs["pred_spans"].flatten(0, 1)
        cost_span = torch.cdist(out_spans, tgt_spans, p=1)

        # Compute the giou cost between spans
        cost_giou = -generalized_temporal_iou(span_cxw_to_xx(out_spans), span_cxw_to_xx(tgt_spans))

        # Compute the classification cost. Contrary to the loss, we don't use the NLL
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, 1]
        cost_class = -out_prob[:, tgt_ids]

        # Compute the IoU cost
        if "pred_quality_scores" in outputs:
            out_iou = outputs["pred_quality_scores"].flatten(0, 1).sigmoid()
            cost_iou = -out_iou[:, tgt_ids]
        else:
            cost_iou = torch.zeros_like(cost_class)

        # Final cost matrix
        cost_matrix = self.cost_span * cost_span
        cost_matrix += self.cost_giou * cost_giou
        cost_matrix += self.cost_class * cost_class
        cost_matrix += self.cost_iou * cost_iou
        cost_matrix += self.cost_reference * cost_reference

        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()
        cost_span = cost_span.view(batch_size, num_queries, -1).cpu()
        cost_giou = cost_giou.view(batch_size, num_queries, -1).cpu()
        cost_class = cost_class.view(batch_size, num_queries, -1).cpu()
        cost_iou = cost_iou.view(batch_size, num_queries, -1).cpu()
        cost_reference = cost_reference.view(batch_size, num_queries, -1).cpu()
        sizes = [len(target["spans"]) for target in targets]
        matrixs = zip(
            cost_matrix.split(sizes, -1),  # type: ignore
            cost_span.split(sizes, -1),  # type: ignore
            cost_giou.split(sizes, -1),  # type: ignore
            cost_class.split(sizes, -1),  # type: ignore
            cost_iou.split(sizes, -1),  # type: ignore
            cost_reference.split(sizes, -1),  # type: ignore
        )
        indices: List[Tuple[Tensor, Tensor]] = []
        for idx, (cost, c_span, c_giou, c_class, c_iou, c_reference) in enumerate(matrixs):  # type: ignore
            selected_pred, selected_target = linear_sum_assignment(cost[idx])
            selected_pred = torch.as_tensor(selected_pred, dtype=torch.int64)
            selected_target = torch.as_tensor(selected_target, dtype=torch.int64)
            indices.append((selected_pred, selected_target))
            index = torch.stack((selected_pred, selected_target), dim=1)

            matched_total_cost += cost[idx][index[:, 0], index[:, 1]].sum()  # noqa: WPS221 WPS204
            matched_span_cost += c_span[idx][index[:, 0], index[:, 1]].sum()  # noqa: WPS221
            matched_giou_cost += c_giou[idx][index[:, 0], index[:, 1]].sum()  # noqa: WPS221
            matched_iou_cost += c_iou[idx][index[:, 0], index[:, 1]].sum()  # noqa: WPS221
            matched_class_cost += c_class[idx][index[:, 0], index[:, 1]].sum()  # noqa: WPS221
            matched_reference_cost += c_reference[idx][index[:, 0], index[:, 1]].sum()  # noqa: WPS221
        costs = {
            "matched_total_cost": matched_total_cost,
            "matched_span_cost": matched_span_cost,
            "matched_giou_cost": matched_giou_cost,
            "matched_iou_cost": matched_iou_cost,
            "matched_class_cost": matched_class_cost,
            "matched_reference_cost": matched_reference_cost,
            "n_gt_spans": n_gt_spans,
        }
        return indices, costs
