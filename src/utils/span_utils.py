"""Module for span-related utility functions."""  # noqa: WPS402

import math
from typing import Any, List, Tuple

import torch
from torch import Tensor

EPS = 1e-6


def span_xx_to_cxw(xx_spans: Tensor) -> Tensor:
    """
    Convert spans from start-end format to center-width format.

    Args:
        xx_spans (Tensor): Tensor of shape (#windows, 2), where each row represents a span with the format (start, end).

    Returns:
        Tensor: Tensor where each row represents a span in the format (center, width).

    Examples:
        >>> spans = torch.Tensor([[0, 1], [0.2, 0.4]])
        >>> span_xx_to_cxw(spans)
        tensor([[0.5000, 1.0000],
                [0.3000, 0.2000]])

        >>> spans = torch.Tensor([[[0, 1], [0.2, 0.4]]])
        >>> span_xx_to_cxw(spans)
        tensor([[[0.5000, 1.0000],
                [0.3000, 0.2000]]])
    """
    center = xx_spans.sum(-1) * 0.5
    width = xx_spans[..., 1] - xx_spans[..., 0]
    return torch.stack([center, width], dim=-1)


def span_cxw_to_xx(cxw_spans: Tensor) -> Tensor:
    """
    Convert spans from center-width format to start-end format.

    Args:
        cxw_spans (Tensor): Tensor of shape (#wndws, 2), each row represents a span with the format (center, width).

    Returns:
        Tensor: A tensor of the same shape as `cxw_spans`, where each row represents a span in the format (start, end).

    Examples:
        >>> spans = torch.Tensor([[0.5000, 1.0000], [0.3000, 0.2000]])
        >>> span_cxw_to_xx(spans)
        tensor([[0.0000, 1.0000],
                [0.2000, 0.4000]])

        >>> spans = torch.Tensor([[[0.5000, 1.0000], [0.3000, 0.2000]]])
        >>> span_cxw_to_xx(spans)
        tensor([[[0.0000, 1.0000],
                [0.2000, 0.4000]]])
    """
    x_start = cxw_spans[..., 0] - 0.5 * cxw_spans[..., 1]
    x_end = cxw_spans[..., 0] + 0.5 * cxw_spans[..., 1]
    return torch.stack([x_start, x_end], dim=-1)


def temporal_iou(spans1: Tensor, spans2: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Calculate the temporal Intersection over Union (IoU) and union of pairs of spans.

    Args:
        spans1 (Tensor): A (N, 2) torch.Tensor, where each row defines a span [start, end].
        spans2 (Tensor): A (M, 2) torch.Tensor, where each row defines a span [start, end].

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing two (N, M) tensors:
            - The first tensor contains the IoU for each pair of spans.
            - The second tensor contains the union for each pair of spans.

    Examples:
        >>> test_spans1 = torch.Tensor([[0, 0.2], [0.5, 1.0]])
        >>> test_spans2 = torch.Tensor([[0, 0.3], [0., 1.0]])
        >>> temporal_iou(test_spans1, test_spans2)
        (tensor([[0.6667, 0.2000],
                [0.0000, 0.5000]]),
        tensor([[0.3000, 1.0000],
                [0.8000, 1.0000]]))
    """
    areas1 = spans1[:, 1] - spans1[:, 0]  # (N, )
    areas2 = spans2[:, 1] - spans2[:, 0]  # (M, )

    left = torch.max(spans1[:, None, 0], spans2[:, 0])  # noqa: WPS221
    right = torch.min(spans1[:, None, 1], spans2[:, 1])  # noqa: WPS221

    inter = (right - left).clamp(min=0)  # (N, M)
    union = areas1[:, None] + areas2 - inter  # (N, M)

    iou = inter / union
    return iou, union


def temporal_intersection_over_pred(gt_spans: Tensor, pred_spans: Tensor) -> Tensor:  # noqa: WPS118
    """
    Calculate the Intersection over Prediction (IoP) for pairs of ground truth and predicted spans.

    This function computes the IoP for each pair of ground truth and predicted spans.
    The IoP is defined as the intersection of the spans divided by the span of the predicted span.

    Args:
        gt_spans (Tensor): A (N, 2) tensor, where each row represents a ground truth span [start, end].
        pred_spans (Tensor): A (M, 2) tensor, where each row represents a predicted span [start, end].

    Returns:
        Tensor: A (N, M) tensor containing the IoP for each pair of ground truth and predicted spans.
    """
    left = torch.max(gt_spans[:, None, 0], pred_spans[:, 0])  # noqa: WPS221
    right = torch.min(gt_spans[:, None, 1], pred_spans[:, 1])  # noqa: WPS221
    inter = (right - left).clamp(min=0)  # (N, M)
    return inter / (pred_spans[:, 1] - pred_spans[:, 0])  # inter_over_pred


def generalized_temporal_iou(spans1: Tensor, spans2: Tensor) -> Tensor:
    """
    Calculate the Generalized Intersection over Union (GIoU) for pairs of spans.

    This function computes the GIoU for each pair of spans. The GIoU is an extension of the IoU metric that also takes
    into account the size of the smallest enclosing span that contains both spans in the pair.
    It provides a more accurate measure of overlap, especially when the spans do not intersect.

    Args:
        spans1 (Tensor): A (N, 2) Tensor, where each row defines a span in [start, end] format.
        spans2 (Tensor): A (M, 2) Tensor, where each row defines a span in [start, end] format.

    Returns:
        Tensor: A (N, M) tensor containing the GIoU for each pair of spans.

    Examples:
        >>> test_spans1 = torch.Tensor([[0, 0.2], [0.5, 1.0]])
        >>> test_spans2 = torch.Tensor([[0, 0.3], [0., 1.0]])
        >>> generalized_temporal_iou(test_spans1, test_spans2)
        tensor([[ 0.6667,  0.2000],
                [-0.2000,  0.5000]])

    References:
        - Generalized IoU: https://giou.stanford.edu/
        - DETR implementation of gIoU: https://github.com/facebookresearch/detr/blob/master/util/box_ops.py#L40
    """
    spans1 = spans1.float()
    spans2 = spans2.float()
    assert (spans1[:, 1] >= spans1[:, 0]).all()
    assert (spans2[:, 1] >= spans2[:, 0]).all()
    iou, union = temporal_iou(spans1, spans2)

    left = torch.min(spans1[:, None, 0], spans2[:, 0])  # noqa: WPS221
    right = torch.max(spans1[:, None, 1], spans2[:, 1])  # noqa: WPS221
    enclosing_area = (right - left).clamp(min=0)  # (N, M)

    return iou - (enclosing_area - union) / enclosing_area


def temporal_intersection_criteria(pred_spans: Tensor, gt_spans: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Calculate the temporal Intersection over GT area and intersection of pairs of spans.

    Args:
        pred_spans (Tensor): A (N, 2) torch.Tensor, where each row defines a span [start, end].
        gt_spans (Tensor): A (M, 2) torch.Tensor, where each row defines a span [start, end].

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing two (N, M) tensors:
            - The first tensor contains the Intersection over GT
            - The second tensor contains the intersection for each pair of spans.

    Examples:
        >>> pred_spans = torch.Tensor([[0, 0.2], [0.5, 1.0]])
        >>> gt_spans = torch.Tensor([[0, 0.3], [0., 1.0]])
        >>> temporal_iou(pred_spans, gt_spans)
        (tensor([[0.6667, 0.2000],
                [0.0000, 0.5000]]),
        tensor([[0.2000, 0.2000],
                [0.0000, 0.5000]]))
    """
    gt_areas = gt_spans[:, 1] - gt_spans[:, 0]  # (M, )

    left = torch.max(pred_spans[:, None, 0], gt_spans[:, 0])  # noqa: WPS221
    right = torch.min(pred_spans[:, None, 1], gt_spans[:, 1])  # noqa: WPS221

    inter = (right - left).clamp(min=0)  # (N, M)
    return inter / (gt_areas + EPS), inter


class SpanList:
    """
    This class represents a set of spans.

    The spans are represented as a Nx2 Tensor.
    In order to uniquely determine the spans with respect to an video, we also store the corresponding video sizes.
    """

    def __init__(self, spans: Tensor, size: int, mode="xx"):
        """Initialize the SpanList.

        Args:
            spans (Tensor): A Nx2 Tensor representing the spans.
            size (int): The size of the video.
            mode (str): Format of the span. Defaults to "xx".

        Raises:
            ValueError: If the mode is not "xx" or "cxw".
        """
        if mode not in {"xx", "cxw"}:
            raise ValueError("mode should be 'xx' or 'cxw'")

        self.spans = spans
        self.size = size
        self.mode = mode
        self.extra_fields: dict = {}

    def add_field(self, field: str, field_data: Any):
        """Add a field to the SpanList.

        Args:
            field (str): The name of the field.
            field_data (Any): The data to be stored in the field.
        """
        self.extra_fields[field] = field_data

    def get_field(self, field: str) -> Any:
        """Get the data stored in a field.

        Args:
            field (str): The name of the field.

        Returns:
            Any: The data stored in the field.
        """
        return self.extra_fields[field]

    def fields(self) -> List[str]:
        """Get the names of the fields stored in the SpanList.

        Returns:
            List[str]: The names of the fields stored in the SpanList.
        """
        return list(self.extra_fields.keys())

    def copy_extra_fields(self, spans: "SpanList") -> None:
        """Copy the extra fields from another SpanList.

        Args:
            spans (SpanList): The SpanList to copy the extra fields from.
        """
        for key, value in spans.extra_fields.items():
            self.extra_fields[key] = value

    def _split_into_xx(self) -> Tuple[Tensor, Tensor]:
        """Split the spans into the format (xmin, xmax).

        Returns:
            Tuple[Tensor, Tensor]: The spans in the format (xmin, xmax).
        """
        if self.mode == "xx":
            xmin, xmax = self.spans.split(1, dim=-1)  # type: ignore
            return xmin, xmax  # noqa: WPS331
        xmin, width = self.spans.split(1, dim=-1)  # type: ignore
        return xmin, xmin + (width - 1).clamp(min=0)

    def convert(self, mode: str) -> "SpanList":
        """Convert the spans to a different format.

        Args:
            mode (str): The format to convert the spans to.

        Returns:
            "SpanList": The spans in the new format.

        Raises:
            ValueError: If the mode is not "xx" or "cxw".
        """
        if mode not in {"xx", "cxw"}:
            raise ValueError("mode should be 'xx' or 'cxw'")
        if mode == self.mode:
            return self

        xmin, xmax = self._split_into_xx()
        if mode == "xx":
            spans = torch.cat((xmin, xmax), dim=-1)
            spans_list = SpanList(spans, self.size, mode=mode)
        else:
            spans = torch.cat((xmin, xmax - xmin + 1), dim=-1)
            spans_list = SpanList(spans, self.size, mode=mode)
        spans_list.copy_extra_fields(self)
        return spans_list

    def __getitem__(self, item: int) -> "SpanList":
        """Get a subset of the spans.

        Args:
            item (int): The index of the span to get.

        Returns:
            SpanList: The subset of spans.
        """
        spans = SpanList(self.spans[item], self.size, self.mode)
        for key, value in self.extra_fields.items():
            spans.add_field(key, value[item])
        return spans

    def __len__(self) -> int:
        """Get the number of spans.

        Returns:
            int: The number of spans.
        """
        return self.spans.shape[0]

    def __repr__(self):
        """Get a string representation of the SpanList.

        Returns:
            str: A string representation of the SpanList.
        """
        string = self.__class__.__name__ + "("  # noqa: WPS336
        string += f"num_boxes={len(self)}, "  # noqa: WPS336,WPS237
        string += f"video_length={self.size}, "  # noqa: WPS336
        string += f"mode={self.mode})"  # noqa: WPS336
        return string


def cat_boxlist(spans: List["SpanList"]):
    """
    Concatenate a list of SpanList (having the same video size) into a single SpanList.

    Args:
        spans (List["SpanList"]): A list of SpanList to concatenate.

    Returns:
        List["SpanList"]: The concatenated SpanList.
    """
    assert isinstance(spans, (list, tuple))
    assert all(isinstance(span, SpanList) for span in spans)

    size = spans[0].size
    assert all(span.size == size for span in spans)

    mode = spans[0].mode
    assert all(span.mode == mode for span in spans)

    fields = set(spans[0].fields())
    assert all(set(span.fields()) == fields for span in spans)

    cat_boxes = SpanList(torch.cat([span.spans for span in spans], dim=0), size, mode)  # noqa: WPS221

    for field in fields:
        data = torch.cat([bbox.get_field(field) for bbox in spans], dim=0)
        cat_boxes.add_field(field, data)

    return cat_boxes


def encode_spans(gt_boxes: Tensor, anchors: Tensor) -> Tensor:
    """Encode spans into deltas between anchors and ground truth boxes.

    Args:
        gt_boxes (Tensor): Ground truth boxes in the format (start, end).
        anchors (Tensor): Anchors in the format (start, end).

    Returns:
        Tensor: Encoded deltas between anchors and ground truth boxes.
    """
    ex_widths = anchors[:, 1] - anchors[:, 0] + 1  # TODO: WHY DO WE ADD 1?
    ex_ctr_x = (anchors[:, 0] + anchors[:, 1]) / 2

    gt_widths = gt_boxes[:, 1] - gt_boxes[:, 0] + 1  # TODO: WHY DO WE ADD 1?
    gt_ctr_x = (gt_boxes[:, 0] + gt_boxes[:, 1]) / 2

    wx, ww = (10.0, 5.0)  # noqa: WPS111
    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dw = ww * torch.log(gt_widths / ex_widths)

    return torch.stack((targets_dx, targets_dw), dim=1)  # type: ignore


def decode_spans(preds: Tensor, anchors: Tensor) -> Tensor:
    """Decode deltas into spans.

    Args:
        preds (Tensor): Predictions in the format (dx, dw).
        anchors (Tensor): Anchors in the format (start, end).

    Returns:
        Tensor: Decoded spans.
    """
    anchors = anchors.to(preds.dtype)

    widths = anchors[:, 1] - anchors[:, 0] + 1  # TODO: WHY DO WE ADD 1?
    ctr_x = (anchors[:, 0] + anchors[:, 1]) / 2

    wx, ww = (10.0, 5.0)  # noqa: WPS111
    delta_w = preds[:, 1::2] / ww
    delta_x = preds[:, 0::2] / wx

    # Prevent sending too large values into torch.exp()
    delta_w = torch.clamp(delta_w, max=math.log(1000.0 / 16))  # noqa: WPS432

    pred_ctr_x = delta_x * widths[:, None] + ctr_x[:, None]
    pred_w = torch.exp(delta_w) * widths[:, None]

    pred_boxes = torch.zeros_like(preds)
    pred_boxes[:, 0::2] = pred_ctr_x - 0.5 * (pred_w - 1)
    pred_boxes[:, 1::2] = pred_ctr_x + 0.5 * (pred_w - 1)
    return pred_boxes
