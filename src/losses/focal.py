"""Focal loss class."""

from functools import partial
from typing import Optional

import torch
from torch.nn import functional as func
from torch.nn.modules.loss import _Loss  # noqa: WPS450


def sigmoid_focal_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[float] = 0.25,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute binary focal loss between target and output logits.

    Source: https://github.com/BloodAxe/pytorch-toolbelt

    Args:
        outputs (torch.Tensor): tensor of arbitrary shape
        targets (torch.Tensor): tensor of the same shape as input
        gamma (float): gamma for focal loss. Defaults to 2.0.
        alpha (Optional[float]): alpha for focal loss. Defaults to 0.25.
        reduction (str): specifies the reduction to apply to the output

    Returns:
        torch.Tensor: computed loss
    """
    targets = targets.type(outputs.type())

    logpt = -func.binary_cross_entropy_with_logits(outputs, targets, reduction="none")
    pt = torch.exp(logpt)  # noqa: WPS111

    # compute the loss
    loss = -((1 - pt).pow(gamma)) * logpt

    if alpha is not None:
        alpha_seq = torch.tensor([alpha, 1 - alpha], device=targets.device)
        alpha_seq = alpha_seq.gather(0, targets.data.view(-1).type(torch.int64))  # noqa: WPS221
        loss = loss * alpha_seq

    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    if reduction == "batchwise_mean":
        loss = loss.sum(0)

    return loss


def reduced_focal_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute the Reduced Focal Loss between the target and output logits.

    This loss function is designed to mitigate the issue of class imbalance by
    reducing the focal loss for well-classified examples, ensuring that the model
    focuses more on hard, misclassified examples. It has been proposed in the
    'Reduced Focal Loss: 1st Place Solution to xView object detection in Satellite
    Imagery' paper.

    Note:
        The `size_average` and `reduce` parameters are being deprecated. Specifying
        either of these two args will override the `reduction` parameter in the meantime.

    Reference:
        - Reduced Focal Loss paper: https://arxiv.org/abs/1903.01347
        - Source code: https://github.com/BloodAxe/pytorch-toolbelt

    Args:
        outputs (torch.Tensor): The output tensor of arbitrary shape.
        targets (torch.Tensor): The target tensor of the same shape as `outputs`.
        threshold (float): The threshold for focal reduction. Defaults to 0.5.
        gamma (float): The focusing parameter for focal reduction. Defaults to 2.0.
        reduction (str): Specifies the method of reduction to apply to the output.

    Returns:
        torch.Tensor: The computed Reduced Focal Loss.
    """
    targets = targets.type(outputs.type())

    logpt = -func.binary_cross_entropy_with_logits(outputs, targets, reduction="none")
    pt = torch.exp(logpt)  # noqa: WPS111

    # compute the loss
    focal_reduction = ((1.0 - pt) / threshold).pow(gamma)
    focal_reduction[pt < threshold] = 1

    loss = -focal_reduction * logpt

    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    if reduction == "batchwise_mean":
        loss = loss.sum(0)

    return loss


class FocalLossBinary(_Loss):
    """Compute focal loss for binary classification problem.

    It has been proposed in `Focal Loss for Dense Object Detection`_ paper.

    .. _Focal Loss for Dense Object Detection: https://arxiv.org/abs/1708.02002
    """

    def __init__(
        self,
        ignore: Optional[int] = None,
        reduced: bool = False,
        gamma: float = 2.0,
        alpha: Optional[float] = 0.25,
        threshold: float = 0.5,
        reduction: str = "mean",
    ):
        """
        Initialize the FocalLossBinary class.

        Args:
            ignore (Optional[int]): Specifies a target value that is ignored and does not contribute to the gradient.
            reduced (bool): If True, use reduced focal loss. Defaults to False.
            gamma (float): Gamma factor for focal loss calculation. Defaults to 2.0.
            alpha (Optional[float]): Alpha factor for focal loss calculation, used to balance the importance of pos/neg.
            threshold (float): Threshold for focal reduction, applicable if reduced is True. Defaults to 0.5.
            reduction (str): Specifies the reduction to apply to the output.
        """
        super().__init__()
        self.ignore = ignore

        if reduced:
            self.loss_fn = partial(
                reduced_focal_loss,
                gamma=gamma,
                threshold=threshold,
                reduction=reduction,
            )
        else:
            self.loss_fn = partial(sigmoid_focal_loss, gamma=gamma, alpha=alpha, reduction=reduction)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the focal loss between logits and targets.

        Args:
            logits (torch.Tensor): Predictions from the model of shape [batch size; ...].
            targets (torch.Tensor): Ground truth values of shape [batch size; ...].

        Returns:
            torch.Tensor: Computed loss as a tensor.
        """
        targets = targets.reshape(-1)
        logits = logits.reshape(-1)

        if self.ignore is not None:
            # Filter predictions with ignore label from loss computation
            not_ignored = targets != self.ignore
            logits = logits[not_ignored]
            targets = targets[not_ignored]
        return self.loss_fn(logits, targets)


class FocalLossMultiClass(FocalLossBinary):
    """
    Compute focal loss for multiclass problem. Ignores targets having -1 label.

    It has been proposed in `Focal Loss for Dense Object Detection`_ paper.

    .. _Focal Loss for Dense Object Detection: https://arxiv.org/abs/1708.02002
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the focal loss between logits and targets for a multiclass classification problem.

        This method automatically handles the computation across all classes and ignores
        instances with a target label of -1, if specified in the `ignore` attribute.

        Args:
            logits (torch.Tensor): Predictions from the model with shape [batch size; num_classes; ...],
                where `num_classes` is the number of classes.
            targets (torch.Tensor): Ground truth labels with shape [batch size; ...]. Each label
                should be in the range [0, num_classes - 1] for valid classes, or -1 for ignored instances.

        Returns:
            torch.Tensor: The computed loss as a tensor. The loss is averaged across all non-ignored instances
            and summed across all classes, depending on the `reduction` attribute specified in the constructor.

        Note:
            The computation of loss takes into account the `gamma`, `alpha`, and `reduction` parameters
            specified during the initialization of the class instance. It adjusts the standard focal loss
            for cases where binary classification per class is performed within a multiclass setting.
        """
        num_classes = logits.size(1)
        loss = torch.tensor(0, device=logits.device)
        targets = targets.view(-1)
        logits = logits.view(-1, num_classes)

        # Filter anchors with -1 label from loss computation
        if self.ignore is not None:
            not_ignored = targets != self.ignore

        for class_id in range(num_classes):
            cls_label_target = (targets == (class_id + 0)).long()  # noqa: WPS345
            cls_label_input = logits[..., class_id]

            if self.ignore is not None:
                cls_label_target = cls_label_target[not_ignored]
                cls_label_input = cls_label_input[not_ignored]

            loss = loss + self.loss_fn(cls_label_input, cls_label_target)

        return loss
