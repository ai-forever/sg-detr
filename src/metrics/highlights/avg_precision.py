"""Average precision computation."""

from typing import Dict, Tuple, Union

import torch
from torch import Tensor
from torch import multiprocessing as mp  # noqa: WPS111
from torcheval.metrics import BinaryPrecisionRecallCurve


def get_ap(y_true: Tensor, y_predict: Tensor, interpolate: bool = True) -> Tensor:
    """Compute the average precision.

    Args:
        y_true (Tensor): The ground truth tensor.
        y_predict (Tensor): The predicted tensor.
        interpolate (bool): Whether to interpolate the precision-recall curve.

    Returns:
        Tensor: The average precision.
    """
    assert y_true.size(0) == y_predict.size(0), "Prediction and ground truth need to be of the same length"
    if torch.unique(y_true).size(0) == 1:
        if y_true[0] == 0:
            return torch.tensor(0).to(y_true.device)
        return torch.tensor(1).to(y_true.device)
    else:
        assert sorted(set(y_true.numpy())) == [0, 1], "Ground truth can only contain elements {0,1}"

    pr_curve = BinaryPrecisionRecallCurve()
    precision, recall, _ = pr_curve.update(y_predict, y_true.long()).compute()
    precision = precision.clone()

    if interpolate:  # Compute the interpolated precision
        for idx in range(1, len(precision)):  # noqa: WPS518
            precision[idx] = torch.max(precision[idx - 1], precision[idx])

    indices = torch.where(torch.diff(recall))
    return torch.mean(precision[indices])


def compute_ap_from_tuple(input_tuple: Tuple[int, int, Tensor, Tensor]) -> Tuple[int, int, Tensor]:
    """Compute the average precision from a tuple of input.

    Args:
        input_tuple (Tuple[int, int, Tensor, Tensor]): A tuple containing indexes, ground truth and prediction.

    Returns:
        Tuple[int, int, Tensor]: A tuple containing indexes and score.
    """
    idx, w_idx, y_true, y_predict = input_tuple
    if y_true.size(0) < y_predict.size(0):
        y_predict = y_predict[: y_true.size(0)]
    elif y_true.size(0) > y_predict.size(0):
        y_predict_new = torch.zeros_like(y_true, device=y_predict.device, dtype=torch.float32)
        y_predict_new[: y_predict.size(0)] = y_predict  # noqa: WPS362
        y_predict = y_predict_new

    score = get_ap(y_true, y_predict)
    return idx, w_idx, score


# pylint: disable=too-many-locals
def compute_hl_ap(  # noqa: WPS210
    qid2preds: Dict[str, Tensor],
    qid2gt_scores_binary: Dict[str, Tensor],
    device: Union[torch.device, str] = "cpu",
    num_workers: int = 1,
    chunksize: int = 50,
) -> Tensor:
    """Compute the average precision for highlight detection.

    Args:
        qid2preds (Dict[str, Tensor]): The model's predictions.
        qid2gt_scores_binary (Dict[str, Tensor]): The ground truth data.
        device (Union[torch.device, str]): The device to use.
        num_workers (int): The number of workers for multiprocessing.
        chunksize (int): The chunk size for multiprocessing.

    Returns:
        Tensor: Precision scores for each query and annotator.
    """
    ap_scores = torch.zeros((len(qid2preds), 3)).to(device)  # (#preds, 3)
    input_tuples = []
    for idx, qid in enumerate(qid2preds.keys()):
        for w_idx in range(3):  # annotation score idx
            y_true = qid2gt_scores_binary[qid][:, w_idx]
            y_predict = qid2preds[qid]
            input_tuples.append((idx, w_idx, y_true, y_predict))

    # Compute AP scores
    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            results = pool.imap_unordered(compute_ap_from_tuple, input_tuples, chunksize=chunksize)
            for idx, w_idx, score in results:
                ap_scores[idx, w_idx] = score
    else:
        for input_tuple in input_tuples:
            idx, w_idx, score = compute_ap_from_tuple(input_tuple)
            ap_scores[idx, w_idx] = score  # noqa: WPS441

    return ap_scores
