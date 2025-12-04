"""Analysis functions."""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.analysis.data_models import AuxiliaryOutput, VideoPrediction
from src.dataset.collate import move_inputs_to_device
from src.losses.matcher import HungarianMatcher
from src.model.model import MRDETR
from src.utils.span_utils import span_cxw_to_xx

MAX_SCORE: int = 12


def _get_saliency_scores(
    relevant_clip_idxs: List[int],
    saliency_clip_scores: List[List[int]],
    duration: int,
) -> np.ndarray:
    saliency_scores = np.zeros(duration // 2)
    for clip_idx, saliency_clip_score in zip(relevant_clip_idxs, saliency_clip_scores):
        saliency_scores[clip_idx] = np.sum(saliency_clip_score) / MAX_SCORE
    return saliency_scores


@torch.no_grad()
def compute_batch_predictions(
    meta: List[Dict[str, Any]],
    targets: Dict[str, Any],
    outputs: Dict[str, Any],
    matching: Dict[str, Any],
) -> List[VideoPrediction]:
    """
    Compute vide prediction for the batch.

    Args:
        meta (List[Dict[str, Any]]): matadata
        targets (Dict[str, Any]): targets
        outputs (Dict[str, Any]): model outputs
        matching (Dict[str, Any]): matching for each head

    Returns:
        List[VideoPrediction]: list of video predictions
    """
    video_predictions: List[VideoPrediction] = []
    # Retrieve the matching
    indices = matching["positive"]["indices"]
    batch_lenght = len(targets["span_labels"])  # type: ignore

    auxiliaries = []
    for idx, aux_decoder_outputs in enumerate(outputs["aux_outputs"]):
        auxiliary_indices = matching["positive_aux"]["indices"][idx]
        auxiliaries.append(
            (
                auxiliary_indices,
                aux_decoder_outputs["pred_logits"],
                aux_decoder_outputs["pred_spans"],
                aux_decoder_outputs["pred_quality_scores"],
            ),
        )
    for batch_idx in range(batch_lenght):
        aux_outputs = []

        for auxiliary in auxiliaries:
            aux_output = AuxiliaryOutput(
                pred_probs=torch.sigmoid(auxiliary[1][batch_idx])[:, 0].cpu().numpy(),  # noqa: WPS221, WPS204
                pred_spans=span_cxw_to_xx(auxiliary[2][batch_idx]).cpu().numpy(),
                pred_quality_scores=torch.sigmoid(auxiliary[3][batch_idx][:, 0]).cpu().numpy(),  # noqa: WPS221
                matching=(  # noqa: WPS221
                    auxiliary[0][batch_idx][0].cpu().numpy(),
                    auxiliary[0][batch_idx][1].cpu().numpy(),
                ),
            )
            aux_outputs.append(aux_output)

        saliency_scores = outputs["saliency_scores"]
        local_saliency_scores = outputs["local_saliency_scores"]

        pred_quality_scores = outputs["pred_quality_scores"][batch_idx][:, 0]
        video_prediction = VideoPrediction(
            vid=meta[batch_idx]["vid"],  # noqa: WPS204
            qid=meta[batch_idx]["qid"],
            gt_spans=span_cxw_to_xx(targets["span_labels"][batch_idx]["spans"]).cpu().numpy(),  # type: ignore
            pred_probs=torch.sigmoid(outputs["pred_logits"][batch_idx])[:, 0].cpu().numpy(),  # noqa: WPS221
            pred_spans=span_cxw_to_xx(outputs["pred_spans"][batch_idx]).cpu().numpy(),
            pred_quality_scores=torch.sigmoid(pred_quality_scores).cpu().numpy(),  # noqa: WPS221
            pred_saliency_scores=torch.sigmoid(saliency_scores[batch_idx]).cpu().numpy(),
            pred_local_saliency_scores=torch.sigmoid(local_saliency_scores[batch_idx]).cpu().numpy(),
            saliency_scores=_get_saliency_scores(
                meta[batch_idx]["relevant_clip_ids"],
                meta[batch_idx]["saliency_scores"],
                meta[batch_idx]["duration"],
            ),
            matching=(indices[batch_idx][0].cpu().numpy(), indices[batch_idx][1].cpu().numpy()),  # noqa: WPS221
            auxiliary=aux_outputs,
        )
        video_predictions.append(video_prediction)
    return video_predictions


@torch.no_grad()
def _compute_matching(
    outputs: Dict[str, Any],
    targets: Dict[str, Any],
    matcher: HungarianMatcher,
    ref_points: Optional[torch.Tensor],
) -> Dict[str, Any]:
    """
    Compute matching for detector heads include auxiliary heads.

    Also compute mean matching costs for each head.

    Args:
        outputs (Dict[str, Any]): model outputs
        targets (Dict[str, Any]): targets
        matcher (HungarianMatcher): span matcher
        ref_points (torch.Tensor): reference points for positive spans

    Returns:
        Dict[str, Any]: dict with matching indexes and corresponding costs.
    """
    matching: Dict[str, Any] = {}
    outputs_without_aux = {key: value for key, value in outputs.items() if key != "aux_outputs"}
    indices, costs = matcher(outputs_without_aux, targets, ref_points)
    matching["positive"] = {"indices": indices, "costs": costs}
    matching["positive_aux"] = {"indices": [], "costs": []}
    for aux_decoder_outputs in outputs["aux_outputs"]:
        auxiliary_indices, auxiliary_costs = matcher(aux_decoder_outputs, targets, ref_points)
        matching["positive_aux"]["indices"].append(auxiliary_indices)
        matching["positive_aux"]["costs"].append(auxiliary_costs)
    return matching


@torch.no_grad()
def compute_predictions(
    loader: DataLoader,
    matcher: HungarianMatcher,
    model: MRDETR,
    ref_points: Optional[torch.Tensor] = None,
) -> List[VideoPrediction]:
    """
    Compute video predictions for loader.

    Args:
        loader (DataLoader): data loader
        matcher (HungarianMatcher): span matcher
        model (MRDETR): moment retrievel model
        ref_points (torch.Tensor): positive anchors

    Returns:
        List[VideoPrediction]: list of prediction, in specific format
    """
    model.eval()
    device = next(model.parameters()).device
    video_predictions: List[VideoPrediction] = []
    for data in loader:
        meta, batch = data
        batch, targets = move_inputs_to_device(batch, device, non_blocking=True)

        outputs = model.forward(
            src_txt=batch["src_txt"],
            src_txt_mask=batch["src_txt_mask"],
            src_vid=batch["src_vid"],
            src_vid_mask=batch["src_vid_mask"],
            vid=batch["vid"],
        )
        if ref_points is None:
            ref_points = outputs["encoder_outputs"]["ref_points"]
        ref_points = ref_points.to(device)
        matching = _compute_matching(
            outputs=outputs,
            targets=targets,  # type: ignore
            matcher=matcher,
            ref_points=None,  # ref_points,
        )
        batch_video_predictions = compute_batch_predictions(
            meta=meta,
            targets=targets,  # type: ignore
            outputs=outputs,
            matching=matching,
        )
        video_predictions.extend(batch_video_predictions)
    return video_predictions
