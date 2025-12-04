"""Data schemas."""

from typing import List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, validator

from src.analysis.map import calculate_map


class AuxiliaryOutput(BaseModel):
    """Auxiliary Outputs schema."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    pred_probs: np.ndarray  # probs for each predicted spans
    pred_spans: np.ndarray  # normalized predicted spans
    pred_quality_scores: np.ndarray  # predicted iou score by the model
    matching: Tuple[np.ndarray, np.ndarray]  # pred idxs and gt idxes by matcher


class VideoPrediction(BaseModel):
    """Video Predictions schema."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    vid: str  # video id
    qid: int  # text query id
    gt_spans: np.ndarray  # normalized spans
    pred_probs: np.ndarray  # probs for each predicted spans
    pred_spans: np.ndarray  # normalized predicted spans
    pred_quality_scores: np.ndarray  # predicted iou score by the model
    pred_saliency_scores: np.ndarray  # predicted saliency score for each token in video
    pred_local_saliency_scores: np.ndarray  # predicted local saliency score for each token in video
    saliency_scores: np.ndarray  # saliency score for each token in video
    matching: Tuple[np.ndarray, np.ndarray]  # pred idxs and gt idxes by matcher

    auxiliary: List[AuxiliaryOutput]
    map: Optional[float] = None  # mean average precision score

    # pylint: disable=E0213:no-self-argument
    @validator("map", always=True, pre=True)
    def calculate_map_field(cls, _, values) -> float:  # noqa: N805
        """
        Calculate the MAP score for the VideoPrediction instance.

        Args:
            _: The value of the 'map' field (unused).
            values: Dictionary of field values for the VideoPrediction instance.

        Returns:
            float: The calculated MAP score.
        """
        gt_spans = values["gt_spans"]
        pred_spans = values["pred_spans"]
        pred_probs = values["pred_probs"]
        map_score, _ = calculate_map(gt_spans.tolist(), pred_spans.tolist(), pred_probs.tolist())
        return map_score
