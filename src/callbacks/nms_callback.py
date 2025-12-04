"""NMS callback."""

from copy import deepcopy
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from pytorch_lightning import Callback, LightningModule, Trainer

from src.dataset.collate import move_inputs_to_device
from src.metrics.moments.metrics import MRAveragePrecision
from src.postprocessor.postprocessing import PostProcessorDETR, Preparator
from src.postprocessor.utils import general_nms

# import multiprocessing as mp
# import os


def calculate_iou_1d(span_a: Tuple[float, float], span_b: Tuple[float, float]) -> float:  # noqa: WPS114
    """
    Calculate the Intersection over Union (IoU) of two spans.

    Args:
        span_a (Tuple[float, float]): Start and end of the first span.
        span_b (Tuple[float, float]): Start and end of the second span.

    Returns:
        float: The IoU between span_a and span_b.
    """
    start_a, end_a = span_a
    start_b, end_b = span_b

    # Find the intersection of the spans
    intersection_start = max(start_a, start_b)
    intersection_end = min(end_a, end_b)
    intersection = max(intersection_end - intersection_start, 0)

    # Find the union of the spans
    union = max(end_a, end_b) - min(start_a, start_b)

    # Calculate the IoU
    return intersection / union if union > 0 else 0


def _evaluate_with_nms(data: List[Any]) -> Tuple[Dict[str, torch.Tensor], float, str]:
    predictions, metas, threshold, nms_attribute = data
    attribute_idx = 2 if nms_attribute == "probs" else 3
    # since the data is transmitted by link, a full copy of it is made
    nms_predictions = deepcopy(predictions)
    # NMS applies to all predictions
    for item in nms_predictions:  # noqa:WPS426
        idxes = general_nms(
            item["pred_relevant_windows"],
            score_function=lambda x: x[attribute_idx],  # noqa: WPS111
            iou_function=lambda x, y: calculate_iou_1d(x[:2], y[:2]),  # noqa: WPS111
            threshold=threshold,
        )
        item["pred_relevant_windows"] = item["pred_relevant_windows"][idxes]
    # the metric is calculated
    test_metric = MRAveragePrecision(window_range="full")
    test_metric(nms_predictions, metas)
    return test_metric.compute(), threshold, nms_attribute


def greed_search(predictions: List[Any], meta: List[Any], target_metric: str) -> Tuple[float, str]:
    """
    Iterate through all possible options for NMS and selects the best MAP by metric.

    Args:
        predictions (List[Any]): predictions
        meta (List[Any]): targets
        target_metric (str): type of MAP metric to maximize

    Returns:
        Tuple[float, str]: NMS best params(threshold and attribute)
    """
    # A pool of all possible options is being created.
    greed_data = []
    for threshold in np.linspace(0.5, 0.9, 41):  # noqa: WPS432
        greed_data.append((predictions, meta, threshold, "probs"))
    for threshold in np.linspace(0.5, 0.9, 41):  # noqa: WPS432
        greed_data.append((predictions, meta, threshold, "iou_score"))
    # calculating options
    metrics_data = [_evaluate_with_nms(data) for data in greed_data]  # type: ignore

    # Fix problems with multiprocessing
    # cpu_to_use = min(np.max([os.cpu_count() - 3, 1]), len(greed_data))
    # with mp.Pool(processes=cpu_to_use) as pool:
    #     metrics_data = list(tqdm(pool.imap(_evaluate_with_nms, greed_data)))
    # determining the best parameters
    best_idx = np.argmax([item[0][target_metric] for item in metrics_data])
    _, best_threshold, best_nms_attribute = metrics_data[best_idx]
    return best_threshold, best_nms_attribute


class NMSTuningCallback(Callback):
    """Callback finds optimal metrics for NMS (in postprocessing) before the testing and overrides the postprocessor."""

    def __init__(self, target_metric: str = "Avg") -> None:
        """
        Initialize NMSTuningCallback.

        Args:
            target_metric (str): The type of MAP metric that will be maximized. Defaults to "Avg".
        """
        super().__init__()
        self.target_metric = target_metric
        self.preparator = Preparator()

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Determine the best NMS postprocessing parameters on the validation dataset and redefines the postprocessor.

        Args:
            trainer (Trainer): pytorch lightning trainer
            pl_module (LightningModule): pl modules
        """
        runner = trainer.model
        # get postprocessor and create a complete duplicate of the postprocessor and at the same time delete the NMS
        # parameters, if they exist (select them separately)
        postprocessor = deepcopy(runner.postprocessor)  # type: ignore
        postprocessor.nms_threshold = None  # no `nms_threshold` -> no NMS
        predictions, meta = self.get_valid_predictions(trainer)
        best_threshold, best_nms_attribute = greed_search(predictions, meta, self.target_metric)
        # Redefining the postprocessor with best NMS params
        runner.postprocessor = PostProcessorDETR(  # type: ignore
            clip_length=postprocessor.clip_length,
            min_ts_val=postprocessor.min_ts_val,
            max_ts_val=postprocessor.max_ts_val,
            min_w_l=postprocessor.min_w_l,
            max_w_l=postprocessor.max_w_l,
            nms_threshold=best_threshold,
            nms_attribute=best_nms_attribute,
        )

    @torch.no_grad()
    def get_valid_predictions(self, trainer: Trainer) -> Tuple[Any, Any]:
        """
        Compute valid predictions and targets.

        Args:
            trainer (Trainer): pytorch lightning trainer

        Returns:
            Tuple[Any, Any]: all predictions(in numpy) and metadata(as target)
        """
        runner = trainer.model
        predictions = []
        meta = []
        runner.model.eval()  # type: ignore
        with torch.no_grad():
            for data in trainer.datamodule.val_dataloader():  # type: ignore
                _meta, batch = data
                batch, _ = move_inputs_to_device(batch, runner.device, non_blocking=True)  # type: ignore
                outputs = runner.model(**batch)  # type: ignore
                prepared_outputs = self.preparator.prepare_detr_outputs(_meta, batch, outputs)
                postprocessed_results = runner.postprocessor(prepared_outputs)  # type: ignore
                # Due to problems with parallelism in torch, use numpy arrays
                for pred in postprocessed_results:
                    pred["pred_relevant_windows"] = pred["pred_relevant_windows"].cpu().numpy()
                    pred["pred_saliency_scores"] = pred["pred_saliency_scores"].cpu().numpy()
                meta.extend(_meta)
                predictions.extend(postprocessed_results)
        return predictions, meta
