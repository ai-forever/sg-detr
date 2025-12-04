"""Boostrap callbacks."""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from clearml.logger import Logger as ClearMLLogger
from joblib import Parallel, delayed
from pytorch_lightning import Callback, LightningModule, Trainer
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection

from src.analysis.metrics import plot_confidence_intervals
from src.dataset.collate import move_inputs_to_device
from src.metrics.moments.metrics import MRAveragePrecision, MRRecallAt1
from src.model.model import MRDETR
from src.postprocessor.postprocessing import PostProcessorDETR, Preparator


def get_metrics(**kwargs) -> MetricCollection:
    """Generate a collection of essential evaluation metrics.

    This function creates a collection of metrics HL/MR tasks.

    Args:
        kwargs: Arbitrary keyword arguments that are forwarded to the initialization of each metric.

    Returns:
        MetricCollection: A collection of initialized metrics.
    """
    return MetricCollection(
        {
            "MR-mAP-Short": MRAveragePrecision(window_range="short", **kwargs),
            "MR-mAP-Middle": MRAveragePrecision(window_range="middle", **kwargs),
            "MR-mAP-Long": MRAveragePrecision(window_range="long", **kwargs),
            "MR-mAP-Full": MRAveragePrecision(window_range="full", **kwargs),
            "MR-R1-Short": MRRecallAt1(window_range="short", **kwargs),
            "MR-R1-Middle": MRRecallAt1(window_range="middle", **kwargs),
            "MR-R1-Long": MRRecallAt1(window_range="long", **kwargs),
            "MR-R1-Full": MRRecallAt1(window_range="full", **kwargs),
        },
    )


def create_title(center: float, left: float, right: float, series_metrics_key: str, avg_key: str, split: str) -> str:
    """
    Create a formatted title string based on provided metrics and keys.

    Args:
        center (float): The center value to be rounded and included in the title.
        left (float): The left value to be rounded and included in the title.
        right (float): The right value to be rounded and included in the title.
        series_metrics_key (str): The key describing the series metrics.
        avg_key (str): The key representing the average metric.
        split (str): The split identifier.

    Returns:
        str: A formatted title string.
    """
    rounded_center = np.round(center, 2)
    rounded_left = np.round(left, 2)
    rounded_right = np.round(right, 2)
    return f"{split} {series_metrics_key}: {avg_key}={rounded_center} ({rounded_left}, {rounded_right})"  # noqa: WPS221


def get_bootstrapped_metric(pair: Tuple[np.ndarray, np.ndarray, int]) -> MetricCollection:
    """
    Calculate bootstrapped metric.

    Args:
        pair (Tuple[List[Dict[str, np.ndarray], List[Dict[str, Any], int]]]): list of prediction, list of targets, seed

    Returns:
        MetricCollection: bootstrapped metric
    """
    # need to pass the seed so that you don't get the same random subsamples
    postprocessed_results, meta, seed = pair
    n_items = len(meta)
    np.random.seed(seed)
    # generating a random subsample with a return
    bootstrapped_idxes = np.random.choice(n_items, size=n_items)
    bootstrapped_preds = postprocessed_results[bootstrapped_idxes]
    bootstrapped_meta = meta[bootstrapped_idxes]
    # calculating metrics on a bootstrapped subsample
    bootstrapped_metrics = get_metrics()
    bootstrapped_metrics(bootstrapped_preds, bootstrapped_meta)
    return bootstrapped_metrics


def compute_percentiles_confidence_interval(  # noqa: WPS118
    bootstraped_metric: List[float],
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Calculate the confidence interval based on percentiles.

    Note: this interval may not be symmetrical

    Args:
        bootstraped_metric (List[float]): list of bootstraped metric
        alpha (float): significance level. Defaults to 0.05.

    Returns:
        Tuple[float, float]: left and right border of the interval
    """
    bootstraped_metric = np.sort(bootstraped_metric)  # type: ignore
    n_bootstrap = len(bootstraped_metric)
    left = bootstraped_metric[int(alpha / 2 * n_bootstrap)]
    right = bootstraped_metric[int((1 - alpha / 2) * n_bootstrap)]
    return left, right


def compute_point(
    metrics: Dict[str, float],
    bootstrapped_metrics: List[Dict[str, float]],
    metric_key: str,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """
    Calculate the confidence interval for a single point.

    Args:
        metrics (Dict[str, float]): dictionary test metric
        bootstrapped_metrics (List[Dict[str, float]]): list of dictionaries bootstrapped metrics
        metric_key (str): the key of the sub metric for which the interval should be given
        alpha (float): significance level. Defaults to 0.05.

    Returns:
        Tuple[float, float, float]: center(test) and left, right of confidence interval
    """
    bootstraped_values: List[float] = [bootstrapped_metric[metric_key] for bootstrapped_metric in bootstrapped_metrics]
    left, right = compute_percentiles_confidence_interval(bootstraped_values, alpha=alpha)
    center = metrics[metric_key]
    return center, left, right


def _get_metrics_keys(metrics_keys: List[str]) -> Tuple[List[float], str]:
    """
    Get the keys for the metric.

    Args:
        metrics_keys (List[str]): all metric keys

    Returns:
        Tuple[List[float], str]: list of float keys(0.5, 0.55, ...) and average key("Avg", "mIoU")
    """
    not_float_keys = [key for key in metrics_keys if key.isalpha()]
    assert len(not_float_keys) == 1
    avg_key = not_float_keys[0]
    per_keys = sorted([float(key) for key in metrics_keys if key != avg_key])
    return per_keys, avg_key


def compute_confidence_intervals(
    test_metrics: Dict[str, float],
    bootstrapped_metrics: List[Dict[str, float]],
    alpha: float = 0.05,
) -> Tuple[List[float], List[float], List[float], float, float, float]:
    """Compute cofidence intervals.

    Args:
        test_metrics (Dict[str, float]): test metrics
        bootstrapped_metrics (List[Dict[str, float]]): bootstrapped metrics
        alpha (float): significance level. Defaults to 0.05.

    Returns:
        Tuple[List[float], List[float], List[float], float, float, float]: _description_
    """
    per_keys, avg_key = _get_metrics_keys(list(test_metrics.keys()))
    left_values = []
    right_values = []
    center_values = []
    for key in per_keys:
        center, left, right = compute_point(test_metrics, bootstrapped_metrics, str(key), alpha=alpha)
        left_values.append(left)
        right_values.append(right)
        center_values.append(center)
    avg_center, avg_left, avg_right = compute_point(test_metrics, bootstrapped_metrics, avg_key, alpha=alpha)
    return center_values, left_values, right_values, avg_center, avg_left, avg_right  # noqa: WPS227


class BootstrapMetricCallback(Callback):
    """Callback for calculating confidence intervals of various metrics."""

    def __init__(self, dirpath: str, alpha: float = 0.05, bootstraps_n: int = 1000) -> None:
        """Initialize BootstrapMetricCallback.

        Args:
            dirpath (str): directory to save plots
            alpha (float): significance level. Defaults to 0.05.
            bootstraps_n (int): number of bootstrapped subsamples. Defaults to 1000.
        """
        super().__init__()
        self.alpha = alpha
        self.bootstraps_n = bootstraps_n
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
            os.mkdir(os.path.join(dirpath, "train"))
            os.mkdir(os.path.join(dirpath, "valid"))
            os.mkdir(os.path.join(dirpath, "test"))
        self.dirpath = dirpath
        self.logger: Optional[ClearMLLogger] = None

    def _upload_image(self, image_path: str, name: str, title: str = "") -> None:
        if self.logger is not None:
            self.logger.report_media(title=title, series=name, local_path=image_path)

    def _get_loader(self, trainer: Trainer, split: str) -> DataLoader:
        assert split in {"valid", "test", "train"}
        if split == "valid":
            return trainer.datamodule.val_dataloader()
        if split == "test":
            return trainer.datamodule.test_dataloader()
        return trainer.datamodule.train_dataloader()

    @torch.no_grad()
    def _compute_predictions(self, trainer: Trainer, split: str) -> Tuple[np.ndarray, np.ndarray]:
        postprocessor: PostProcessorDETR = trainer.model.postprocessor
        all_postprocessed_results = []
        all_meta = []
        loader: DataLoader = self._get_loader(trainer, split)
        model: MRDETR = trainer.model.model
        preparator: Preparator = trainer.model.preparator
        model.eval()
        for data in loader:
            raw_meta, batch = data
            batch, targets = move_inputs_to_device(batch, trainer.model.device, non_blocking=True)
            outputs = model(targets=targets, **batch)
            _, prepared_outputs, _ = preparator(raw_meta, batch, outputs)
            raw_predictions = postprocessor(prepared_outputs)
            # leave only the keys necessary for metrics
            batch_meta = [{"qid": meta["qid"], "relevant_windows": meta["relevant_windows"]} for meta in raw_meta]
            # in order to use a lot of processes (for bootstrap), we convert torch to numpy
            batch_predictions = [
                {"qid": pred["qid"], "pred_relevant_windows": pred["pred_relevant_windows"].cpu().numpy()}
                for pred in raw_predictions
            ]
            all_meta.extend(batch_meta)
            all_postprocessed_results.extend(batch_predictions)
        return np.array(all_postprocessed_results), np.array(all_meta)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Calculate confidence intervals for test, validation, and train.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer instance.
            pl_module (LightningModule): The current PyTorch Lightning Module.
        """
        # define the logger if it is available
        if self.logger is None:
            task = trainer.clearml_task if hasattr(trainer, "clearml_task") else None  # noqa: WPS421
            logger = task.get_logger() if task is not None else None
            self.logger = logger
        self._compute_loader_confidence_intervals(trainer, "test")
        self._compute_loader_confidence_intervals(trainer, "valid")
        self._compute_loader_confidence_intervals(trainer, "train")

    # pylint: disable=too-many-locals
    def _compute_loader_confidence_intervals(self, trainer: Trainer, split: str) -> None:  # noqa: WPS118,WPS210
        all_postprocessed_results, all_meta = self._compute_predictions(trainer, "test")
        # calculating test metrics (one point)
        central_metrics = get_metrics()
        central_metrics(all_postprocessed_results, all_meta)

        # Calculating bootstrapped metrics
        cpu_count = os.cpu_count() if os.cpu_count() is not None else 1
        cpu_count = np.max([cpu_count - 3, 1])
        cpu_to_use: int = np.min([cpu_count, self.bootstraps_n])  # type: ignore
        # create tasks for each iteration, seed is needed so that the randomness is different on different processes
        tasks = [(all_postprocessed_results, all_meta, seed) for seed in range(self.bootstraps_n)]
        bootstrapped_metrics = Parallel(n_jobs=cpu_to_use)(delayed(get_bootstrapped_metric)(task) for task in tasks)
        # A metric is a list of many metrics: MAP-Full, MAP-short, and so on. For each of these metrics, need to
        # calculate confidence intervals
        series_metrics_keys: List[str] = list(central_metrics.keys())
        for series_metrics_key in series_metrics_keys:
            # select a group of metrics for which confidence intervals will be calculated
            central_metrics_item = central_metrics[series_metrics_key].compute()
            bootstrapped_metrics_item = [bt_metric[series_metrics_key].compute() for bt_metric in bootstrapped_metrics]
            # We select the IOU keys (ex. 0.5, 0.6) and the average key(ex. mIOU)
            per_keys, avg_key = _get_metrics_keys(list(central_metrics_item.keys()))
            # build a confidence interval for the metric (average and for different IOU)
            (  # noqa: WPS236
                center_values,
                left_values,
                right_values,
                avg_center,
                avg_left,
                avg_right,
            ) = compute_confidence_intervals(  # noqa: WPS236
                central_metrics_item,
                bootstrapped_metrics_item,
                alpha=self.alpha,
            )
            title = create_title(
                avg_center,
                avg_left,
                avg_right,
                series_metrics_key,
                avg_key,
                split,
            )
            image_path = os.path.join(self.dirpath, split, f"{title}.png")
            plot_confidence_intervals(
                per_keys,
                center_values,
                left_values,
                right_values,
                title,
                show=False,
                save_path=image_path,
            )
            self._upload_image(image_path=image_path, name=f"{split}: ", title=title)
