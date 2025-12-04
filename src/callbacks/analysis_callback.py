"""Analysis callback class."""

import os
from typing import Dict, List, Optional

import numpy as np
import torch
from clearml.logger import Logger as ClearMLLogger
from pytorch_lightning import Callback, LightningModule, Trainer

from src.analysis.analysis import compute_batch_predictions, compute_predictions
from src.analysis.data_models import VideoPrediction
from src.analysis.distributions import plot_gt_spans_distribution
from src.analysis.metrics import plot_average_metrics_by_gt_width
from src.analysis.mismatch import plot_matching_per_reference, plot_matching_pie_chart
from src.analysis.multiple import (
    plot_area_under_predictions,
    plot_multiple_covering_gt_distribution,
    plot_multiple_gt_distribution,
)
from src.analysis.reference import (
    plot_matched_spans_histogram,
    plot_mean_reference_shifts,
    plot_ref_points,
    plot_reference_shift_boxplot,
    plot_spans_grouped_by_reference,
)
from src.callbacks.utils import ignore_errors
from src.losses.matcher import HungarianMatcher
from src.model.model import MRDETR
from src.utils.span_utils import span_cxw_to_xx


# pylint: disable=unused-argument
class AnalysisCallback(Callback):  # noqa: WPS214
    """
    A PyTorch Lightning callback to compute and log the mismatch statistics of model predictions during all phases.

    Callback compute 2 statistics:
    - epoch_mismatches: a single true span can be predicted using a specific reference. In the next epoch, it can be
    predicted by another reference. If the reference has changed, then consider that this is mismatch, otherwise it
    is a match. The fewer the mismatches, the more stable the model is.
    - decoder_mismatches: The model makes predictions after each decoder layer. Therefore, for each gt span, a
    different reference can be found on different layers. If one reference is used for each layer, assume that it
    is a match, otherwise it is not a match
    """

    def __init__(self, dirpath: str, multiple_threshold: float = 0.5, frequency: int = 1) -> None:
        """Initialize AnalysisCallback.

        Args:
            dirpath (str): path to dir where to save results.
            multiple_threshold (float): use to identify multiple coverage. Defaults to 0.5.
            frequency (int): if epochs % frequency times == 0 trigger callback. Defaults to 1.
        """
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
            os.mkdir(os.path.join(dirpath, "train"))
            os.mkdir(os.path.join(dirpath, "valid"))
            os.mkdir(os.path.join(dirpath, "test"))
        self.dirpath = dirpath
        self.multiple_threshold = multiple_threshold
        self.logger: Optional[ClearMLLogger] = None
        self.frequency = frequency
        self._epoch = 1
        self.storages: Dict[str, List[VideoPrediction]] = {"train": [], "valid": [], "test": []}

    def _get_loader(self, trainer: Trainer, split: str):
        assert split in {"valid", "test", "train"}
        if split == "valid":
            return trainer.datamodule.val_dataloader()
        if split == "test":
            return trainer.datamodule.test_dataloader()
        return trainer.datamodule.train_dataloader()

    def _upload_image(self, image_path: str, name: str, title: str = "") -> None:
        if self.logger is not None:
            self.logger.report_media(title=title, series=name, local_path=image_path, iteration=self._epoch)

    def _get_reference_points(self, model: MRDETR) -> Optional[np.ndarray]:
        """
        Get reference points from model state dict.

        Args:
            model (MRDETR): MRDETR model

        Returns:
            np.ndarray: normalized reference points, shape [n_points, 2], center of the interval and it weights
        """
        ref_points = model.main_det_head.refpoint_embed.get_reference_points()  # type: ignore
        reference_points = torch.sigmoid(ref_points)
        return span_cxw_to_xx(reference_points).detach().cpu().numpy()

    @ignore_errors()
    def _plot_gt_spans_distribution(self, video_predictions: List[VideoPrediction], split: str) -> None:
        prefixs, predictions_list = [""], [video_predictions]  # noqa: WPS204
        for prefix, predictions in zip(prefixs, predictions_list):
            name = f"{prefix}_gt_spans_distribution"
            image_path = os.path.join(self.dirpath, split, f"{name}.png")  # noqa: WPS204
            title = f"{split}/{prefix} Distribution of GT Spans Lengths"
            plot_gt_spans_distribution(
                video_predictions=predictions,  # type: ignore
                title=title,
                save_path=image_path,
                show=False,
            )
            self._upload_image(image_path=image_path, name=f"{split}: {name}", title=title)

    @ignore_errors()
    def _plot_average_metrics_by_gt_width(  # noqa: WPS118
        self,
        video_predictions: List[VideoPrediction],
        split: str,
    ) -> None:
        prefixs, predictions_list = [""], [video_predictions]
        for prefix, predictions in zip(prefixs, predictions_list):  # noqa: WPS426
            name = f"{prefix}_average_metrics_by_gt_width"
            image_path = os.path.join(self.dirpath, split, f"{name}.png")
            title = f"{split}/{prefix} Basic metrics depending on the length of the gt spans"
            plot_average_metrics_by_gt_width(
                video_predictions=predictions,  # type: ignore
                title=title,
                save_path=image_path,
                show=False,
            )
            self._upload_image(image_path=image_path, name=f"{split}: {name}", title=title)

            # plot metrics for samples where one span
            name = f"{prefix}_average_metrics_by_gt_width_single_span"
            image_path = os.path.join(self.dirpath, split, f"{name}.png")
            title = f"{split}/{prefix} Basic metrics depending on the length of the gt spans(Single Span)"
            plot_average_metrics_by_gt_width(
                video_predictions=list(filter(lambda val: len(val.gt_spans) == 1, predictions)),  # type: ignore
                title=title,
                save_path=image_path,
                show=False,
            )
            self._upload_image(image_path=image_path, name=f"{split}: {name}", title=title)

            # plot metrics for samples where many spans
            name = f"{prefix}_average_metrics_by_gt_width_multi_span"
            image_path = os.path.join(self.dirpath, split, f"{name}.png")
            title = f"{split}/{prefix} Basic metrics depending on the length of the gt spans(Multiple Span)"
            plot_average_metrics_by_gt_width(
                video_predictions=list(filter(lambda val: len(val.gt_spans) > 1, predictions)),  # type: ignore
                title=title,
                save_path=image_path,
                show=False,
            )
            self._upload_image(image_path=image_path, name=f"{split}: {name}", title=title)

    @ignore_errors()
    def _plot_matching_per_reference(self, video_predictions: List[VideoPrediction], split: str) -> None:
        name = "matching_per_reference"
        image_path = os.path.join(self.dirpath, split, f"{name}.png")
        title = f"{split}/Match and Mismatch Counts for Each Reference"
        plot_matching_per_reference(video_predictions=video_predictions, title=title, save_path=image_path, show=False)
        self._upload_image(image_path=image_path, name=f"{split}: {name}", title=title)

    @ignore_errors()
    def _plot_matching_pie_chart(self, video_predictions: List[VideoPrediction], split: str) -> None:
        name = "matching_pie_chart"
        image_path = os.path.join(self.dirpath, split, f"{name}.png")
        title = f"{split}/Matches vs Mismatches"
        plot_matching_pie_chart(video_predictions=video_predictions, save_path=image_path, title=title, show=False)
        self._upload_image(image_path=image_path, name=f"{split}: {name}", title=title)

    @ignore_errors()
    def _plot_multiple_covering_gt_distribution(  # noqa: WPS118
        self,
        video_predictions: List[VideoPrediction],
        split: str,
    ) -> None:
        prefixs, predictions_list = [""], [video_predictions]
        for prefix, predictions in zip(prefixs, predictions_list):
            name = f"{prefix}_multiple_covering_gt_distribution"
            image_path = os.path.join(self.dirpath, split, f"{name}.png")
            title = f"{split}/{prefix} Multiple covering gt distribution"
            plot_multiple_covering_gt_distribution(
                video_predictions=predictions,  # type: ignore
                threshold=self.multiple_threshold,
                title=title,
                save_path=image_path,
                show=False,
            )
            self._upload_image(image_path=image_path, name=f"{split}: {name}", title=title)

    @ignore_errors()
    def _plot_multiple_gt_distribution(self, video_predictions: List[VideoPrediction], split: str) -> None:
        name = "multiple_gt_distribution"
        image_path = os.path.join(self.dirpath, split, f"{name}.png")
        title = f"{split}/Multiple gt distribution"
        plot_multiple_gt_distribution(
            video_predictions=video_predictions,
            threshold=self.multiple_threshold,
            title=title,
            save_path=image_path,
            show=False,
        )
        self._upload_image(image_path=image_path, name=f"{split}: {name}", title=title)

    @ignore_errors()
    def _plot_matched_spans_histogram(
        self,
        video_predictions: List[VideoPrediction],
        reference_points: np.ndarray,
        split: str,
    ) -> None:
        prefixs, predictions_list, ref_points_list = [""], [video_predictions], [reference_points]
        for prefix, predictions, ref_points in zip(prefixs, predictions_list, ref_points_list):
            name = f"{prefix}_matched_spans_histogram"
            image_path = os.path.join(self.dirpath, split, f"{name}.png")
            title = f"{split}/{prefix} Matched GT Spans by Category and Reference Point"
            plot_matched_spans_histogram(
                video_predictions=predictions,  # type: ignore
                reference_points=ref_points,
                title=title,
                save_path=image_path,
                show=False,
            )
            self._upload_image(image_path=image_path, name=f"{split}: {name}", title=title)

    @ignore_errors()
    def _plot_mean_reference_shifts(
        self,
        video_predictions: List[VideoPrediction],
        reference_points: np.ndarray,
        split: str,
    ) -> None:
        prefixs, predictions_list, ref_points_list = [""], [video_predictions], [reference_points]
        for prefix, predictions, ref_points in zip(prefixs, predictions_list, ref_points_list):
            name = f"{prefix}_mean_reference_shifts"
            image_path = os.path.join(self.dirpath, split, f"{name}.png")
            title = f"{split}/{prefix} Mean Shifts and Width Changes of Reference Points"
            plot_mean_reference_shifts(
                video_predictions=predictions,  # type: ignore
                reference_points=ref_points,
                title=title,
                save_path=image_path,
                show=False,
            )
            self._upload_image(image_path=image_path, name=f"{split}: {name}", title=title)

    @ignore_errors()
    def _plot_ref_points(self, reference_points: np.ndarray, split: str) -> None:
        name = "ref_points"
        image_path = os.path.join(self.dirpath, split, f"{name}.png")
        title = f"{split}/Visualization of ref_points"
        plot_ref_points(ref_points=reference_points, title=title, save_path=image_path, show=False)
        self._upload_image(image_path=image_path, name=f"{split}: {name}", title=title)

    @ignore_errors()
    def _plot_reference_shift_boxplot(
        self,
        video_predictions: List[VideoPrediction],
        reference_points: np.ndarray,
        split: str,
    ) -> None:
        prefixs, predictions_list, ref_points_list = [""], [video_predictions], [reference_points]
        for prefix, predictions, ref_points in zip(prefixs, predictions_list, ref_points_list):
            name = f"{prefix}_reference_shift_boxplot"
            image_path = os.path.join(self.dirpath, split, f"{name}.png")
            title = f"{split}/{prefix} Distribution of Shifts for Reference Points"
            plot_reference_shift_boxplot(
                video_predictions=predictions,  # type: ignore
                reference_points=ref_points,
                title=title,
                save_path=image_path,
                show=False,
            )
            self._upload_image(image_path=image_path, name=f"{split}: {name}", title=title)

    @ignore_errors()
    def _plot_spans_grouped_by_reference(  # noqa: WPS118
        self,
        video_predictions: List[VideoPrediction],
        reference_points: np.ndarray,
        split: str,
    ) -> None:
        prefixs, predictions_list, ref_points_list = [""], [video_predictions], [reference_points]
        for prefix, predictions, ref_points in zip(prefixs, predictions_list, ref_points_list):
            name = f"{prefix}_spans_grouped_by_reference"
            image_path = os.path.join(self.dirpath, split, f"{name}.png")
            title = f"{split}/{prefix} Spans grouped by Reference Points"
            plot_spans_grouped_by_reference(
                video_predictions=predictions,  # type: ignore
                reference_points=ref_points,
                save_path=image_path,
                title=title,
                show=False,
            )
            self._upload_image(image_path=image_path, name=f"{split}: {name}", title=title)

    @ignore_errors()
    def _plot_area_under_predictions(self, video_predictions: List[VideoPrediction], split: str) -> None:
        prefixs, predictions_list = [""], [video_predictions]
        for prefix, predictions in zip(prefixs, predictions_list):  # noqa: WPS426
            name = f"{prefix}_area_under_predictions"
            image_path = os.path.join(self.dirpath, split, f"{name}.png")
            title = f"{split}/{prefix} Area under matched predictioned spans"
            plot_area_under_predictions(
                video_predictions=predictions,  # type: ignore
                save_path=image_path,
                title=title,
                show=False,
            )
            self._upload_image(image_path=image_path, name=f"{split}: {name}", title=title)

            # plot metrics for samples where one span
            name = f"{prefix}_area_under_predictions_single_span"
            image_path = os.path.join(self.dirpath, split, f"{name}.png")
            title = f"{split}/{prefix} Area under matched predictioned spans(Single Span)"
            plot_area_under_predictions(
                video_predictions=list(filter(lambda val: len(val.gt_spans) == 1, predictions)),  # type: ignore
                save_path=image_path,
                title=title,
                show=False,
            )
            self._upload_image(image_path=image_path, name=f"{split}: {name}", title=title)

            # plot metrics for samples where many spans
            name = f"{prefix}_area_under_predictions_multi_span"
            image_path = os.path.join(self.dirpath, split, f"{name}.png")
            title = f"{split}/{prefix} Area under matched predictioned spans(Multiple Span)"
            plot_area_under_predictions(
                video_predictions=list(filter(lambda val: len(val.gt_spans) > 1, predictions)),  # type: ignore
                save_path=image_path,
                title=title,
                show=False,
            )
            self._upload_image(image_path=image_path, name=f"{split}: {name}", title=title)

    # pylint: disable=missing-function-docstring
    def on_epoch_end(self, trainer: Trainer, split: str) -> None:
        """Plot graphs on the end of the epoch.

        Args:
            trainer (Trainer): trainer instance
            split (str): data split
        """
        if self.logger is None:
            task = trainer.clearml_task if hasattr(trainer, "clearml_task") else None  # noqa: WPS421
            logger = task.get_logger() if task is not None else None
            self.logger = logger
        if self._epoch % self.frequency == 0:
            video_predictions = self.storages[split]
            reference_points = self._get_reference_points(model=trainer.model.model)
            self._plot_graphs(
                video_predictions=video_predictions,  # type: ignore
                reference_points=reference_points,  # type: ignore
                split=split,
            )
        self.storages[split] = []

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:  # noqa: D102
        self.on_epoch_end(trainer=trainer, split="train")
        self._epoch += 1

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:  # noqa: D102
        self.on_epoch_end(trainer=trainer, split="valid")

    # pylint: disable=W0212
    def on_batch_end(self, pl_module: LightningModule, split: str) -> None:
        """Compute batch predictions.

        Args:
            pl_module (LightningModule): instance of lightning module
            split (str): data split
        """
        if self._epoch % self.frequency == 0:
            meta = pl_module._current_meta  # noqa: WPS437
            targets = pl_module._current_targets  # noqa: WPS437
            outputs = pl_module._current_outputs  # noqa: WPS437
            matching = pl_module._matching  # noqa: WPS437
            batch_predictions = compute_batch_predictions(
                meta=meta,
                targets=targets,
                outputs=outputs,
                matching=matching,
            )
            self.storages[split].extend(batch_predictions)

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, *args, **kwargs) -> None:  # noqa: D102
        self.on_batch_end(pl_module=pl_module, split="train")

    def on_validation_batch_end(  # noqa: D102
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        *args,
        **kwargs,
    ) -> None:
        self.on_batch_end(pl_module=pl_module, split="valid")

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule, *args, **kwargs):
        """
        Log the computed metrics.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer instance.
            pl_module (LightningModule): The current PyTorch Lightning Module.
            args: other args
            kwargs: other kwargs
        """
        if self.logger is None:
            task = trainer.clearml_task if hasattr(trainer, "clearml_task") else None  # noqa: WPS421
            logger = task.get_logger() if task is not None else None
            self.logger = logger
        reference_points = self._get_reference_points(model=trainer.model.model)

        self.analyse_loader(
            trainer=trainer,
            reference_points=reference_points,  # type: ignore
            split="valid",
        )
        self.analyse_loader(
            trainer=trainer,
            reference_points=reference_points,  # type: ignore
            split="train",
        )
        self.analyse_loader(
            trainer=trainer,
            reference_points=reference_points,  # type: ignore
            split="test",
        )

    def analyse_loader(
        self,
        trainer: Trainer,
        reference_points: np.ndarray,
        split: str,
    ) -> None:
        """Compute predictions.

        Args:
            trainer (Trainer): trainer instance
            reference_points (np.ndarray): reference points
            split (str): train/val split
        """
        runner = trainer.model
        matcher: HungarianMatcher = trainer.model.losses.matcher
        loader = self._get_loader(trainer=trainer, split=split)
        ref_points = torch.Tensor(self._get_reference_points(runner.model))
        video_predictions = compute_predictions(
            loader=loader,
            matcher=matcher,
            model=runner.model,
            ref_points=ref_points,
        )
        self._plot_graphs(
            video_predictions=video_predictions,
            reference_points=reference_points,
            split=split,
        )

    def _plot_graphs(  # noqa: WPS213
        self,
        video_predictions: List[VideoPrediction],
        reference_points: np.ndarray,
        split: str,
    ) -> None:
        self._plot_gt_spans_distribution(video_predictions=video_predictions, split=split)
        self._plot_average_metrics_by_gt_width(video_predictions=video_predictions, split=split)

        self._plot_matching_per_reference(video_predictions=video_predictions, split=split)
        self._plot_matching_pie_chart(video_predictions=video_predictions, split=split)

        self._plot_multiple_covering_gt_distribution(video_predictions=video_predictions, split=split)
        self._plot_multiple_gt_distribution(video_predictions=video_predictions, split=split)

        self._plot_matched_spans_histogram(
            video_predictions=video_predictions,
            reference_points=reference_points,
            split=split,
        )
        self._plot_mean_reference_shifts(
            video_predictions=video_predictions,
            reference_points=reference_points,
            split=split,
        )

        self._plot_ref_points(reference_points=reference_points, split=split)
        self._plot_reference_shift_boxplot(
            video_predictions=video_predictions,
            reference_points=reference_points,
            split=split,
        )
        self._plot_spans_grouped_by_reference(
            video_predictions=video_predictions,
            reference_points=reference_points,
            split=split,
        )
        self._plot_area_under_predictions(
            video_predictions=video_predictions,
            split=split,
        )
