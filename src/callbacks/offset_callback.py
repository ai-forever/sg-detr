"""Custom offset callback."""

from typing import Any, Dict

import numpy as np
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.trainer.states import RunningStage

from src.model.model import MRDETR


# pylint: disable=unused-argument,protected-access
class OffsetCallback(Callback):  # noqa: WPS214
    """
    Callback to compute the difference between the reference points and the final predictedspans.

    The smaller these values are, the better the anchors are arranged and the model does not have to calculate a
    complex offset.

    Callback compute 3 statistics:
    - ref_center_diff: The mean difference between the anchor and the predicted fragments centers
    - ref_width_diff: The mean difference between the anchor and the predicted fragments width
    - ref_distance: The mean Euclidean distance between the anchor and the prediction

    Note: Only matched predictions are used, since only they are used for loss (there is no reasons in all of them,
        since if the probability of a span is small, then anything can be predicted as the center and width)
    """

    def __init__(self):
        """Initialize storage for keeping track of reference difference during different stages."""
        self.storage = {
            RunningStage.TRAINING: {"center": 0, "width": 0, "distance": 0, "n_spans": 0},
            RunningStage.VALIDATING: {"center": 0, "width": 0, "distance": 0, "n_spans": 0},
            RunningStage.TESTING: {"center": 0, "width": 0, "distance": 0, "n_spans": 0},
            RunningStage.SANITY_CHECKING: {"center": 0, "width": 0, "distance": 0, "n_spans": 0},
        }

    def _reset_storages(self, stage: RunningStage) -> None:
        """
        Reset currecnt storage.

        Args:
            stage (RunningStage): The current running stage of the training process.
        """
        self.storage[stage] = {"center": 0, "width": 0, "distance": 0, "n_spans": 0}  # noqa: WPS221

    def _get_ref_points(self, model: MRDETR, outputs: Dict[str, Any]) -> torch.Tensor:
        if "encoder_outputs" in outputs:
            return outputs["encoder_outputs"]["pred_spans"]
        ref_points = model.main_det_head.refpoint_embed.get_reference_points()  # type: ignore
        return torch.sigmoid(ref_points)

    def _on_epoch_end(self, stage: RunningStage) -> Dict[str, float]:
        """
        Calculate the mean difference between reference points and predicted spans at the end of an epoch.

        Args:
            stage (RunningStage): The current running stage of the training process.

        Returns:
            Dict[str, float]: _description_
        """
        metrics = {}
        storage = self.storage[stage]
        metrics.update(
            {
                "ref_center_diff": storage["center"] / storage["n_spans"],
                "ref_width_diff": storage["width"] / storage["n_spans"],
                "ref_distance": storage["distance"] / storage["n_spans"],
            },
        )
        self._reset_storages(stage)
        return metrics

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule, *args, **kwargs):
        """
        Log the computed metrics.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer instance.
            pl_module (LightningModule): The current PyTorch Lightning Module.
            args: other args
            kwargs: other kwargs
        """
        current_stage = trainer.state.stage
        metrics = self._on_epoch_end(current_stage)
        for metric_key, metric_value in metrics.items():
            pl_module.log(  # noqa
                f"train/{metric_key}",
                metric_value,
                on_epoch=True,
                on_step=False,
                prog_bar=True,
                logger=True,
            )

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule, *args, **kwargs):
        """
        Log the computed metrics.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer instance.
            pl_module (LightningModule): The current PyTorch Lightning Module.
            args: other args
            kwargs: other kwargs
        """
        current_stage = trainer.state.stage
        metrics = self._on_epoch_end(current_stage)
        for metric_key, metric_value in metrics.items():
            pl_module.log(  # noqa
                f"val/{metric_key}",
                metric_value,
                on_epoch=True,
                on_step=False,
                prog_bar=True,
                logger=True,
            )

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule, *args, **kwargs):
        """
        Log the computed metrics.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer instance.
            pl_module (LightningModule): The current PyTorch Lightning Module.
            args: other args
            kwargs: other kwargs
        """
        current_stage = trainer.state.stage
        metrics = self._on_epoch_end(current_stage)
        for metric_key, metric_value in metrics.items():
            pl_module.log(  # noqa
                f"val/{metric_key}",
                metric_value,
                on_epoch=True,
                on_step=False,
                prog_bar=True,
                logger=True,
            )

    def on_batch_end(self, trainer: Trainer, pl_module: LightningModule, *args, **kwargs):
        """
        Calculate the difference between reference points and predicted spans.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer instance.
            pl_module (LightningModule): The current PyTorch Lightning Module.
            args: other args
            kwargs: other kwargs
        """
        # get data from module
        outputs = pl_module._current_outputs  # noqa: WPS437
        matching = pl_module._matching  # noqa: WPS437

        # get current storage
        current_stage = trainer.state.stage
        current_storage = self.storage[current_stage]
        # get ref points
        ref_points = self._get_ref_points(trainer.model.model, outputs)
        if ref_points.ndim == 2:
            ref_points = ref_points[None].repeat(outputs["pred_spans"].size(0), 1, 1)
        np_ref_points = ref_points.detach().cpu().numpy()
        # compute matched predicted spans
        pred_spans = outputs["pred_spans"].detach().cpu().numpy()
        indices = matching["positive"]["indices"]

        # for each span compute diffs
        for in_batch_idx in range(pred_spans.shape[0]):
            pred_idxs = indices[in_batch_idx][0].detach().cpu().numpy()
            matched_ref_points = np_ref_points[in_batch_idx][pred_idxs]
            matched_pred_spans = pred_spans[in_batch_idx][pred_idxs]

            center_diff = np.abs(matched_pred_spans[:, 0] - matched_ref_points[:, 0])  # noqa: WPS221
            width_diff = np.abs(matched_pred_spans[:, 1] - matched_ref_points[:, 1])  # noqa: WPS221
            distance = np.sqrt(center_diff**2 + width_diff**2)

            # update storage
            current_storage["center"] += np.sum(center_diff)
            current_storage["width"] += np.sum(width_diff)
            current_storage["distance"] += np.sum(distance)
            current_storage["n_spans"] += len(pred_idxs)

    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, *args, **kwargs):  # noqa: D102
        self.on_batch_end(trainer, pl_module, *args, **kwargs)

    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, *args, **kwargs):  # noqa: D102
        self.on_batch_end(trainer, pl_module, *args, **kwargs)

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, *args, **kwargs):  # noqa: D102
        self.on_batch_end(trainer, pl_module, *args, **kwargs)
