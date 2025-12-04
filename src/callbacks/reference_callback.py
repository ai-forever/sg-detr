"""History references callback."""

import os
from copy import deepcopy
from typing import Any, List, Optional

import numpy as np
import torch
from clearml.logger import Logger as ClearMLLogger
from pytorch_lightning import Callback, LightningModule, Trainer
from torch import Tensor

from src.analysis.reference import plot_reference_history
from src.model.model import MRDETR


class HistoryReferenceCallback(Callback):
    """History Reference Callback."""

    def __init__(self, dirpath: str, sensitivity: float = 0.05, frequency: Optional[int] = 1) -> None:
        """Initialize HistoryReferenceCallback.

        Args:
            dirpath (str): path to save data.
            sensitivity (float): plot sensitivity. Defaults to 0.05.
            frequency (Optional[int]): run frequency. Defaults to 1.
        """
        super().__init__()
        self.sensitivity = sensitivity
        self.frequency = frequency
        self._epoch = 1
        self.storage: Optional[List[Any]] = None  # type: ignore
        self.dirpath = os.path.join(dirpath, "train")
        if not os.path.exists(self.dirpath):
            os.mkdir(self.dirpath)
        self.logger: Optional[ClearMLLogger] = None

    def _get_ref_points(self, model: MRDETR) -> Tensor:
        if model.main_det_head.refpoint_embed is not None:
            ref_points = model.main_det_head.refpoint_embed.get_reference_points()  # type: ignore
        else:
            raise NotImplementedError("Dynamic queries not supported!")
        reference_points = torch.sigmoid(ref_points)
        return reference_points.detach().cpu().numpy()

    def _upload_image(self, image_path: str, name: str, title: str = "") -> None:
        if self.logger is not None:
            self.logger.report_media(title=title, series=name, local_path=image_path, iteration=self._epoch)

    def _update_storages(self, trainer: Trainer) -> None:
        ref_points = self._get_ref_points(trainer.model.model)
        if self.storage is None:
            num_ref_points = len(ref_points)
            self.storage = [[ref_points[idx]] for idx in range(num_ref_points)]  # type: ignore
        else:
            for idx, point in enumerate(ref_points):
                self.storage[idx].append(point)  # type: ignore

    def _plot_history(self) -> None:
        numpy_current_storage = deepcopy(self.storage)
        for idx, storage_value in enumerate(numpy_current_storage):  # type: ignore
            numpy_current_storage[idx] = np.array(storage_value)  # type: ignore
        title = f"References Movement across epochs {self._epoch}"
        save_path = os.path.join(self.dirpath, f"references_movement_{self._epoch}.png")  # noqa: WPS221
        plot_reference_history(
            ref_history=numpy_current_storage,  # type: ignore
            sensitivity=self.sensitivity,
            title=title,
            save_path=save_path,
            show=False,
        )
        self._upload_image(
            image_path=save_path,
            name=f"references_movement_{self._epoch}.png",  # noqa: WPS221
            title=title,
        )

    # pylint: disable=unused-argument
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule, *args, **kwargs):
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

        # update data
        self._update_storages(trainer=trainer)
        # plot new graph
        if self.frequency is not None and self._epoch % self.frequency == 0:
            self._plot_history()
        self._epoch += 1

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:  # noqa: D102
        self._plot_history()
