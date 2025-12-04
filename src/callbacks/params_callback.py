import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from clearml.logger import Logger as ClearMLLogger
from pytorch_lightning import Callback, LightningModule, Trainer

from src.callbacks.utils import ignore_errors


class ModeParamsCallback(Callback):
    def __init__(self, dirpath: str, frequency: Optional[int] = 1) -> None:
        super().__init__()
        self.frequency = frequency
        self._epoch = 1
        self.storage: Dict[str, List[float]] = {}  # type: ignore
        self.dirpath = os.path.join(dirpath, "train")
        if not os.path.exists(self.dirpath):
            os.mkdir(self.dirpath)
        self.logger: Optional[ClearMLLogger] = None

    def _upload_image(self, image_path: str, name: str, title: str = "") -> None:
        if self.logger is not None:
            self.logger.report_media(title=title, series=name, local_path=image_path, iteration=self._epoch)

    def _update_storages(self, params) -> None:
        for param_name, param_value in params.items():
            if param_name not in self.storage:
                self.storage[param_name] = [param_value]
            else:
                self.storage[param_name].append(param_value)

    def _plot_history(self) -> None:
        title = f"Model Params Movement epochs {self._epoch}"
        save_path = os.path.join(self.dirpath, f"model_params_movement_{self._epoch}.png")
        plt.figure(figsize=(12, 6))
        for param_name, params_values in self.storage.items():
            plt.plot(range(1, len(params_values) + 1), params_values, label=param_name)
        plt.grid()
        plt.legend()
        plt.title(title)
        plt.xlabel("epoch")
        plt.savefig(save_path)
        plt.close()
        self._upload_image(image_path=save_path, name=f"model_params_{self._epoch}.png", title=title)

    @ignore_errors(default_value=None)
    def _get_local_saliency_head_a(self, model) -> Dict[str, float]:
        return {"local_saliency_head_a": model.local_saliency_head.a.item()}

    @ignore_errors(default_value=None)
    def _get_local_saliency_head_b(self, model) -> Dict[str, float]:
        return {"local_saliency_head_b": model.local_saliency_head.b.item()}

    @ignore_errors(default_value=None)
    def _get_local_saliency_head_temp(self, model) -> Dict[str, float]:
        return {"local_saliency_head_temp": model.local_saliency_head.temp.item()}

    @ignore_errors(default_value=None)
    def _get_saliency_amplifier_alpha(self, model) -> Dict[str, float]:
        return {"saliency_amplifier_alpha": model.saliency_amplifier.alpha.item()}

    @ignore_errors(default_value=None)
    def _get_mr2hd_temperature(self, model) -> Dict[str, float]:
        return {"mr2hd_temperature": model.mr2hd.temperature.item()}

    def _get_params(self, model) -> Dict[str, float]:
        params: Dict[str, float] = {}
        param = self._get_local_saliency_head_a(model)
        if param is not None:
            params.update(param)
        param = self._get_local_saliency_head_b(model)
        if param is not None:
            params.update(param)
        param = self._get_local_saliency_head_temp(model)
        if param is not None:
            params.update(param)
        param = self._get_saliency_amplifier_alpha(model)
        if param is not None:
            params.update(param)
        param = self._get_mr2hd_temperature(model)
        if param is not None:
            params.update(param)
        return params

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule, *args, **kwargs):
        """
        Handler for the end of a training epoch. Logs the computed metrics.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer instance.
            pl_module (LightningModule): The current PyTorch Lightning Module.
        """
        if self.logger is None:
            task = trainer.clearml_task if hasattr(trainer, "clearml_task") else None  # noqa: WPS421
            logger = task.get_logger() if task is not None else None
            self.logger = logger
        model = trainer.model.model
        params = self._get_params(model)
        self._update_storages(params)
        # plot new graph
        if self.frequency is not None and self._epoch % self.frequency == 0:
            self._plot_history()
        self._epoch += 1

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._plot_history()
