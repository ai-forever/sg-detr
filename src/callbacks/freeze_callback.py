"""Freeze refpoints callback."""

from typing import Optional

from pytorch_lightning import Callback, LightningModule, Trainer


class FreezeCallback(Callback):
    """Callback that allows you to freeze and defrost certain groups of weights during training."""

    def __init__(self, new_lr: float, freeze_epoch: Optional[int], freeze_param_group: str) -> None:
        """
        Initialize FreezeCallback.

        Args:
            new_lr (float): lr after defrosting the scales
            freeze_epoch (Optional[int]): Determines when the weights need to be unfrozen.
            freeze_param_group (str): Defines a group of weights that will be frozen for a while
        """
        self.new_lr = new_lr
        self.freeze_epoch = freeze_epoch
        self.freeze_param_group = freeze_param_group

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Freeze weights before starting training.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer instance.
            pl_module (LightningModule): The current PyTorch Lightning Module.
        """
        for param_group in trainer.optimizers[0].param_groups:
            if param_group.get("name") == self.freeze_param_group:
                param_group["lr"] = 0

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Defrost weights if necessary.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer instance.
            pl_module (LightningModule): The current PyTorch Lightning Module.
        """
        # Checking for an epoch
        if self.freeze_epoch is not None and trainer.current_epoch == self.freeze_epoch:
            # Changing the learning rate for a certain group of parameters
            for param_group in trainer.optimizers[0].param_groups:
                if param_group.get("name") == self.freeze_param_group:
                    param_group["lr"] = self.new_lr
