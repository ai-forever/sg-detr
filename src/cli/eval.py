# pylint: disable=protected-access
"""Command-line script to evaluate models using PyTorch Lightning."""

from typing import List

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback

from src.cli.utils.instantiators import instantiate_callbacks
from src.cli.utils.visualization import visualize_metrics
from src.datamodule import MomentRetrievalDataModule
from src.litmodule import MomentRetrievalRunner
from src.utils.rw_utils import save_json


def prepare_trainer(config: DictConfig) -> Trainer:
    """Prepare the trainer object and other objects for training.

    Args:
        config (DictConfig): Configuration object containing training parameters.

    Returns:
        Trainer: Configured trainer object.
    """
    logger.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(config.get("callbacks"))

    logger.info(f"Instantiating trainer <{config.trainer._target_}>")  # noqa: WPS437, WPS237
    return hydra.utils.instantiate(config.trainer, callbacks=callbacks)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="eval.yaml")
def main(config: DictConfig):
    """
    Evaluate the model based on the provided configurations.

    Args:
        config (DictConfig): Configuration object containing details for the data module and model runner.
    """
    logger.info(f"Instantiating datamodule <{config.data._target_}>")  # noqa: WPS437, WPS237
    datamodule: MomentRetrievalDataModule = hydra.utils.instantiate(config.data)
    logger.info(f"Annotation file: {datamodule.annotation_path_test}")

    logger.info(f"Instantiating model <{config.model.runner._target_}>")  # noqa: WPS437, WPS237
    model: MomentRetrievalRunner = hydra.utils.instantiate(config.model.runner)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(config.checkpoint, map_location=device)["state_dict"]
    model.load_state_dict(checkpoint)
    trainer = prepare_trainer(config)

    logger.info("Testing model...")
    trainer.test(model=model, datamodule=datamodule)
    metrics = {key: float(value) for key, value in trainer.callback_metrics.items()}
    save_json(metrics, filename=f"{trainer.log_dir}/metrics.json", save_pretty=True)
    if trainer.log_dir:
        visualize_metrics(metrics, trainer.log_dir)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
