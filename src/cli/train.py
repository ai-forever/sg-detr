# pylint: disable=protected-access
"""Command-line script for training models using PyTorch Lightning."""

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import hydra
import pytorch_lightning as pylight
import torch
from clearml import Task
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import Logger

from src.cli.utils.instantiators import instantiate_callbacks, instantiate_loggers
from src.datamodule import MomentRetrievalDataModule
from src.litmodule import MomentRetrievalRunner
from src.utils.optuna import check_sampled_params


def convert_tensors_to_numbers(  # noqa: WPS234
    metrics: Dict[str, Union[float, torch.Tensor, Dict[str, torch.Tensor]]],  # noqa: WPS221
) -> Dict[str, float]:
    """
    Recursively convert all torch tensors in a dictionary to numbers (e.g., float).

    Args:
        metrics (dict): The dictionary containing metrics with possible torch tensors.

    Returns:
        dict: A new dictionary with tensors converted to numbers.
    """
    converted_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            assert value.numel() == 1, f"Tensor for key '{key}' contains more than one element, expected 1."
            converted_metrics[key] = value.item()
        elif isinstance(value, dict):
            converted_metrics[key] = convert_tensors_to_numbers(value)  # type: ignore
        else:
            converted_metrics[key] = value
    return converted_metrics


def get_experiment_dir(trainer: Trainer) -> str:
    """
    Get experiment directory from pytorch lightning Trainer

    Args:
        trainer (Trainer): pytorch lightning Trainer

    Returns:
        str: experiment directory
    """
    # Get the full path to the logs (includes the tensorboard/version_X folder)
    full_log_dir = trainer.logger.log_dir  # type: ignore

    # Get the directory to a higher level by deleting 'tensorboard/version_X'
    experiment_dir = os.path.dirname(os.path.dirname(full_log_dir))

    return experiment_dir


def setup_clearml_task(config: DictConfig) -> Task:
    """
    Initialize a logger task using given configuration.

    Args:
        config (DictConfig): The configuration object with project and experiment names.

    Returns:
        Task: An initialized Task object.
    """
    task = Task.init(
        project_name=config.project_name,
        task_name=config.task_name,
        reuse_last_task_id=False,
        tags=config.tags,
        auto_connect_frameworks={
            "tensorboard": {"report_hparams": True},
            "pytorch": "*.ckpt",
            "detect_repository": True,
            "jsonargparse": True,
        },
    )
    task.set_comment(config.description)
    return task


def prepare_trainer(config: DictConfig) -> Trainer:
    """Prepare the trainer object and other objects for training.

    Args:
        config (DictConfig): Configuration object containing training parameters.

    Returns:
        Trainer: Configured trainer object.
    """
    logger.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(config.get("callbacks"))

    logger.info("Instantiating loggers...")
    loggers: List[Logger] = instantiate_loggers(config.get("logger"))

    logger.info(f"Instantiating trainer <{config.trainer._target_}>")
    return hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=loggers)


def train(config: DictConfig) -> Tuple[Dict[str, Any], str]:
    """
    Train and test a model based on the provided configuration and trainer parameters.

    Args:
        config (DictConfig): Configuration object containing details for the data module and model runner.

    Returns:
        Dict[str, Any]: A dictionary containing the training and testing metrics.

    Raises:
        RuntimeError: If no seed is found in the configuration.

    Note:
        The function will set global seeds for reproducibility and will also run the test phase on
        the best checkpoint after the training phase is complete.
    """
    # Set reproducibility
    if config.get("seed", False):
        pylight.seed_everything(config.seed, workers=True)
    else:
        raise RuntimeError("No seed found! Unable to ensure reproducibility.")

    logger.info(f"Instantiating datamodule <{config.data._target_}>")
    datamodule: MomentRetrievalDataModule = hydra.utils.instantiate(config.data)

    logger.info(f"Instantiating model <{config.model.runner._target_}>")
    model: MomentRetrievalRunner = hydra.utils.instantiate(config.model.runner)

    trainer = prepare_trainer(config)

    if not config.is_local_run:
        load_dotenv(config.dotenv_path)
        task = setup_clearml_task(config)
        # in order to use clearml logger in callback define clearml task in trainer
        trainer.clearml_task = task  # type: ignore

    logger.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)
    train_metrics = trainer.callback_metrics

    if config.get("test"):
        logger.info("Starting testing!")
        trainer.test(datamodule=datamodule, ckpt_path="best")

    test_metrics = trainer.callback_metrics
    # merge train and test metrics
    return {**train_metrics, **test_metrics}, get_experiment_dir(trainer)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train.yaml")  # type: ignore
def main(config: DictConfig) -> Optional[float]:
    """
    Initialize training based on the provided configuration.

    Args:
        config (DictConfig): Configuration object containing details for training.

    Returns:
        Optional[float]: The value of the optimized metric if available, otherwise None.
    """
    if config.hyper_search:
        metric = check_sampled_params(config)
        if metric is not None:
            logger.info(f"Cached metric: {metric}")
            return metric

    metrics, output_dir = train(config=config)
    target_metric_name = config.get("optimized_metric", None)
    target_metric = metrics.get(target_metric_name, None)

    # Save metrics to JSON
    metrics = convert_tensors_to_numbers(metrics)
    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f)

    return float(target_metric) if target_metric is not None else None


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
