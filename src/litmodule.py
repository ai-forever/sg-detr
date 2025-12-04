# pylint:disable=arguments-differ,unused-argument
"""MomentRetrievalRunner Module."""

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.optim.lr_scheduler import _LRScheduler  # noqa: WPS450

from src.dataset.collate import move_inputs_to_device
from src.losses.losses import SetCriterion
from src.losses.utils import fix_loss_name
from src.metrics.matching.metrics import MatchingMetric
from src.metrics.metrics_collection import (
    get_aux_metrics,
    get_charades_metrics,
    get_metrics,
    get_tvsum_metrics,
    get_youtube_metrics,
)
from src.model.model import MRDETR
from src.model.utils.params import get_params_by_name
from src.postprocessor.postprocessing import (
    OutputCombiner,
    PostProcessorDETR,
    Preparator,
)
from src.utils.rw_utils import save_jsonl
from src.utils.span_utils import span_cxw_to_xx

MetaTypes = List[Dict[str, Any]]
MAIN_METRICS_DICT: Tuple[str, ...] = (
    "HL-HIT@1-VeryGood",
    "HL-mAP-VeryGood",
    "MR-mAP-Full_0.5",
    "MR-mAP-Full_0.75",
    "MR-mAP-Full_Avg",
    "MR-mAP-Short_Avg",
    "MR-mAP-Middle_Avg",
    "MR-mAP-Long_Avg",
    "MR-R1-Full_0.3",
    "MR-R1-Full_0.5",
    "MR-R1-Full_0.7",
    "MR-R1-Full_mIoU",
)


def _fix_output(postprocessed_outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    They fix the output of post-processing. This is necessary during training because the model might produce zero
    objects in the early epochs, which causes issues in calculating metrics.

    Args:
        postprocessed_outputs (List[Dict[str, Any]]): postprocessed outputs

    Returns:
        List[Dict[str, Any]]: fixes postprocessed outputs
    """
    new_outputs: List[Dict[str, Any]] = []
    for outputs in postprocessed_outputs:
        if len(outputs["pred_relevant_windows"]) == 0:
            outputs["pred_relevant_windows"] = torch.tensor([[0.0, 0.0, 0.0]])
        new_outputs.append(outputs)
    return new_outputs


# pylint: disable=too-many-instance-attributes
class MomentRetrievalRunner(LightningModule):  # noqa: WPS214,WPS230,E302
    """The main LightningModule for moment retrieval tasks."""

    def __init__(
        self,
        model: MRDETR,
        optimizer: torch.optim.Optimizer,
        scheduler: _LRScheduler,
        postprocessor: PostProcessorDETR,
        preparator: Preparator,
        combiner: OutputCombiner,
        losses: SetCriterion,
        metrics_mode: str = "qvhighlights",
        checkpoint_path: Optional[str] = None,
        check_train_every_n_epoch: int = 5,
    ) -> None:
        """
        Initialize the MomentRetrievalRunner.

        Args:
            model (MRDETR): The model to train.
            optimizer (torch.optim.Optimizer): The optimizer to use.
            scheduler (_LRScheduler): The learning rate scheduler to use.
            preparator(Preparator): Convert predictions to format.
            postprocessor (PostProcessorDETR): The postprocessor to use.
            losses (SetCriterion): The loss function to use.
            checkpoint_path (Optional[str]): path to base checkpoint.
            check_train_every_n_epoch (int): compute train metrics every N epoch
        """
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.save_hyperparameters(ignore=["model", "losses", "postprocessor", "scheduler", "combiner"])
        self.check_train_every_n_epoch = check_train_every_n_epoch
        self.model = model
        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path)["state_dict"]
            state_dict = {key[6:]: value for key, value in state_dict.items()}
            self.model.load_state_dict(state_dict, strict=False)

        self.losses = losses
        self.postprocessor = postprocessor
        self.preparator = preparator
        self.combiner = combiner
        self.scheduler = scheduler
        self._init_metrics(metrics_mode)
        self.metrics_mode = metrics_mode

    def _init_metrics(self, metrics_mode: str) -> None:  # noqa: C901
        """
        Initialize metrics for validation and testing.

        Args:
            metrics_mode (str): mode of metrics, define the metric list as well as the main metric
        """
        assert metrics_mode in {"qvhighlights", "youtube", "tvsum", "charades", "tacos"}
        aux_metrics = get_aux_metrics()
        self.aux_train_metrics = aux_metrics.clone(prefix="train/")
        self.aux_valid_metrics = aux_metrics.clone(prefix="val/")
        self.aux_test_metrics = aux_metrics.clone(prefix="test/")

        comb_metrics = get_aux_metrics()
        self.comb_train_metrics = comb_metrics.clone(prefix="train/")
        self.comb_valid_metrics = comb_metrics.clone(prefix="val/")
        self.comb_test_metrics = comb_metrics.clone(prefix="test/")

        if metrics_mode == "qvhighlights":
            metrics = get_metrics()
        elif metrics_mode == "tvsum":
            metrics = get_tvsum_metrics()
        elif metrics_mode in {"charades", "tacos"}:
            metrics = get_charades_metrics()
        else:
            metrics = get_youtube_metrics()
        self.train_metrics = metrics.clone(prefix="train/")
        self.valid_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        self.train_matching_metrics = MatchingMetric()
        self.valid_matching_metrics = MatchingMetric()
        self.test_matching_metrics = MatchingMetric()

        self.best_metric = 0
        self.submission: List[Dict[str, Any]] = []

        if metrics_mode == "qvhighlights":
            self.main_metric = "val/MR-mAP-Full_Avg"
        elif metrics_mode == "tvsum":
            self.main_metric = "val/HL-mAP-top5"
        elif metrics_mode == "charades":
            self.main_metric = "val/MR-R1-Full_0.5"
        elif metrics_mode == "tacos":
            self.main_metric = "val/MR-R1-Full_0.3"
        else:
            self.main_metric = "val/HL-mAP-Binary"

    def configure_optimizers(self):
        """Configure the optimizer and learning rate scheduler.

        Returns:
            dict: A dictionary containing the optimizer and the learning rate scheduler.
        """
        # select 3 groups of parameters: anchors, local_sal_params and everything else
        local_sal_params = get_params_by_name(
            self.model,
            include_prefixes=["local_saliency_head"],
            exclude_prefixes=None,
        )

        reference_params = get_params_by_name(
            self.model,
            include_prefixes=["main_det_head.refpoint_embed"],
            exclude_prefixes=None,
        )
        other_params = get_params_by_name(
            self.model,
            include_prefixes=None,
            exclude_prefixes=["main_det_head.refpoint_embed", "local_saliency_head"],
        )

        lr = self.hparams.optimizer.keywords["lr"]  # type: ignore # noqa: WPS111
        if reference_params:
            optimizer = self.hparams.optimizer(  # type: ignore
                params=[
                    {"params": local_sal_params, "lr": lr, "name": "local_sal", "weight_decay": 1e-1},
                    {"params": reference_params, "lr": lr, "name": "reference"},
                    {"params": other_params, "lr": lr, "name": "other_layers"},
                ],
            )
        else:
            optimizer = self.hparams.optimizer(  # type: ignore
                params=[
                    {"params": local_sal_params, "lr": lr, "name": "local_sal", "weight_decay": 1e-1},
                    {"params": other_params, "lr": lr, "name": "other_layers"},
                ],
            )

        if self.scheduler is not None:  # type: ignore
            scheduler = self.scheduler(optimizer=optimizer)  # type: ignore

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    # pylint: disable=attribute-defined-outside-init
    def save_submission(  # noqa: WPS234
        self,
        submission_batch: Optional[List[Dict[str, Any]]],
        prefix: str,
        metric: Optional[float] = None,
    ) -> None:
        """Accumulate and save submission.

        Args:
            submission_batch (Optional[List[Dict[str, Any]]]): postprocessed predictions.
            prefix (str): Prefix indicating the phase.
            metric (Optional[float]): current model score.
        """
        if submission_batch is not None:
            self.submission.extend(submission_batch)
            return

        for result in self.submission:
            for key, value in result.items():
                if isinstance(value, torch.Tensor):
                    result[key] = value.tolist()
        submission_name = os.path.join(self.trainer.log_dir, f"submission_{prefix}_last.jsonl")  # type: ignore
        save_jsonl(self.submission, submission_name)
        if (metric is not None) and (metric > self.best_metric):
            save_jsonl(self.submission, submission_name.replace("_last", "_best"))
            self.best_metric = metric  # type: ignore
        self.submission = []

    def _get_ref_points(self) -> Optional[Tensor]:
        """
        Get reference points of the model.

        Returns:
            Optional[Tensor]: reference points
        """
        if self.model.main_det_head.use_rpn:
            return None
        ref_points = self.model.main_det_head.refpoint_embed.get_reference_points()  # type: ignore
        return span_cxw_to_xx(torch.sigmoid(ref_points))

    def _process_batch(  # noqa: WPS210,C901,WPS213
        self,
        data: Tuple[MetaTypes, Dict[str, Any]],
        prefix: str,
    ) -> Optional[Tensor]:
        """
        Process a batch of embeddings for either training, validation, or testing.

        Args:
            data (Tuple[MetaTypes, Dict[str, Any]]): A tuple containing seq and ground truth labels.
            prefix (str): Prefix indicating the phase.

        Returns:
            Optional[torch.Tensor]: Computed total loss for train step.
        """
        meta, batch = data
        batch, targets = move_inputs_to_device(batch, self.device, non_blocking=True)
        outputs = self.model(targets=targets, meta=meta, **batch)
        matching = self.losses.compute_matches(outputs, targets, self._get_ref_points())  # type: ignore
        # Add data to model in order to use in MatcherCallback
        self._current_meta = meta
        self._current_targets = targets
        self._current_outputs = outputs
        self._matching = matching
        if "train" in prefix:
            losses = self.losses(outputs, targets, meta, matching)
            total_loss = torch.Tensor([0]).to(self.device)
            weight_dict = self.losses.weight_dict
            for loss_name, loss_value in losses.items():
                self.log(
                    f"{prefix}/{loss_name}",
                    loss_value,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=self.model.batch_size,
                )
                total_loss = total_loss + loss_value * weight_dict.get(fix_loss_name(loss_name), 0)

            self.log(
                f"{prefix}/total_loss",
                total_loss,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
                batch_size=self.model.batch_size,
            )
            self.train_matching_metrics.update(matching)

            if self.current_epoch % self.check_train_every_n_epoch == 0 and self.current_epoch != 0:
                with torch.no_grad():
                    aux_outputs, detr_outputs = self.preparator(meta, batch, outputs)  # type: ignore
                    comb_outputs = self.combiner(meta, batch, outputs)
                    postprocessed_detr_outputs = self.postprocessor(detr_outputs)  # type: ignore
                    postprocessed_aux_outputs = self.postprocessor(aux_outputs, aux_head=True)  # type: ignore
                    postprocessed_comb_outputs = self.postprocessor(comb_outputs, aux_head=True)
                    # plug ################
                    postprocessed_comb_outputs = _fix_output(postprocessed_comb_outputs)
                    postprocessed_aux_outputs = _fix_output(postprocessed_aux_outputs)
                    postprocessed_detr_outputs = _fix_output(postprocessed_detr_outputs)
                    # plug ################

                    self.train_metrics(submissions=postprocessed_detr_outputs, targets=meta)
                    self.aux_train_metrics(submissions=postprocessed_aux_outputs, targets=meta)
                    self.comb_train_metrics(submissions=postprocessed_comb_outputs, targets=meta)
            return total_loss

        with torch.autocast(dtype=torch.float32, device_type="cuda"):  # type: ignore
            aux_outputs, detr_outputs = self.preparator(meta, batch, outputs)  # type: ignore
            comb_outputs = self.combiner(meta, batch, outputs)
            postprocessed_detr_outputs = self.postprocessor(detr_outputs)  # type: ignore
            postprocessed_aux_outputs = self.postprocessor(aux_outputs, aux_head=True)  # type: ignore
            postprocessed_comb_outputs = self.postprocessor(comb_outputs, aux_head=True)

            # plug ################
            postprocessed_comb_outputs = _fix_output(postprocessed_comb_outputs)
            postprocessed_aux_outputs = _fix_output(postprocessed_aux_outputs)
            postprocessed_detr_outputs = _fix_output(postprocessed_detr_outputs)
            # plug ################

            if "val" in prefix:
                self.valid_metrics(submissions=postprocessed_detr_outputs, targets=meta)
                self.aux_valid_metrics(submissions=postprocessed_aux_outputs, targets=meta)
                self.comb_valid_metrics(submissions=postprocessed_comb_outputs, targets=meta)
                self.valid_matching_metrics.update(matching)
            else:
                self.test_metrics(submissions=postprocessed_detr_outputs, targets=meta)
                self.aux_test_metrics(submissions=postprocessed_aux_outputs, targets=meta)
                self.comb_test_metrics(submissions=postprocessed_comb_outputs, targets=meta)
                self.test_matching_metrics.update(matching)
            self.save_submission(postprocessed_detr_outputs, prefix)
        return None

    def training_step(self, batch, batch_idx: int) -> Tensor:
        """
        Process a batch during training.

        Args:
            batch (tuple): A tuple containing seqs and labels.
            batch_idx (int): Index of the current batch.

        Returns:
            Tensor: Computed Loss

        """
        return self._process_batch(batch, "train")  # type: ignore

    def validation_step(self, batch, batch_idx: int) -> None:
        """
        Process a batch during validation.

        Args:
            batch (tuple): A tuple containing seqs and labels.
            batch_idx (int): Index of the current batch.
        """
        self._process_batch(batch, "val")

    def test_step(self, batch, batch_idx: int) -> None:
        """
        Process a batch during testing.

        Args:
            batch (tuple): A tuple containing seqs and labels.
            batch_idx (int): Index of the current batch.
        """
        self._process_batch(batch, "test")

    def on_validation_epoch_start(self) -> None:
        """Reset the validation metrics at the start of a validation epoch."""
        self.valid_metrics.reset()
        self.aux_valid_metrics.reset()
        self.comb_valid_metrics.reset()
        self.valid_matching_metrics.reset()

    def on_validation_epoch_end(self):  # noqa: C901,WPS231
        """Log the computed validation metrics at the end of a validation epoch."""
        detr_val_metrics = self.valid_metrics.compute()
        aux_val_metrics = self.aux_valid_metrics.compute()
        comb_val_metrics = self.comb_valid_metrics.compute()
        valid_matching_metrics = self.valid_matching_metrics.compute()

        # log detr metrics
        if self.metrics_mode in {"charades", "tacos"}:
            metric = float(comb_val_metrics[self.main_metric])
        else:
            metric = float(detr_val_metrics[self.main_metric])
        self.save_submission(submission_batch=None, prefix="val", metric=metric)
        for name, value in detr_val_metrics.items():  # noqa: WPS204
            if "HL" in name:
                self.log(name, value, on_epoch=True, sync_dist=True, prog_bar=True)
            elif name[4:] in MAIN_METRICS_DICT:
                self.log(name, value, on_epoch=True, sync_dist=True, prog_bar=True)
            else:
                self.log(name, value, on_epoch=True, sync_dist=True)

        # log aux metrics
        for name, value in aux_val_metrics.items():
            if name[4:] in MAIN_METRICS_DICT:
                self.log(f"{name}-AUX", value, on_epoch=True, sync_dist=True)

        # log comb metrics
        for name, value in comb_val_metrics.items():
            if name[4:] in MAIN_METRICS_DICT:
                self.log(f"{name}-COMB", value, on_epoch=True, sync_dist=True)

        # log matching
        for name, value in valid_matching_metrics.items():
            self.log(f"val/{name}", value, on_epoch=True, sync_dist=True)

    def on_test_epoch_start(self) -> None:
        """Reset the validation metrics at the start of a test epoch."""
        self.test_metrics.reset()
        self.aux_test_metrics.reset()
        self.comb_test_metrics.reset()
        self.test_matching_metrics.reset()

    def on_test_epoch_end(self) -> None:
        """Log the computed validation metrics at the end of a test epoch."""
        detr_test_metrics = self.test_metrics.compute()
        aux_test_metrics = self.aux_test_metrics.compute()
        comb_test_metrics = self.comb_test_metrics.compute()
        matching_test_metrics = self.test_matching_metrics.compute()
        # log detr metrics
        self.log_dict(detr_test_metrics, on_epoch=True, sync_dist=True)
        for name, value in matching_test_metrics.items():
            self.log(f"test/{name}", value, on_epoch=True, sync_dist=True)

        # log aux metrics
        for name, value in aux_test_metrics.items():
            if name[5:] in MAIN_METRICS_DICT:
                self.log(f"{name}-AUX", value, on_epoch=True, sync_dist=True)

        # log comb metrics
        for name, value in comb_test_metrics.items():
            if name[5:] in MAIN_METRICS_DICT:
                self.log(f"{name}-COMB", value, on_epoch=True, sync_dist=True)

    def on_train_epoch_start(self) -> None:
        """Reset matching on train epoch end."""
        self.train_matching_metrics.reset()
        self.train_metrics.reset()
        self.aux_train_metrics.reset()
        self.comb_train_metrics.reset()

    def on_train_epoch_end(self) -> None:  # noqa: C901
        """Log metrics on train epoch end."""
        train_matching_metrics = self.train_matching_metrics.compute()

        # matching metrics
        for name, value in train_matching_metrics.items():
            self.log(f"train/{name}", value, on_epoch=True, sync_dist=True)

        if self.current_epoch % self.check_train_every_n_epoch == 0 and self.current_epoch != 0:
            detr_train_metrics = self.train_metrics.compute()
            aux_train_metrics = self.aux_train_metrics.compute()
            comb_train_metrics = self.comb_train_metrics.compute()

            # log detr metrics
            for name, value in detr_train_metrics.items():
                if name[6:] in MAIN_METRICS_DICT:
                    self.log(f"{name}", value, on_epoch=True, sync_dist=True)

            # log aux metrics
            for name, value in aux_train_metrics.items():
                if name[6:] in MAIN_METRICS_DICT:
                    self.log(f"{name}-AUX", value, on_epoch=True, sync_dist=True)
            # log comb metrics
            for name, value in comb_train_metrics.items():
                if name[6:] in MAIN_METRICS_DICT:
                    self.log(f"{name}-COMB", value, on_epoch=True, sync_dist=True)
