"""Custom matcher callback."""

from typing import Any, Dict, List, Tuple

import numpy as np
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.trainer.states import RunningStage

LoaderDataType = Dict[str, Dict[Tuple[float, float], List[int]]]  # noqa: WPS221
EPS: float = 1e-5


# pylint: disable=too-many-locals
def _compute_batch_matches(
    meta: List[Dict[str, Any]],
    targets: Dict[str, Any],
    current_matching: Any,
    current_aux_matching: Any,
) -> LoaderDataType:
    """
    Compute the mismatches between the predicted and ground truth (GT) spans for each batch.

    Args:
        meta (List[Dict[str, Any]]): Metadata for each item in the batch, including video and query IDs.
        targets (Dict[str, Any]): Ground truth data, including span labels.
        current_matching (Any): current matched indices
        current_aux_matching (Any): current matched indices

    Returns:
        LoaderDataType: A dictionary with video keys and item dictionaries indicating matches.
    """
    # Exclude auxiliary outputs to prepare for matching
    indices = current_matching["indices"]

    # Perform matching for auxiliary decoder outputs
    auxiliary_indices = current_aux_matching["indices"]

    # Prepare batch data storage
    batch_data: Dict[str, Dict[Tuple[float, float], List[int]]] = {}  # noqa: WPS234
    batch_lenght = len(targets["span_labels"])
    for in_batch_idx in range(batch_lenght):
        item_dict: Dict[Tuple[float, float], List[int]] = {}
        # Calculate a key unique to the query and video pair
        video_key = meta[in_batch_idx]["vid"] + str(meta[in_batch_idx]["qid"])
        pred_idxs = indices[in_batch_idx][0].detach().cpu().numpy()
        gt_idxs = indices[in_batch_idx][1].detach().cpu().numpy()

        # Collect matching information for both main and auxiliary predictions
        for pred_idx, gt_idx in zip(pred_idxs, gt_idxs):
            gt_span_array = targets["span_labels"][in_batch_idx]["spans"][gt_idx]
            gt_span_array = gt_span_array.detach().cpu().numpy()
            # In order to use the span as a key in the hash table, round it up and convert it to Tuple
            gt_span_as_key = tuple(gt_span_array.round(3))
            item_dict[gt_span_as_key] = [pred_idx]  # type: ignore

            for auxiliary_indice in auxiliary_indices:
                aux_pred_idxs = auxiliary_indice[in_batch_idx][0].detach().cpu().numpy()
                aux_gt_idxs = auxiliary_indice[in_batch_idx][1].detach().cpu().numpy()

                idxes = np.where(aux_gt_idxs == gt_idx)[0]
                # the target on the main output and in the additional ones must match
                assert len(idxes) < 2, f"there are {len(idxes)} indexes"  # noqa: WPS237

                # if we have #gt_spans > num_reference points it could leed to mismatch in aux_gt_idxs
                if not idxes:
                    continue

                idx = idxes[0]
                # adding reference indexes from additional outputs
                item_dict[gt_span_as_key].append(aux_pred_idxs[idx])  # type: ignore
        batch_data[video_key] = item_dict
    return batch_data


def _compute_decoder_mismatches(loader_data: LoaderDataType) -> float:
    """
    Compute the ratio of mismatches in the decoded spans for a loader dataset.

    Args:
        loader_data (LoaderDataType): A dictionary with video keys and item dictionaries indicating matches.

    Returns:
        float: The ratio of mismatches in the decoded spans.

    Note: The model makes predictions after each decoder layer. Therefore, for each gt span, a different reference can
        be found on different layers. If one reference is used for each layer, assume that it is a match,
        otherwise it is not a match
    """
    matched_n = 0
    missmatch_n = 0
    for _, video_data in loader_data.items():
        for _, reference_idxs in video_data.items():
            # If all outputs match to one reference, then we consider that the match
            if np.all(np.array(reference_idxs) == reference_idxs[0]):
                matched_n += 1
            else:
                missmatch_n += 1
    return missmatch_n / (missmatch_n + matched_n + EPS)


def _compute_loader_mismatches(  # noqa: WPS231
    currect_loader_storage: LoaderDataType,
    prev_loader_storage: LoaderDataType,
) -> float:  # noqa: WPS231
    """
    Compute the ratio of mismatches between the current and previous epoch.

    Args:
        currect_loader_storage (LoaderDataType): A dictionary (current epch) with video indicating matches
        prev_loader_storage (LoaderDataType): A dictionary (previous epoch) with video indicating matches

    Returns:
        float: the ratio of mismatches between current and previous epoch.

    Note: A single true span can be predicted using a specific reference. In the next epoch, it can be predicted by
        another reference. If the reference has changed, then consider that this is mismatch, otherwise it is a match.
        The fewer the mismatches, the more stable the model is.
    """
    matched_n = 0
    missmatch_n = 0
    for currect_video_id, currect_video_data in currect_loader_storage.items():
        # The data may differ from epoch to epoch, for example, if "cut off" the last batch
        if currect_video_id not in prev_loader_storage:
            continue
        prev_video_data = prev_loader_storage[currect_video_id]
        for span, currect_referecnce_idxes in currect_video_data.items():
            # There can be a lot of spans for one video. Therefore, the Dataset can cut off part of the spans.
            # Therefore, the sets may not match completely
            if span not in prev_video_data:
                continue
            prev_referecnce_idxes = prev_video_data[span]
            # Compare the outputs from the last (final) are compared with each other, so the index 0 is used
            if currect_referecnce_idxes[0] == prev_referecnce_idxes[0]:
                matched_n += 1
            else:
                missmatch_n += 1
    return missmatch_n / (missmatch_n + matched_n + EPS)


# pylint: disable=unused-argument,protected-access
class MatcherCallback(Callback):  # noqa: WPS214
    """
    Callback to compute and log the mismatch statistics of model predictions during all phases.

    Callback compute 2 statistics:
    - epoch_mismatches: a single true span can be predicted using a specific reference. In the next epoch, it can be
    predicted by another reference. If the reference has changed, then consider that this is mismatch, otherwise it
    is a match. The fewer the mismatches, the more stable the model is.
    - decoder_mismatches: The model makes predictions after each decoder layer. Therefore, for each gt span, a
    different reference can be found on different layers. If one reference is used for each layer, assume that it
    is a match, otherwise it is not a match
    """

    def __init__(self):
        """Initialize storage for keeping track of mismatches during different stages."""
        self.storage = {
            RunningStage.TRAINING: {"current": None, "prev": None},
            RunningStage.VALIDATING: {"current": None, "prev": None},
            RunningStage.TESTING: {"current": None, "prev": None},
            RunningStage.SANITY_CHECKING: {"current": None, "prev": None},
        }

    def _swap_storages(self, stage: RunningStage) -> None:
        """
        Swap the storage dictionaries, moving the current data to previous and initializing a new current storage.

        Args:
            stage (RunningStage): The current running stage of the training process.
        """
        self.storage[stage]["prev"] = self.storage[stage]["current"]  # noqa
        self.storage[stage]["current"] = None

    def on_epoch_end(self, stage: RunningStage) -> Dict[str, float]:
        """
        Compute mismatches at the end of an epoch.

        Args:
            stage (RunningStage): The current running stage of the training process.

        Returns:
            Dict[str, float]: _description_
        """
        metrics = {}
        assert self.storage[stage]["current"] is not None  # noqa: WPS204
        decoder_mismatches = _compute_decoder_mismatches(self.storage[stage]["current"])  # type: ignore
        metrics["decoder_mismatches"] = decoder_mismatches
        # Compute mismatches between epochs if previous data is available
        if self.storage[stage]["prev"] is not None:
            current_storage = self.storage[stage]["current"]
            prev_storage = self.storage[stage]["prev"]
            epoch_mismatches = _compute_loader_mismatches(current_storage, prev_storage)  # type: ignore
            metrics["epoch_mismatches"] = epoch_mismatches
        self._swap_storages(stage)
        return metrics

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule, *args, **kwargs):
        """
        Log the computed metrics after epoch ends.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer instance.
            pl_module (LightningModule): The current PyTorch Lightning Module.
            args: other args
            kwargs: other kwargs
        """
        current_stage = trainer.state.stage
        metrics = self.on_epoch_end(current_stage)
        for metric_key, metric_value in metrics.items():
            pl_module.log(
                f"train/{metric_key}",
                metric_value,
                on_epoch=True,
                on_step=False,
                prog_bar=True,
                logger=True,
            )

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule, *args, **kwargs):
        """
        Log the computed metrics after validation.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer instance.
            pl_module (LightningModule): The current PyTorch Lightning Module.
            args: other args
            kwargs: other kwargs
        """
        current_stage = trainer.state.stage
        metrics = self.on_epoch_end(current_stage)
        for metric_key, metric_value in metrics.items():
            pl_module.log(  # noqa: WPS221
                f"val/{metric_key}",
                metric_value,
                on_epoch=True,
                on_step=False,
                prog_bar=True,
                logger=True,
            )

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule, *args, **kwargs):
        """
        Log the computed metrics after test.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer instance.
            pl_module (LightningModule): The current PyTorch Lightning Module.
            args: other args
            kwargs: other kwargs
        """
        current_stage = trainer.state.stage
        metrics = self.on_epoch_end(current_stage)
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
        Compute matches for each batch and updates the storage accordingly.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer instance.
            pl_module (LightningModule): The current PyTorch Lightning Module.
            args: other args
            kwargs: other kwargs
        """
        # get data from module
        meta = pl_module._current_meta  # noqa: WPS437
        targets = pl_module._current_targets  # noqa: WPS437
        matching = pl_module._matching  # noqa: WPS437
        # get current storage
        current_stage = trainer.state.stage
        current_storage = self.storage[current_stage]
        current_matching = matching["positive"]
        current_aux_matching = matching["positive_aux"]
        # update the storage
        current_meta, current_targets = meta, targets
        batch_data = _compute_batch_matches(
            current_meta,
            current_targets,
            current_matching,
            current_aux_matching,
        )
        if current_storage["current"] is None:
            current_storage["current"] = batch_data  # type: ignore
        else:
            current_storage["current"].update(batch_data)  # type: ignore

    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, *args, **kwargs):  # noqa: D102
        self.on_batch_end(trainer, pl_module, *args, **kwargs)

    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, *args, **kwargs):  # noqa: D102
        self.on_batch_end(trainer, pl_module, *args, **kwargs)

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, *args, **kwargs):  # noqa: D102
        self.on_batch_end(trainer, pl_module, *args, **kwargs)
