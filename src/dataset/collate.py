"""Module for collate function for dataloader."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.utils.tensor_utils import pad_sequences_1d

MetaTyping = List[Dict[str, Any]]
DataTyping = Dict[str, Any]


def custom_collate(batch: List[Dict[str, Any]]) -> Tuple[MetaTyping, DataTyping]:  # noqa: WPS231
    """
    Collate data for dataloader.

    Args:
        batch (List[Dict[str, Any]]): list of uncollated data.

    Returns:
        Tuple[MetaTyping, DataTyping]: Meta data and batched data.
    """
    batch_meta = [sample["meta"] for sample in batch]
    model_inputs_keys = batch[0]["model_inputs"].keys()
    batched_data: Dict[str, Any] = {}

    for key in model_inputs_keys:
        if key == "span_labels":
            batched_data[key] = [{"spans": sample["model_inputs"][key]} for sample in batch]
            continue

        if key in {"saliency_pos_labels", "saliency_neg_labels"}:
            batched_data[key] = torch.LongTensor([sample["model_inputs"][key] for sample in batch])
            continue

        if key == "saliency_all_labels":
            pad_data, _ = pad_sequences_1d(
                [sample["model_inputs"][key] for sample in batch],
                device=torch.device("cpu"),
                dtype=np.float32,
                fixed_length=None,
            )
            batched_data[key] = torch.tensor(pad_data, dtype=torch.float32)
            continue

        if key in {"qid", "vid"}:
            batched_data[key] = [sample["model_inputs"][key] for sample in batch]
            continue

        batched_data[key] = pad_sequences_1d(
            [sample["model_inputs"][key] for sample in batch],
            device=torch.device("cpu"),
            dtype=torch.float32,
            fixed_length=None,
        )

    return batch_meta, batched_data


def move_inputs_to_device(  # noqa: WPS234
    batched_model_inputs: Dict[str, Any],
    device: torch.device,
    non_blocking: bool = False,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Move model inputs to device.

    Args:
        batched_model_inputs (Dict[str, Any]): batched model inputs.
        device (torch.device): device to move to.
        non_blocking (bool): whether to use non_blocking mode. Defaults to False.

    Returns:
        Tuple[Dict[str, Any], Optional[Dict[str, Any]]]: model_inputs, targets
    """
    model_inputs = {
        "src_txt": batched_model_inputs["query_feat"][0].to(device, non_blocking=non_blocking),
        "src_txt_mask": batched_model_inputs["query_feat"][1].to(device, non_blocking=non_blocking),
        "src_vid": batched_model_inputs["video_feat"][0].to(device, non_blocking=non_blocking),
        "src_vid_mask": batched_model_inputs["video_feat"][1].to(device, non_blocking=non_blocking),
        "vid": batched_model_inputs["vid"],
        "qid": batched_model_inputs["qid"],
    }
    if "audio_feat" in batched_model_inputs:
        model_inputs["src_aud"] = batched_model_inputs["audio_feat"][0].to(device, non_blocking=non_blocking)

    targets: Dict[str, Any] = {}
    if "span_labels" in batched_model_inputs:
        targets["span_labels"] = [
            {"spans": sample["spans"].to(device, non_blocking=non_blocking)}
            for sample in batched_model_inputs.get("span_labels")  # type: ignore
        ]
    if "saliency_pos_labels" in batched_model_inputs:
        for name in ("saliency_pos_labels", "saliency_neg_labels"):
            targets[name] = batched_model_inputs[name].to(device, non_blocking=non_blocking)

    if "saliency_all_labels" in batched_model_inputs:
        targets["saliency_all_labels"] = batched_model_inputs.get("saliency_all_labels")
        targets["saliency_all_labels"] = targets["saliency_all_labels"].to(device, non_blocking=non_blocking)
        targets["relevant_clips"] = targets["saliency_all_labels"].to(device, non_blocking=non_blocking)

    return model_inputs, targets if targets else None
