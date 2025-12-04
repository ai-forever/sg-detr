"""Tensor utils."""

from typing import Any, List

import numpy as np
import torch


def pad_sequences_1d(sequences: List[Any], dtype, device, fixed_length):  # noqa: WPS114
    """Pad a single-nested list or a sequence of n-dimensional arrays into an (n+1)-dimensional array.

    This function allows padding of sequences where only the first dimension can have variable lengths.
    It is suitable for padding lists of tensors or numpy arrays to a uniform shape by adding zeros.

    Args:
        sequences (list of n-d tensor or list): A list of n-dimensional tensors or lists to be padded.
        dtype (np.dtype or torch.dtype): The data type of the output padded sequences.
        device (torch.device): The device on which the tensors will be allocated. Default is CPU.
        fixed_length (int): The fixed length to which all sequences will be padded.

    Returns:
        tuple:
            - padded_seqs (n+1-d tensor): The padded sequences. If `fixed_length` is not None, the shape will be
                `[len(sequences), fixed_length, ...]`. Otherwise, it depends on the length of the longest sequence.
            - mask (2d tensor): A tensor of the same shape as the first two dimensions of `padded_seqs`. Each element
                is 1 if the corresponding element in `padded_seqs` is valid, and 0 otherwise.
    """
    if isinstance(sequences[0], list):
        if "torch" in str(dtype):
            sequences = [torch.tensor(seq, dtype=dtype, device=device) for seq in sequences]
        else:
            sequences = [np.asarray(seq, dtype=dtype) for seq in sequences]

    extra_dims = sequences[0].shape[1:]  # the extra dims should be the same for all elements
    lengths = [len(seq) for seq in sequences]
    if fixed_length is not None:
        max_length = fixed_length
    else:
        max_length = max(lengths)
    if isinstance(sequences[0], torch.Tensor):
        assert "torch" in str(dtype), "dtype and input type does not match"
        padded_seqs = torch.zeros((len(sequences), max_length) + extra_dims, dtype=dtype, device=device)  # noqa: WPS221
        mask = torch.zeros((len(sequences), max_length), dtype=torch.float32, device=device)
    else:  # np
        assert "numpy" in str(dtype), "dtype and input type does not match"
        padded_seqs = np.zeros((len(sequences), max_length) + extra_dims, dtype=dtype)  # type: ignore
        mask = np.zeros((len(sequences), max_length), dtype=np.float32)  # type: ignore

    for idx, seq in enumerate(sequences):
        end = lengths[idx]
        padded_seqs[idx, :end] = seq
        mask[idx, :end] = 1
    return padded_seqs, mask  # , lengths


def pad_sequences_2d(sequences: List[Any], dtype=torch.long) -> tuple:  # noqa: WPS210,WPS114
    """Pad a double-nested list or a sequence of n-d torch tensor into a (n+1)-d tensor.

    Args:
        sequences (List[Any]): list(n-d tensor or list)
        dtype (torch.long): for word indices / torch.float (float32) for other cases

    Returns:
        tuple: padded_seqs, mask

    Examples:
        >>> test_data_list = [[[1, 3, 5], [3, 7, 4, 1]], [[98, 34, 11, 89, 90], [22], [34, 56]],]
        >>> pad_sequences_2d(test_data_list, dtype=torch.long)  # torch.Size([2, 3, 5])
        >>> test_data_3d = [torch.randn(2,2,4), torch.randn(4,3,4), torch.randn(1,5,4)]
        >>> pad_sequences_2d(test_data_3d, dtype=torch.float)  # torch.Size([2, 3, 5])
        >>> test_data_3d2 = [[torch.randn(2,4), ], [torch.randn(3,4), torch.randn(5,4)]]
        >>> pad_sequences_2d(test_data_3d2, dtype=torch.float)  # torch.Size([2, 3, 5])
    """
    bsz = len(sequences)
    para_lengths = [len(seq) for seq in sequences]
    max_para_len = max(para_lengths)
    sen_lengths = [[len(word_seq) for word_seq in seq] for seq in sequences]
    max_sen_len = max([max(sen) for sen in sen_lengths])

    if isinstance(sequences[0], torch.Tensor):
        extra_dims = sequences[0].shape[2:]
    elif isinstance(sequences[0][0], torch.Tensor):
        extra_dims = sequences[0][0].shape[1:]
    else:
        sequences = [[torch.Tensor(word_seq, dtype=dtype) for word_seq in seq] for seq in sequences]  # type: ignore
        extra_dims = ()  # type: ignore

    padded_seqs = torch.zeros((bsz, max_para_len, max_sen_len) + extra_dims, dtype=dtype)
    mask = torch.zeros(bsz, max_para_len, max_sen_len).float()

    for b_i in range(bsz):
        for sen_i, sen_l in enumerate(sen_lengths[b_i]):
            padded_seqs[b_i, sen_i, :sen_l] = sequences[b_i][sen_i]
            mask[b_i, sen_i, :sen_l] = 1
    return padded_seqs, mask  # , sen_lengths


def l2_normalize_np_array(array: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Normilize array along the last dimention.

    Args:
        array (np.ndarray): input array. Shape: (x, D).
        eps (float): Epsilon to avoid division by zero. Defaults to 1e-5.

    Returns:
        np.ndarray: Normalized array. Shape: (x, D).
    """
    return array / (np.linalg.norm(array, axis=-1, keepdims=True) + eps)


def l2_normalize_tensor(tensor: torch.Tensor, eps: float = 1e-5) -> np.ndarray:
    """Normilize tensor along the last dimention.

    Args:
        tensor (torch.Tensor): input tensor. Shape: (x, D).
        eps (float): Epsilon to avoid division by zero. Defaults to 1e-5.

    Returns:
        torch.Tensor: Normalized tensor. Shape: (x, D).
    """
    return tensor / (torch.norm(tensor, dim=-1, keepdim=True) + eps)
