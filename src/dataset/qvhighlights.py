"""QVHighlights dataset."""

import random
from os.path import join
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from torch import Tensor
from torch.utils.data import Dataset

from src.dataset.utils import (
    add_padding,
    get_qv_hard_indices,
    get_qv_pos_neg_indices,
    initialize_qv_score_array,
)
from src.utils.rw_utils import load_jsonl
from src.utils.span_utils import span_xx_to_cxw
from src.utils.tensor_utils import l2_normalize_np_array, l2_normalize_tensor

MAX_QUERY_LENGTH: int = 150
MAX_VIDEO_LENGTH: int = 1000000
CENTERNESS_THRESHOLD: float = 0.3


# pylint: disable=too-many-instance-attributes
class QVHighlights(Dataset):  # noqa: WPS230,WPS214
    """Dataset for QVHighlights data."""

    # pylint: disable=too-many-arguments
    def __init__(  # noqa: WPS211
        self,
        data_path: str,
        video_feat_dir: str,
        query_feat_dir: str,
        audio_feat_dir: Optional[str] = None,
        max_query_length: int = 32,
        max_video_length: int = 75,
        data_ratio: float = 1.0,
        normalize_video: bool = True,
        normalize_query: bool = True,
        use_tef: bool = True,
        clip_len: int = 2,
        max_windows: int = 10,
    ):
        """Initialize the dataset.

        Args:
            data_path (str): Path to the data.
            video_feat_dir (str): Path to the video features.
            query_feat_dir (str): Path to the query features.
            audio_feat_dir (Optional[str]): Path to the audio features.
            max_query_length (int): Max length of the query tensor. Defaults to 32.
            max_video_length (int): Max length of the video tensor. Defaults to 75.
            data_ratio (float): Portion of the data to use. Defaults to 1.0.
            normalize_video (bool): Whether to norm the video emb or not. Defaults to True.
            normalize_query (bool): Whether to norm the query emb or not. Defaults to True.
            use_tef (bool): Whether to use time positional features or not. Defaults to True.
            clip_len (int): Length of the clip in secs. Defaults to 2.
            max_windows (int): Maximum number of windows to use as labels. Defaults to 10.
        """
        self.data_path = data_path
        self.data_ratio = data_ratio
        self.audio_feat_dir = audio_feat_dir
        self.video_feat_dir = video_feat_dir
        self.query_feat_dir = query_feat_dir
        self.max_query_length = max_query_length if max_query_length > 0 else MAX_QUERY_LENGTH
        self.max_video_length = max_video_length if max_video_length > 0 else MAX_VIDEO_LENGTH

        self.normalize_query = normalize_query
        self.normalize_video = normalize_video
        self.clip_len = clip_len
        self.use_tef = use_tef
        self.max_windows = max_windows
        self.shrink_rates = [1, 2, 4]

        # data
        self.data = self.load_data()

    def load_data(self) -> List[Dict[str, Any]]:
        """Load data from data_path, and filter by data_ratio.

        Returns:
            List[Dict[str, Any]]: List of data.
        """
        datalist = load_jsonl(self.data_path)
        if self.data_ratio != 1:
            n_examples = int(len(datalist) * self.data_ratio)
            datalist = datalist[:n_examples]
            data_portion = self.data_ratio * 100
            logger.info(f"Using {data_portion}% of the data: {n_examples} examples")
        return datalist

    def get_query_feat_by_qid(self, qid: int) -> torch.Tensor:
        """Get query features by qid.

        Args:
            qid (int): index of the query

        Returns:
            torch.Tensor: Query features of shape: (Lq, D)
        """
        query_feat_path = join(self.query_feat_dir, f"{qid}.npz")
        query_features = np.load(query_feat_path)["features"].astype(np.float32)
        if query_features.ndim == 3:
            query_features = query_features[0]
        query_features = query_features[: self.max_query_length]
        if self.normalize_query:
            query_features = l2_normalize_np_array(query_features)
        return torch.from_numpy(query_features)

    @staticmethod
    def add_tef_features(video_feat: Tensor) -> Tensor:  # noqa: WPS602
        """Time based features.

        Args:
            video_feat (Tensor): video features

        Returns:
            Tensor: concated features
        """
        length = len(video_feat)
        tef_st: Tensor = torch.arange(0, length, 1.0) / length  # type: ignore
        tef_ed: Tensor = tef_st + 1.0 / length  # type: ignore
        tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
        return torch.cat([video_feat, tef], dim=1)  # (Lv, Dv + 2)

    def get_video_feat_by_vid(self, vid: int) -> torch.Tensor:
        """Get video features by vid.

        Args:
            vid (int): index of the video.

        Returns:
            torch.Tensor: Video features of shape: (Lv, D)
        """
        feature_path = join(self.video_feat_dir, f"{vid}.pt")
        v_feat = torch.load(feature_path)
        v_feat = v_feat[: self.max_video_length].type(torch.float32)
        if self.normalize_video:
            v_feat = l2_normalize_tensor(v_feat)
        if self.use_tef:
            return self.add_tef_features(v_feat)
        return v_feat

    def get_audio_feat_by_vid(self, vid: int) -> Optional[torch.Tensor]:
        """Get audio features by vid.

        Args:
            vid (int): index of the video.

        Returns:
            torch.Tensor: Video features of shape: (Lv, D)
        """
        if self.audio_feat_dir is None:
            return None
        feature_path = join(self.audio_feat_dir, f"{vid}.pt")
        a_feat = torch.load(feature_path)
        a_feat = a_feat[: self.max_video_length].type(torch.float32)
        if self.normalize_video:
            a_feat = l2_normalize_tensor(a_feat)
        if self.use_tef:
            return self.add_tef_features(a_feat)
        return a_feat

    def add_fpn_padding(self, video_emb: Tensor, audio_emb: Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]:
        """Add FPN stride padding.

        Args:
            video_emb (Tensor): video features
            audio_emb (Optional[Tensor]): audio features.

        Returns:
            Tuple[Tensor, Optional[Tensor]]: updated features.
        """
        if audio_emb is not None:
            a_dim = audio_emb.size(0)
            v_dim = video_emb.size(0)
            min_dim = min(a_dim, v_dim)
            audio_emb = audio_emb[:min_dim]
            video_emb = video_emb[:min_dim]

        max_shrink_rate = self.shrink_rates[-1]
        audio_emb = add_padding(audio_emb, max_shrink_rate)
        video_emb = add_padding(video_emb, max_shrink_rate)  # type: ignore
        return video_emb, audio_emb

    def get_span_labels(self, windows: List[List[int]], duration: int) -> torch.Tensor:
        """Get span labels.

        Args:
            windows (List[List[int]]): GT spans in seconds. E.g. [[26, 36]] (inclusive)
            duration (int): duration of the video in seconds.

        Returns:
            torch.Tensor: Span labels of shape: (#windows, 2)
        """
        if len(windows) == 0:  # noqa: WPS507
            return torch.empty((0, 2))
        if len(windows) > self.max_windows:
            random.shuffle(windows)
            windows = windows[: self.max_windows]
        normed_windows: Tensor = torch.Tensor(windows) / duration  # type: ignore
        return span_xx_to_cxw(normed_windows)  # normalized windows in cxw

    def get_saliency_labels_all(
        self,
        relevant_clip_ids: List[int],
        gt_scores: List[List[int]],
        number_of_clips: int,
        max_n: int = 1,
        add_easy_negative: bool = True,
    ) -> Tuple[List[int], List[int], np.ndarray]:
        """
        Determine saliency labels for video clips by aggregating annotation scores and selecting pos and neg samples.

        This function sums the scores from three annotations for each clip.
        It then selects clips with the highest and lowest scores as positive and negative samples, respectively.
        Additionally, it can sample easy negatives from clips outside the set of relevant clips.

        Args:
            relevant_clip_ids (List[int]): A list of IDs for relevant clips.
            gt_scores (List[List[int]]): A list of lists containing scores from three annotations for each clip.
            number_of_clips (int): The total number of clips.
            max_n (int): The number of clips to use as positive and negative. Defaults to 1.
            add_easy_negative (bool): If True, samples easy negatives from clips not in `relevant_clip_ids`.

        Returns:
            Tuple[List[int], List[int], np.ndarray]: A tuple containing three elements:
                - A list of positive clip indices.
                - A list of negative clip indices.
                - An ndarray of aggregated scores, where the score of each clip is at the index corresponding to its ID.

        Note:
            The function uses random sampling which can lead to different results on different executions.
        """
        agg_scores = np.sum(np.array(gt_scores), axis=1)
        sorted_indices = np.argsort(agg_scores)

        score_array = initialize_qv_score_array(relevant_clip_ids, number_of_clips, agg_scores)

        # hard_pos_indices - the highest relevant idx
        # hard_neg_indices - the lowest relevant idx
        hard_pos_indices, hard_neg_indices = get_qv_hard_indices(
            relevant_clip_ids,
            sorted_indices,
            max_n,
            number_of_clips,
        )
        # pos_clip_indices - [hard_pos_indices + random relevand idx]
        # neg_clip_indices - [hard_neg_indices + random irrelevand idx]
        pos_clip_indices, neg_clip_indices = get_qv_pos_neg_indices(
            relevant_clip_ids,
            number_of_clips,
            hard_pos_indices,
            hard_neg_indices,
            add_easy_negative,
            max_n,
        )

        return pos_clip_indices, neg_clip_indices, score_array

    def __len__(self) -> int:
        """Get len of the dataset.

        Returns:
            int: Datset length.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get item by index.

        Args:
            index (int): Index of the item.

        Returns:
            Dict[str, Any]: Dict of meta and model inputs.
        """
        meta = self.data[index]
        model_inputs: Dict[str, Any] = {}
        model_inputs["query_feat"] = self.get_query_feat_by_qid(meta["qid"])
        video_emb = self.get_video_feat_by_vid(meta["vid"])
        audio_emb = self.get_audio_feat_by_vid(meta["vid"])
        video_emb, audio_emb = self.add_fpn_padding(video_emb, audio_emb)
        number_of_clips = len(video_emb)
        meta["duration"] = number_of_clips * self.clip_len
        model_inputs["video_feat"] = video_emb
        if audio_emb is not None:
            model_inputs["audio_feat"] = audio_emb

        if "relevant_windows" in meta:
            # moment retrieval
            model_inputs["span_labels"] = self.get_span_labels(
                meta.get("relevant_windows"),  # type: ignore
                meta.get("duration"),  # type: ignore
            )
            # highlight detection
            saliency_pos, saliency_neg, saliency_all = self.get_saliency_labels_all(
                meta["relevant_clip_ids"],
                meta["saliency_scores"],
                number_of_clips,
            )
            # due to self.add_fpn_padding
            padding_size = video_emb.shape[0] - len(saliency_all)
            if padding_size > 0:
                saliency_all = np.concatenate([saliency_all, np.zeros(padding_size)])
            model_inputs["saliency_pos_labels"] = saliency_pos
            model_inputs["saliency_neg_labels"] = saliency_neg
            model_inputs["saliency_all_labels"] = saliency_all
            model_inputs["relevant_clip_ids"] = meta["relevant_clip_ids"]

        model_inputs["vid"] = meta["vid"]
        model_inputs["qid"] = meta["qid"]
        return {"meta": meta, "model_inputs": model_inputs}
