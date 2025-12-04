import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from src.dataset.qvhighlights import QVHighlights
from src.dataset.utils import get_irrelevant_windows
from src.utils.rw_utils import load_jsonl

TVSUM_SPLITS = {
    "BK": {"train": ["WxtbjNsCQ8A", "EE-bNr36nyA", "oDXZc0tZe04", "uGu_10sucQo"], "val": ["Se3oxnaPsz0"]},
    "BT": {"train": ["eQu1rNs0an0", "qqR6AEXwxoQ", "EYqVtI9YWJA", "iVt07TCkFM0"], "val": ["JgHubY5Vw3Y"]},
    "DS": {"train": ["kLxoNp-UchI", "NyBmCxDoHJU", "jcoYJXDG9sw", "-esJrBWj2d8"], "val": ["E11zDS9XGzg"]},
    "FM": {"train": ["_xMr-HKMfVA", "byxOvuiIJV0", "VuWGsYPqAX8", "xmEERLqJ2kU"], "val": ["JKpqYvAdIsw"]},
    "GA": {"train": ["xxdtq8mxegs", "i3wAGJaaktw", "0tmA_C6XwfM", "3eYKfiOEJNs"], "val": ["Bhxk-O1Y7Ho"]},
    "MS": {"train": ["Hl-__g2gn_A", "WG0MBPpPC6I", "LRw_obCPUt0", "37rzWOQsNIw"], "val": ["Yi4Ij2NM7U4"]},
    "PK": {"train": ["GsAD1KT1xo8", "XkqCExn6_Us", "b626MiF1ew4", "PJrm840pAUI"], "val": ["cjibtmSLxQ4"]},
    "PR": {"train": ["RBCABdttQmI", "z_6gVvQb2d0", "4wU_LUjG5Ic", "91IHQYk1IQM"], "val": ["fWutDQy1nnY"]},
    "VT": {"train": ["gzDbaEs1Rlg", "XzYM3PfTM4w", "98MoyGZKHXc", "AwmHb44_ouw"], "val": ["J0nA4VgnoCo"]},
    "VU": {"train": ["akI8YFjEmUw", "HT5vyqe0Xaw", "vdmoEJ5YbrQ", "xwqBXPGE9pQ"], "val": ["sTEELN-vY30"]},
}


# pylint: disable=too-many-instance-attributes
class TVSum(QVHighlights):  # noqa: WPS230,WPS214
    """Dataset for TVSum data."""
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        domain: str,
        data_path: str,
        video_feat_dir: str,
        query_feat_dir: str,
        audio_feat_dir: Optional[str] = None,
        max_query_length: int = 32,
        max_video_length: int = 75,
        data_ratio: float = 1,
        normalize_video: bool = True,
        normalize_query: bool = True,
        use_tef: bool = True,
        clip_len: int = 2,
        max_windows: int = 10,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            domain (str): TVSum Domain
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
        self.domain = domain
        super().__init__(
            data_path,
            video_feat_dir,
            query_feat_dir,
            audio_feat_dir,
            max_query_length,
            max_video_length,
            data_ratio,
            normalize_video,
            normalize_query,
            use_tef,
            clip_len,
            max_windows,
        )

    def load_data(self) -> List[Dict[str, Any]]:
        """Load data from data_path, and filter by data_ratio.

        Returns:
            List[Dict[str, Any]]: List of data.
        """
        datalist = load_jsonl(self.data_path)
        domain_data = []
        for sample in datalist:
            # take data only from the desired domain
            if (
                sample["qid"] in TVSUM_SPLITS[self.domain]["train"] or sample["qid"] in TVSUM_SPLITS[self.domain]["val"]
            ):  # noqa: WPS221
                domain_data.append(sample)

        datalist = domain_data
        if self.data_ratio != 1:
            n_examples = int(len(datalist) * self.data_ratio)
            datalist = datalist[:n_examples]
            data_portion = self.data_ratio * 100
            logger.info(f"Using {data_portion}% of the data: {n_examples} examples")
        return datalist

    def get_saliency_labels_all_tvsum(
        self,
        gt_scores: np.ndarray,
        ctx_l: int,
        max_n: int = 1,
        add_easy_negative: bool = False,
    ) -> Tuple[List[int], List[int], np.ndarray]:
        """
        Determine saliency labels for video clips by aggregating annotation scores and selecting pos and neg samples.

        This function sums the scores from three annotations for each clip.
        It then selects clips with the highest and lowest scores as positive and negative samples, respectively.
        Additionally, it can sample easy negatives from clips outside the set of relevant clips.

        Args:
            gt_scores (List[List[int]]): A list of lists containing scores from three annotations for each clip.
            ctx_l (int): The total number of clips.
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
        # TVSum score is in [1, 5], QVHighlights in [0, 4], so minus 1
        agg_scores = np.sum(gt_scores - np.ones_like(gt_scores), axis=-1)[:ctx_l]
        # Convert 20 ratings to 3 ratings 3 as in QWHighlits
        score_array = agg_scores / 80 * 12  # noqa: WPS221
        sort_indices = np.argsort(agg_scores)  # increasing

        hard_pos_clip_indices = [min(idx, ctx_l - 1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(idx, ctx_l - 1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []  # type: ignore
        easy_neg_clip_indices = []
        if add_easy_negative:
            # conside that relevant clips are those where the average score is more than 1
            rel_clip_ids = np.where(score_array > 1)[0]
            easy_neg_pool = list(set(range(ctx_l)) - set(rel_clip_ids))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)  # type: ignore
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices

        return pos_clip_indices, neg_clip_indices, score_array

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get item by index.

        Args:
            index (int): Index of the item.

        Returns:
            Dict[str, Any]: Dict of meta and model inputs.
        """
        meta = self.data[index]
        meta["irrelevant_windows"] = get_irrelevant_windows(meta["relevant_windows"], meta["duration"])
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
            saliency_pos, saliency_neg, saliency_all = self.get_saliency_labels_all_tvsum(
                meta["saliency_scores"],
                number_of_clips,
            )
            # remove last second if it nessesary
            model_inputs["video_feat"] = model_inputs["video_feat"][: len(saliency_all)]
            model_inputs["saliency_pos_labels"] = saliency_pos
            model_inputs["saliency_neg_labels"] = saliency_neg
            model_inputs["saliency_all_labels"] = saliency_all
            model_inputs["relevant_clip_ids"] = meta["relevant_clip_ids"]

        model_inputs["vid"] = meta["vid"]
        model_inputs["qid"] = meta["qid"]
        return {"meta": meta, "model_inputs": model_inputs}
