"""Plot distributions."""

from typing import List, Optional, Tuple

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt

from src.analysis.data_models import VideoPrediction


# pylint: disable=W1203
def plot_gt_spans_distribution(  # noqa: WPS213
    video_predictions: List[VideoPrediction],
    title: str = "Distribution of GT Spans Lengths",
    figsize: Tuple[int, int] = (12, 5),
    bins_for_lengths: int = 20,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Display the distribution of data (the number of fragments in one video and the length of fragments).

    Args:
        video_predictions (List[VideoPrediction]): Predictions for each video.
        title: str: plot title
        figsize (Tuple[int, int]): Figure size as a tuple (width, height). Defaults to (12, 5).
        bins_for_lengths (int): Number of bins for span lebgth distribution. Defaults to 20.
        save_path (Optional[str]): The path to save the graph (if None, then the graph is not saved).
        show (bool): display a graph or not. Defaults to True.
    """
    plt.figure(figsize=figsize)

    # Graph of the number of GT spans in the video
    number_gt_spans = np.array([len(pred.gt_spans) for pred in video_predictions])
    min_gt_spans = number_gt_spans.min()
    max_gt_spans = number_gt_spans.max()
    xticks = np.arange(min_gt_spans, max_gt_spans + 1)
    plt.subplot(1, 2, 1)  # The first graph of the two
    bins = np.arange(min_gt_spans, max_gt_spans + 2) - 0.5
    plt.hist(number_gt_spans, bins=bins, edgecolor="black")  # type: ignore
    plt.xlabel("Number of GT spans")
    plt.ylabel("Frequency")
    plt.title("Number of GT spans per video")
    plt.xticks(xticks)

    # GT spans length distribution graph
    gt_span_lengths = np.concatenate(
        [vpreds.gt_spans[:, 1] - vpreds.gt_spans[:, 0] for vpreds in video_predictions],  # noqa: WPS221
    )
    plt.subplot(1, 2, 2)  # Второй график из двух
    plt.hist(gt_span_lengths, bins=bins_for_lengths, color="skyblue", edgecolor="black")
    plt.xlabel("Length of GT Spans")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved as: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()
