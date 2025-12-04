"""Multiple spans analysis."""

from typing import List, Optional, Tuple

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt

from src.analysis.data_models import VideoPrediction
from src.analysis.utils import (
    calculate_average_distance_between_spans,
    calculate_intersection_over_gt_size,
    compute_intersection_matrix,
    find_external_gaps,
    find_gaps,
)


# pylint: disable=too-many-locals
def plot_multiple_covering_gt_distribution(  # noqa: WPS210,WPS213
    video_predictions: List[VideoPrediction],
    threshold: float = 0.5,
    title: str = "Multiple covering gt distribution",
    figsize: Tuple[int, int] = (8, 5),
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Draw a bar chart based on how often predicted spans cover several gt spans.

    In addition, it also draws the average confidence of the spans in each group

    Args:
        video_predictions (List[VideoPrediction]): Predictions for each video.
        threshold (float): The threshold that determines how much the predicted span must cover the gt
        title (str): Plot title. Defaults to 'Multiple covering gt distribution'.
        figsize (Tuple[int, int]): Figure size as a tuple (width, height). Defaults to (8, 5).
        save_path (Optional[str]): The path to save the graph (if None, then the graph is not saved).
        show (bool): display a graph or not. Defaults to True.

    Note: for this purpose, only predictions are used in which there are several gt spans.
    """
    video_predictions_many = list(filter(lambda x: len(x.gt_spans) > 1, video_predictions))
    list_covered_gt_spans: List[int] = []
    list_predicted_span_probs: List[float] = []
    for vp in video_predictions_many:
        matched_pred_idxes, matched_gt_idxes = vp.matching
        for pred_idx in matched_pred_idxes:
            pred_span = vp.pred_spans[pred_idx]
            iou_list = np.array(
                [calculate_intersection_over_gt_size(pred_span, vp.gt_spans[gt_idx]) for gt_idx in matched_gt_idxes]
            )
            covered_gt_idxs = np.where(iou_list > threshold)[0]
            list_covered_gt_spans.append(len(covered_gt_idxs))
            list_predicted_span_probs.append(vp.pred_probs[pred_idx])
    covered_gt_spans = np.array(list_covered_gt_spans)
    predicted_span_probs = np.array(list_predicted_span_probs)

    unique, counts = np.unique(covered_gt_spans, return_counts=True)

    # Preparing data for average probabilities
    mean_probs = [np.mean(predicted_span_probs[covered_gt_spans == n]) for n in unique]

    fig, ax1 = plt.subplots(figsize=figsize)

    # Rendering the number of predictions covering a certain number of GT spans
    bars = ax1.bar(unique, counts, color="gray", label="Count")
    ax1.set_xlabel("Number of GT Spans Covered")
    ax1.set_ylabel("Count", color="gray")
    ax1.tick_params(axis="y", labelcolor="gray")
    ax1.set_title("Distribution of Predictions Covering Multiple GT Spans")
    ax1.set_yscale("log")  # Setting the logarithmic scale for the Y axis

    # Adding captions above the columns
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, height, f"{int(height)}", ha="center", va="bottom", color="black")

    # Add a second Y axis for average probabilities
    ax2 = ax1.twinx()
    ax2.plot(unique, mean_probs, color="blue", marker="o", label="Mean Probability")
    ax2.set_ylabel("Mean Probability", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    fig.tight_layout()
    plt.title(title)
    fig.legend()
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved as: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_multiple_gt_distribution(
    video_predictions: List[VideoPrediction],
    threshold: float = 0.5,
    title: str = "Multiple gt distribution",
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Visualizes distributions of various characteristics for predictions covering multiple GT (ground truth) spans.

    This function plots four histograms showing:
    1. The mean distance between covered GT spans.
    2. The distribution of the lengths of the GT spans.
    3. The distribution of the lengths of the predicted spans.
    4. The probability distribution for predicted spans.

    Args:
        video_predictions (List[VideoPrediction]): Predictions for each video.
        threshold (float): The threshold that determines how much the predicted span must cover the gt
                        span in order to consider that it covers it. Defaults to 0.5.
        title (str): Plot title. Defaults to 'Multiple gt distribution'.
        figsize (Tuple[int, int]): Figure size as a tuple (width, height). Defaults to (8, 5).
        save_path (Optional[str]): The path to save the graph (if None, then the graph is not saved).
        show (bool): display a graph or not. Defaults to True.
    """
    # Filter video predictions to only those covering more than one GT span
    video_predictions_many = list(filter(lambda x: len(x.gt_spans) > 1, video_predictions))

    # Lists to hold computed statistics for the histograms
    average_distance_between_spans: List[float] = []
    mean_covered_gt_widths: List[float] = []
    pred_span_widths: List[float] = []
    pred_span_probs: List[float] = []
    # Calculate statistics for each video prediction
    for vp in video_predictions_many:
        matched_pred_idxes, matched_gt_idxes = vp.matching
        for pred_idx in matched_pred_idxes:
            pred_span = vp.pred_spans[pred_idx]
            # Compute the intersection over GT size for each GT span
            iou_list = np.array(
                [calculate_intersection_over_gt_size(pred_span, vp.gt_spans[gt_idx]) for gt_idx in matched_gt_idxes]
            )
            # Filter GT indexes based on the threshold
            covered_gt_idxs = np.where(iou_list > threshold)[0]
            covered_gt_idxs = matched_gt_idxes[covered_gt_idxs]
            covered_gt_spans = vp.gt_spans[covered_gt_idxs]
            # Compute statistics if more than one GT span is covered
            if len(covered_gt_spans) > 1:
                average_distance_between_spans.append(calculate_average_distance_between_spans(covered_gt_spans))
                mean_covered_gt_widths.append(np.mean([span[1] - span[0] for span in covered_gt_spans]))
                pred_span_widths.append(pred_span[1] - pred_span[0])
                pred_span_probs.append(vp.pred_probs[pred_idx])

    # Create the figure with 2 columns and 2 rows
    fig, axs = plt.subplots(2, 2, figsize=figsize)

    # Plotting the histograms
    # Histogram 1: Mean distance between GT spans
    axs[0, 0].hist(average_distance_between_spans, color="steelblue", edgecolor="black", log=False, bins=20)
    axs[0, 0].grid(linestyle="--", linewidth=0.5)
    axs[0, 0].set_xlabel("Mean distance between gt spans")
    axs[0, 0].set_ylabel("Count (log)")
    axs[0, 0].set_title("Mean distance distribution between covered gt spans")

    # Histogram 2: Mean width of GT spans
    axs[0, 1].hist(mean_covered_gt_widths, edgecolor="black", log=False)
    axs[0, 1].grid(linestyle="--", linewidth=0.5)
    axs[0, 1].set_xlabel("Mean width gt spans")
    axs[0, 1].set_ylabel("Count")
    axs[0, 1].set_title("Distribution of the lengths of the gt spans")

    # Histogram 3: Mean width of predicted spans
    axs[1, 0].hist(pred_span_widths, color="steelblue", edgecolor="black", log=False)
    axs[1, 0].grid(linestyle="--", linewidth=0.5)
    axs[1, 0].set_xlabel("Mean distance between gt spans")
    axs[1, 0].set_ylabel("Count")
    axs[1, 0].set_title("The distribution of the lengths of the pred spans")

    # Histogram 4: Probability distribution for predicted spans
    axs[1, 1].hist(pred_span_probs, edgecolor="black", log=False)
    axs[1, 1].grid(linestyle="--", linewidth=0.5)
    axs[1, 1].set_xlabel("Predicted span probability")
    axs[1, 1].set_ylabel("Count")
    axs[1, 1].set_title("Probability distribution for predicted spans")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # type: ignore

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved as: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_area_under_predictions(
    video_predictions: List[VideoPrediction],
    title: str = "Area under matched predictioned spans",
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Draws a pie chart that shows which area is under the predicted matched spans.

    4 options are considered:
    - The area with the corresponding true segments
    - The area under the non-conforming true segment
    - The area between the true segments (when there are several of them
    - The area outside spans

    Args:
        video_predictions (List[VideoPrediction]): Predictions for each video.
        title (str): Plot title.. Defaults to "Area under matched predictioned spans".
        figsize (Tuple[int, int]): Figure size as a tuple (width, height). Defaults to (8, 5).
        save_path (Optional[str]): The path to save the graph (if None, then the graph is not saved).
        show (bool): display a graph or not. Defaults to True.
    """
    matched_gt_area = 0
    mismatch_gt_area = 0
    gap_area = 0
    external_area = 0

    for prediction in video_predictions:
        pred_idxs, gt_idxes = prediction.matching
        pred_spans = prediction.pred_spans[pred_idxs]
        gt_spans = prediction.gt_spans[gt_idxes]
        gap_spans = find_gaps(gt_spans)
        external_gaps = find_external_gaps(gt_spans)

        gt_intersection_matrix = compute_intersection_matrix(pred_spans, gt_spans)
        item_matched_gt_area = np.diag(gt_intersection_matrix).sum()
        item_mismatch_gt_area = gt_intersection_matrix.sum() - item_matched_gt_area
        item_gap_area = compute_intersection_matrix(pred_spans, gap_spans).sum()
        item_external_area = compute_intersection_matrix(pred_spans, external_gaps).sum()

        matched_gt_area += item_matched_gt_area
        mismatch_gt_area += item_mismatch_gt_area
        gap_area += item_gap_area
        external_area += item_external_area
    preliminary_sizes = np.array([matched_gt_area, mismatch_gt_area, gap_area, external_area])
    labels = np.array(["matched_gt_area", "mismatch_gt_area", "gap_area", "external_area"])
    colors = np.array(["skyblue", "orange", "lightgreen", "red"])  # Colors for each category
    # Some categories may not exist (for example, if only videos with one fragment are considered)
    sizes = preliminary_sizes[np.where(preliminary_sizes != 0)]
    labels = labels[np.where(preliminary_sizes != 0)]
    colors = colors[np.where(preliminary_sizes != 0)]

    figsize = (10, 10)
    # Creating the pie chart
    plt.figure(figsize=figsize)
    plt.pie(
        sizes,
        labels=labels,  # type: ignore
        colors=colors,  # type: ignore
        autopct=lambda p: f"{p * (sum(sizes)) / 100:.0f}\n({p:.1f})",
        startangle=90,
    )
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved as: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()
