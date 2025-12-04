# flake8: noqa: C901
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from src.analysis.data_models import VideoPrediction
from src.analysis.utils import calculate_iou_1d

GT_SPAN_OFFEST = -0.05  # offset for gt spans down from the graph
REF_Y_DIFF = 0.03  # offset between reference spans


def plot_average_metrics_by_gt_width(
    video_predictions: List[VideoPrediction],
    num_bins: int = 25,
    title: str = "Basic metrics depending on the length of the gt spans",
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Draws graphs of the main metrics depending on the width of the fragments (there are the same number of objects in
    each width bin)

    Args:
        video_predictions (List[VideoPrediction]): Predictions for each video.
        num_bins (int, optional): Number of bins. Defaults to 25.
        title: str: Plot title. Defaults to "Basic metrics depending on the length of the gt spans".
        figsize (Tuple[int, int], optional): Figure size as a tuple (width, height). Defaults to (14, 6).
        save_path (Optional[str], optional): The path to save the graph (if None, then the graph is not saved).
        Defaults to None.
        show (bool, optional): display a graph or not. Defaults to True.
    """
    list_gt_widths: List[float] = []  # list of lengths of true spans
    list_true_ious: List[float] = []  # list iou
    list_pred_probs: List[float] = []  # list of predicted confidence
    list_iou_diffs: List[float] = []  # list of difference between true and predicted iou

    for vp in video_predictions:
        for pred_idx, gt_idx in zip(*vp.matching):
            gt_span = vp.gt_spans[gt_idx]
            pred_span = vp.pred_spans[pred_idx]
            gt_width = gt_span[1] - gt_span[0]
            iou = calculate_iou_1d(gt_span, pred_span)
            pred_iou = vp.pred_quality_scores[pred_idx]
            iou_diff = abs(iou - pred_iou)
            prob = vp.pred_probs[pred_idx]

            list_gt_widths.append(gt_width)
            list_true_ious.append(iou)
            list_pred_probs.append(prob)
            list_iou_diffs.append(iou_diff)
    gt_widths = np.array(list_gt_widths)
    true_ious = np.array(list_true_ious)
    pred_probs = np.array(list_pred_probs)
    iou_diffs = np.array(list_iou_diffs)
    # Determination of uniformly filled bins by quantiles of GT spans width
    bins = np.percentile(gt_widths, np.linspace(0, 100, num_bins + 1))
    bin_indices = np.digitize(gt_widths, bins)

    # Calculate the average values for each bin
    avg_width_per_bin = []
    avg_true_iou_per_bin = []
    avg_pred_prob_per_bin = []
    avg_iou_diffs_per_bin = []

    for i in range(1, len(bins)):
        in_bin = bin_indices == i
        if np.any(in_bin):
            avg_width_per_bin.append(np.mean(gt_widths[in_bin]))
            avg_true_iou_per_bin.append(np.mean(true_ious[in_bin]))
            avg_pred_prob_per_bin.append(np.mean(pred_probs[in_bin]))
            avg_iou_diffs_per_bin.append(np.mean(iou_diffs[in_bin]))

    plt.figure(figsize=figsize)
    plt.plot(avg_width_per_bin, avg_true_iou_per_bin, "o--", color="red", linewidth=2, markersize=8, label="IOU")
    plt.plot(
        avg_width_per_bin, avg_iou_diffs_per_bin, "o--", color="purple", linewidth=2, markersize=8, label="IOU diff"
    )
    plt.plot(avg_width_per_bin, avg_pred_prob_per_bin, "o--", color="blue", linewidth=2, markersize=8, label="Probs")
    plt.xlabel("Average Width of GT Spans")
    plt.xticks(np.linspace(0, 1, 11))
    plt.yticks(np.linspace(0, 1, 11))
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved as: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def draw_prediction(  # noqa: C901
    video_prediction: VideoPrediction,
    reference_points: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Displays basic information on the video: draws graphs of the salience score, draws predicted spans, matches them
    with gt spans and links them to reference points

    Args:
        video_prediction (VideoPrediction): Predictions for each video.
        reference_points (Optional[np.ndarray], optional): Reference spans(shape: [n_points, 2]), beginning and
        end(normalized from 0 d to 1). Defaults to None.
        figsize (Tuple[int, int], optional): Figure size as a tuple (width, height). Defaults to (12, 8).
        save_path (Optional[str], optional): The path to save the graph (if None, then the graph is not saved).
        Defaults to None.
        show (bool, optional): display a graph or not. Defaults to True.
    """
    normalized_video_length = np.linspace(0, 1, len(video_prediction.pred_saliency_scores))
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()
    # Plot saliency scores
    ax2.plot(
        normalized_video_length,
        video_prediction.pred_saliency_scores,
        color="green",
        label="Predicted Saliency Scores",
    )
    ax3 = ax1.twinx()  # axis for true saliency_scores
    ax3.spines["right"].set_position(("outward", 60))
    ax3.plot(normalized_video_length, video_prediction.saliency_scores, color="orange", label="Saliency Scores")
    # To synchronize the axes, 2 points are put on ax2 that correspond to the maximum and minimum values along ax1.
    # At the same time, keep the correspondence that the zeros coincide and that the maximum silence coincides with
    # the maximum probs
    max_saliency = video_prediction.pred_saliency_scores.max()
    max_probs = video_prediction.pred_probs.max()
    if reference_points is not None:
        n_reference_points = reference_points.shape[0]
        ref_y_values = np.array([max_probs + i * REF_Y_DIFF for i in range(1, n_reference_points + 1)])
        ax1_max_value = ref_y_values.max()
    else:
        ax1_max_value = max_probs

    y_min = max_saliency * GT_SPAN_OFFEST / max_probs
    y_max = max_saliency * ax1_max_value / max_probs
    ax2.scatter(x=0, y=y_min, alpha=0)  # alpha = 0 since these points cannot be visible on the graph
    ax2.scatter(x=0, y=y_max, alpha=0)  # alpha = 0 since these points cannot be visible on the graph
    # Similarly, synchronize the axes for true saliency. At the same time, correlate the maximum of the predicted
    # saliency to 1 along the axis of true saliency
    y_min = GT_SPAN_OFFEST / max_probs
    y_max = ax1_max_value / max_probs
    ax3.scatter(x=0, y=y_min, alpha=0)  # alpha = 0 since these points cannot be visible on the graph
    ax3.scatter(x=0, y=y_max, alpha=0)  # alpha = 0 since these points cannot be visible on the graph

    # Create a set of matched predicted span indices for coloring
    matched_pred_indices = set(video_prediction.matching[0])

    # Draw the ground truth spans just below the main graph
    for idx, (start, end) in enumerate(video_prediction.gt_spans):
        ax1.hlines(GT_SPAN_OFFEST, start, end, colors="red", label="GT Span(with IOU)", linewidth=2)

    if reference_points is not None:
        # Draw a reference_point with a slight upward shift along the Y axis for each subsequent point
        for idx, (start, end) in enumerate(reference_points):
            ax1.hlines(ref_y_values[idx], start, end, colors="purple", label="Reference Point", linewidth=2)

    # Plot predicted spans
    for idx, span in enumerate(video_prediction.pred_spans):
        score = video_prediction.pred_probs[idx]
        # Color matched spans differently from unmatched
        color = "blue" if idx in matched_pred_indices else "grey"
        label = "Matched Pred Span(with pred IOU)" if idx in matched_pred_indices else "Unmatched Pred Span"
        ax1.hlines(score, span[0], span[1], colors=color, label=label, linewidth=2)

    # Draw lines between matched GT and pred spans and annotate matched predictions with quality score
    for pred_idx, gt_idx in zip(*video_prediction.matching):
        gt_span = video_prediction.gt_spans[gt_idx]
        pred_span = video_prediction.pred_spans[pred_idx]
        iou_score = calculate_iou_1d(gt_span, pred_span)
        gt_center = np.mean(gt_span)
        pred_center = np.mean(pred_span)

        # Draw a matching between the true and predicted spans
        ax1.plot(
            [gt_center, pred_center],
            [GT_SPAN_OFFEST, video_prediction.pred_probs[pred_idx]],
            "k--",
            label="Match to GT",
        )
        # Annotate matched pred span with predicted iou score
        ax1.text(
            pred_center,
            video_prediction.pred_probs[pred_idx] + 0.015,
            f"{video_prediction.pred_quality_scores[pred_idx]:.2f}",
            ha="center",
        )
        # Annotate matched gt span with iou score
        ax1.text(gt_center, GT_SPAN_OFFEST - 0.025, f"{iou_score:.2f}", ha="center", color="red")
        if reference_points is not None:
            ref_point = reference_points[pred_idx]
            ref_center = np.mean(ref_point)
            # Draw a matching between the predicted span and corresponding reference point
            ax1.plot(
                [pred_center, ref_center],
                [video_prediction.pred_probs[pred_idx], ref_y_values[pred_idx]],
                "purple",
                linestyle="dotted",
                label="Match to Ref",
            )

    # Labels, title and legend
    # draw the scales only for the part for which they are defined
    ax1.set_yticks(np.linspace(0, max_probs, num=5))  # num -this is the number of labels on the axis
    ax2.set_yticks(np.linspace(0, max_saliency, num=5))  # num -this is the number of labels on the axis

    ax1.set_xlabel("Normalized Video Length")
    ax1.set_ylabel("Prediction Probs", color="blue")
    ax2.set_ylabel("Saliency Scores", color="green")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax2.tick_params(axis="y", labelcolor="green")

    # draw legend of the figure
    handles_1, labels_1 = ax1.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels_1 + labels_2, handles_1 + handles_2))  # Remove duplicates
    ax1.legend(by_label.values(), by_label.keys(), loc="upper left")
    ax1.grid(True)
    fig.tight_layout()  # For better spacing

    # save figure if nessesaty
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved as: {save_path}")
    if show:
        plt.show()
    plt.close()  # Closes the figure to free up memory


def plot_confidence_intervals(
    x_values: List[float],
    center_values: List[float],
    left_values: List[float],
    right_values: List[float],
    title: str,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plots the metric and confidence intervals for it

    Args:
        x_values (List[float]): X-axis values(IOU list for MAP or something else)
        center_values (List[float]): Metric values for each iou
        left_values (List[float]): list of left border of the interval
        right_values (List[float]): list of right border of the interval
        title (str, optional): plot title
        figsize (Tuple[int, int], optional): Figure size as a tuple (width, height). Defaults to (12, 5).
        save_path (Optional[str], optional): The path to save the graph (if None, then the graph is not saved).
        Defaults to None.
        show (bool, optional): display a graph or not. Defaults to True.
    """
    plt.figure(figsize=figsize)
    plt.plot(x_values, left_values, color="b", label="confidence interval")
    plt.plot(x_values, right_values, color="b")
    plt.plot(x_values, center_values, color="r", label="metric")

    y_min = min(left_values)
    y_max = max(right_values)
    y_start = int((y_min // 5) * 5)
    # Rounding y_max up to the nearest multiple of 5
    y_end = int(((y_max + 4) // 5) * 5)

    # Creating a list with step 5
    yticks = list(range(y_start, y_end + 1, 5))

    plt.xticks(x_values)
    plt.yticks(yticks)
    plt.fill_between(x_values, left_values, right_values, color="b", alpha=0.15)
    plt.xlabel("IOU")
    plt.ylabel("value")
    plt.legend()
    plt.title(title)
    plt.grid()
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved as: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()
