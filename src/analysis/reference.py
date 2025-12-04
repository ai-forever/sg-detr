# flake8: noqa: C901
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from loguru import logger
from torch import Tensor

from src.analysis.data_models import VideoPrediction
from src.analysis.utils import WINDOW_RANGES, categorize_span
from src.utils.span_utils import span_cxw_to_xx


def get_reference_points(state_dict: Dict[str, Tensor]) -> np.ndarray:
    """
    Get reference points from model state dict

    Args:
        state_dict (Dict[str, Tensor]): model state dict

    Returns:
        np.ndarray: normalized reference points, shape [n_points, 2], center of the interval and it weights
    """
    center = state_dict["main_det_head.refpoint_embed.center.weight"]
    width = state_dict["main_det_head.refpoint_embed.width.weight"]
    ref_points = torch.cat([center, width], dim=-1)
    reference_points = torch.sigmoid(ref_points)
    return span_cxw_to_xx(reference_points).detach().cpu().numpy()


def plot_ref_points(ref_points, title: str = "Visualization of ref_points", show=True, save_path=None, figsize=(10, 6)):
    """
    Visualizes trainable ref_points.

    Args:
        ref_points (np.ndarray): An array of points, where each point has a center and a weight.
        title (str): Plot title. Defaults to "Visualization of ref_points".
        show (bool): If True, displays the plot.
        save_path (str): If provided, saves the plot to the specified file.
        figsize (tuple): Figure size as a tuple (width, height).
    """
    center_points = ref_points.mean(1)
    sorted_idx = np.argsort(center_points)

    starts = ref_points[:, 0]
    ends = ref_points[:, 1]

    plt.figure(figsize=figsize)  # Allows custom figure size

    for idx in range(len(sorted_idx)):
        s_idx = sorted_idx[idx]
        plt.plot([starts[s_idx], ends[s_idx]], [idx, idx], marker="o")

    plt.ylabel("Sorted ref points idx")
    plt.xlabel("Ref interval")
    plt.title(title)

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved as: {save_path}")

    if show:
        plt.show()
    plt.close()  # Closes the figure to free up memory


def plot_reference_shift_boxplot(
    video_predictions: List[VideoPrediction],
    reference_points: np.ndarray,
    title: str = "Distribution of Shifts for Reference Points",
    figsize: Tuple[int, int] = (16, 6),
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plots the distribution of absolute shifts in center and width of spans from their corresponding reference points.

    This function generates two boxplots side by side in a single figure. The first boxplot shows the distribution of
    absolute shifts in the centers of the predicted spans relative to their reference points. The second boxplot shows
    the distribution of absolute shifts in the widths.

    Args:
        video_predictions (List[VideoPrediction]): Predictions for each video.
        reference_points (np.ndarray): Reference spans(shape: [n_points, 2]), beginning and
        end(normalized from 0 d to 1).
        title: str: Plot title. Defaults to "Distribution of Shifts for Reference Points".
        figsize (Tuple[int, int], optional): Figure size as a tuple (width, height).. Defaults to (10, 6).
        save_path (Optional[str], optional): The path to save the graph (if None, then the graph is not saved).
        Defaults to None.
        show (bool, optional): display a graph or not. Defaults to True.
    """
    # Extract centers and widths of reference points
    reference_centers = [(rp[0] + rp[1]) / 2 for rp in reference_points]
    reference_widths = [rp[1] - rp[0] for rp in reference_points]

    # Initialize lists to store predicted centers, widths, and corresponding reference indexes
    list_pred_centers = []
    list_pred_widths = []
    list_reference_idxs = []

    # Extract centers, widths of predicted spans and their corresponding reference indexes
    for vp in video_predictions:
        for pred_idx, _ in zip(*vp.matching):
            pred_span = vp.pred_spans[pred_idx]
            list_pred_centers.append((pred_span[0] + pred_span[1]) / 2)
            list_pred_widths.append(pred_span[1] - pred_span[0])
            list_reference_idxs.append(pred_idx)

    # Convert lists to numpy arrays for easier manipulation
    pred_centers = np.array(list_pred_centers)
    pred_widths = np.array(list_pred_widths)
    reference_idxs = np.array(list_reference_idxs)

    # Calculate absolute shifts in centers and widths for each reference point
    delta_centers = [np.abs(pred_centers[reference_idxs == ri] - rc) for ri, rc in enumerate(reference_centers)]
    delta_widths = [np.abs(pred_widths[reference_idxs == ri] - rw) for ri, rw in enumerate(reference_widths)]

    _, axs = plt.subplots(1, 2, figsize=figsize, sharey=True)

    # Boxplot for shifts in center
    axs[0].boxplot(
        delta_centers,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        boxprops=dict(facecolor="lightblue"),
        meanprops=dict(linestyle="-", linewidth=2, color="red"),
        medianprops=dict(linestyle="-", linewidth=2, color="green"),
    )
    axs[0].set_title("Shift in Center")
    axs[0].set_xlabel("Reference Point Index")
    axs[0].set_ylabel("Absolute Center Shift")
    axs[0].set_xticklabels(range(len(delta_centers)))  # Set X-axis labels to start from 0

    # Boxplot for shifts in width
    axs[1].boxplot(
        delta_widths,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        boxprops=dict(facecolor="lightgreen"),
        meanprops=dict(linestyle="-", linewidth=2, color="purple"),
        medianprops=dict(linestyle="-", linewidth=2, color="orange"),
    )
    axs[1].set_title("Shift in Width")
    axs[1].set_xlabel("Reference Point Index")
    axs[1].set_xticklabels(range(len(delta_widths)))  # Set X-axis labels to start from 0

    # axs[1].set_ylabel('Absolute Width Shift') # Removed due to sharey=True

    # Adjust layout and display the figure
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # type: ignore

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved as: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_mean_reference_shifts(
    video_predictions: List[VideoPrediction],
    reference_points: np.ndarray,
    title: str = "Mean Shifts and Width Changes of Reference Points",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Draws the average shifts for each anchor

    Args:
        video_predictions (List[VideoPrediction]): Predictions for each video.
        reference_points (np.ndarray): Reference spans(shape: [n_points, 2]), beginning and
        end(normalized from 0 d to 1).
        title: str: Plot title. Defaults to 'Mean Shifts and Width Changes of Reference Points'.
        figsize (Tuple[int, int], optional): Figure size as a tuple (width, height). Defaults to (10, 6).
        save_path (Optional[str], optional): The path to save the graph (if None, then the graph is not saved).
        Defaults to None.
        show (bool, optional): display a graph or not. Defaults to True.
    """
    list_reference_centers = []
    list_reference_widths = []
    for reference_point in reference_points:
        list_reference_centers.append((reference_point[0] + reference_point[1]) / 2)
        list_reference_widths.append(reference_point[1] - reference_point[0])

    list_pred_centers = []
    list_pred_widths = []
    list_reference_idxs = []
    for video_prediction in video_predictions:
        for pred_idx, _ in zip(*video_prediction.matching):
            pred_span = video_prediction.pred_spans[pred_idx]
            pred_center = (pred_span[0] + pred_span[1]) / 2
            pred_width = pred_span[1] - pred_span[0]

            list_pred_centers.append(pred_center)
            list_pred_widths.append(pred_width)
            list_reference_idxs.append(pred_idx)

    pred_centers = np.array(list_pred_centers)
    pred_widths = np.array(list_pred_widths)
    reference_centers = np.array(list_reference_centers)
    reference_widths = np.array(list_reference_widths)
    reference_idxs = np.array(list_reference_idxs)

    list_delta_centers = []
    list_delta_widths = []
    for reference_idx, (reference_center, reference_width) in enumerate(zip(reference_centers, reference_widths)):
        list_delta_centers.append(np.mean(pred_centers[reference_idxs == reference_idx] - reference_center))
        list_delta_widths.append(np.mean(pred_widths[reference_idxs == reference_idx] - reference_width))
    delta_centers = np.array(list_delta_centers)
    delta_widths = np.array(list_delta_widths)

    plt.figure(figsize=figsize)
    plt.quiver(
        reference_centers,
        reference_widths,
        delta_centers,
        delta_widths,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="red",
        width=0.004,
    )
    plt.scatter(reference_centers, reference_widths, color="blue", label="Reference Points")
    plt.scatter(
        reference_centers + delta_centers,
        reference_widths + delta_widths,
        color="green",
        label="Mean Predicted Points",
        alpha=0.5,
    )
    plt.xlabel("Center of Spans")
    plt.ylabel("Width of Spans")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, format="png", metadata=None)
        logger.info(f"Plot saved as: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_spans_grouped_by_reference(
    video_predictions: List[VideoPrediction],
    reference_points: np.ndarray,
    mode: str = "predicted",
    reference_size: int = 50,
    title: str = "Spans grouped by Reference Points",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plots spans (either predicted or ground truth) grouped by reference points. Reference points
    are highlighted with a special marker.

    Args:
        video_predictions (List[VideoPrediction]): Predictions for each video.
        reference_points (np.ndarray): Reference spans(shape: [n_points, 2]), beginning and
        end(normalized from 0 d to 1).
        mode (str, optional): if `predicted` predicted spans will be shown if `gt` groun thruth will be shown.
        Defaults to 'predicted'.
        reference_size (int, optional): size of reference points. Defaults to 50.
        title: str: Plot title. Defaults to "Spans grouped by Reference Points".
        figsize (Tuple[int, int], optional): Figure size as a tuple (width, height).. Defaults to (10, 6).
        save_path (Optional[str], optional):The path to save the graph (if None, then the graph is not saved).
        Defaults to None.
        show (bool, optional): display a graph or not. Defaults to True.

    Raises:
        ValueError: if mode is incorrect
    """
    if mode not in ("predicted", "gt"):
        raise ValueError(f"Unexpected `mode`: {mode}, expected `predicted` or `gt`")
    group_colors = plt.colormaps.get_cmap("tab20")
    grouped_spans: Dict[int, Tuple[List[float], List[float]]] = {
        ref_idx: ([], []) for ref_idx in range(len(reference_points))
    }
    reference_centers = []
    reference_widths = []
    for reference_point in reference_points:
        reference_centers.append((reference_point[0] + reference_point[1]) / 2)
        reference_widths.append(reference_point[1] - reference_point[0])

    for vp in video_predictions:
        for pred_idx, gt_idx in zip(*vp.matching):
            span = vp.pred_spans[pred_idx] if mode == "predicted" else vp.gt_spans[gt_idx]
            center = (span[0] + span[1]) / 2
            width = span[1] - span[0]
            grouped_spans[pred_idx][0].append(center)
            grouped_spans[pred_idx][1].append(width)
    plt.figure(figsize=figsize)
    for reference_idx, (reference_center, reference_width) in enumerate(zip(reference_centers, reference_widths)):
        color = group_colors(reference_idx)
        plt.scatter(
            reference_center,
            reference_width,
            color=color,
            edgecolors="black",
            label=f"Reference {reference_idx}",
            s=reference_size,
        )
        span_centers, span_widths = grouped_spans[reference_idx]
        plt.scatter(span_centers, span_widths, color=color, s=2, alpha=0.4)

    plt.xlabel("Center of Spans")
    plt.ylabel("Width of Spans")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(title)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))  # Move legend outside of the plot
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, format="png", metadata=None)
        logger.info(f"Plot saved as: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_matched_spans_histogram(
    video_predictions: List[VideoPrediction],
    reference_points: Optional[np.ndarray] = None,
    title: str = "Matched GT Spans by Category and Reference Point",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Builds a distribution of the number of fragments for each query, broken down by their types (by length)

    Args:
        video_predictions (List[VideoPrediction]): Predictions for each video.
        reference_points (Optional[np.ndarray], optional): Reference spans(shape: [n_points, 2]), beginning and
        end(normalized from 0 d to 1). Defaults to None.
        title: str: Plot title. Defaults to 'Matched GT Spans by Category and Reference Point'.
        figsize (Tuple[int, int], optional): Figure size as a tuple (width, height). Defaults to (12, 5).
        save_path (Optional[str], optional): The path to save the graph (if None, then the graph is not saved).
        Defaults to None.
        show (bool, optional): display a graph or not. Defaults to True.
    """

    ref_point_matches: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    if reference_points is not None:
        ref_point_centers = [(start + end) / 2 for start, end in reference_points]
        sorted_centers = np.sort(ref_point_centers)
    else:
        sorted_centers = np.arange(len(video_predictions[0].pred_probs))

    for video_prediction in video_predictions:
        for pred_idx, gt_idx in zip(*video_prediction.matching):
            gt_span = video_prediction.gt_spans[gt_idx]
            category = categorize_span(gt_span)
            if reference_points is not None:
                ref_center = (video_prediction.pred_spans[pred_idx][0] + video_prediction.pred_spans[pred_idx][1]) / 2
                sorted_pred_idx = np.argmin(np.abs(sorted_centers - ref_center))
            else:
                sorted_pred_idx = pred_idx
            ref_point_matches[sorted_pred_idx][category] += 1  # type: ignore

    _, ax = plt.subplots(figsize=figsize)
    categories = list(WINDOW_RANGES.keys())
    ind = np.arange(len(ref_point_matches))  # X locations for groups
    width = 0.2

    for idx, category in enumerate(categories):
        counts = [matches[category] for key, matches in sorted(ref_point_matches.items(), key=lambda x: x[0])]
        ax.bar(ind + idx * width, counts, width, label=category)

    ax.set_xlabel("Reference Points" if reference_points is None else "Reference Point Centers")
    ax.set_ylabel("Number of Matched GT Spans")
    ax.set_title(title)
    ax.set_xticks(ind + width / len(categories) * (len(categories) - 1))
    ax.set_xticklabels(
        (
            [f"{i}" for i in sorted_centers]
            if reference_points is None
            else [f"{center:.2f}" for center in sorted_centers]
        ),
        rotation=45,
    )
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved as: {save_path}")
    if show:
        plt.show()
    plt.close()


def plot_mean_reference_shifts_plotly(
    video_predictions,
    reference_points,
    figsize=(600, 400),
    save_path=None,
    show=True,
):
    """
    Draws the average shifts for each anchor using Plotly.
    Note: same as `plot_mean_reference_shifts`

    Args:
        video_predictions: Predictions for each video.
        reference_points: Reference spans (shape: [n_points, 2]), beginning and end (normalized from 0 to 1).
        figsize: Figure size in pixels (width, height).
        save_path: The path to save the graph (if None, then the graph is not saved).
        show: Display the graph or not.
    """

    list_reference_centers = []
    list_reference_widths = []
    for reference_point in reference_points:
        list_reference_centers.append((reference_point[0] + reference_point[1]) / 2)
        list_reference_widths.append(reference_point[1] - reference_point[0])

    list_pred_centers = []
    list_pred_widths = []
    list_reference_idxs = []
    for video_prediction in video_predictions:
        for pred_idx, _ in zip(*video_prediction.matching):
            pred_span = video_prediction.pred_spans[pred_idx]
            pred_center = (pred_span[0] + pred_span[1]) / 2
            pred_width = pred_span[1] - pred_span[0]

            list_pred_centers.append(pred_center)
            list_pred_widths.append(pred_width)
            list_reference_idxs.append(pred_idx)

    pred_centers = np.array(list_pred_centers)
    pred_widths = np.array(list_pred_widths)
    reference_centers = np.array(list_reference_centers)
    reference_widths = np.array(list_reference_widths)
    reference_idxs = np.array(list_reference_idxs)

    list_delta_centers = []
    list_delta_widths = []
    for reference_idx, (reference_center, reference_width) in enumerate(zip(reference_centers, reference_widths)):
        list_delta_centers.append(np.mean(pred_centers[reference_idxs == reference_idx] - reference_center))
        list_delta_widths.append(np.mean(pred_widths[reference_idxs == reference_idx] - reference_width))
    delta_centers = np.array(list_delta_centers)
    delta_widths = np.array(list_delta_widths)

    # Create a Plotly figure
    fig = go.Figure()

    # Add scatter trace for reference points
    fig.add_trace(
        go.Scatter(
            x=reference_centers, y=reference_widths, mode="markers", name="Reference Points", marker=dict(color="blue")
        )
    )

    # Add scatter trace for mean predicted points
    fig.add_trace(
        go.Scatter(
            x=reference_centers + delta_centers,
            y=reference_widths + delta_widths,
            mode="markers",
            name="Mean Predicted Points",
            marker=dict(color="green"),
        )
    )

    # Add lines between the reference and predicted points
    for ref_center, ref_width, delta_center, delta_width in zip(
        reference_centers, reference_widths, delta_centers, delta_widths
    ):
        # print(ref_center, ref_width, delta_center, delta_width)
        if not np.isnan(delta_center) and not np.isnan(delta_width):
            fig.add_shape(
                type="line",
                x0=ref_center,
                y0=ref_width,
                x1=ref_center + delta_center,
                y1=ref_width + delta_width,
                line=dict(color="red", width=2),
            )

    # Update layout
    fig.update_layout(
        title="Mean Shifts and Width Changes of Reference Points",
        xaxis_title="Center of Spans",
        yaxis_title="Width of Spans",
        width=figsize[0],
        height=figsize[1],
    )

    # Show or save the figure
    if show:
        fig.show()
    if save_path:
        fig.write_image(save_path)


def plot_reference_history(
    ref_history: List[np.ndarray],
    sensitivity: float = 0.05,
    title: str = "References Movement across epochs",
    figsize=(10, 8),
    save_path=None,
    show=True,
) -> None:
    """
    Plots the movement of points over time from a list of np.ndarray with lines and arrows.

    Args:
        ref_history (List[np.ndarray]): List of numpy arrays, each with shape [n, 2], representing point movements over time.
        sensitivity (float): Threshold to skip small movements.
    """
    plt.figure(figsize=figsize)
    num_histories = len(ref_history)
    colors = cm.rainbow(np.linspace(0, 1, num_histories))  # type: ignore

    for history, color in zip(ref_history, colors):
        filtered_points_list = [history[0]]
        for point in history[1:]:
            if np.linalg.norm(point - filtered_points_list[-1]) >= sensitivity:
                filtered_points_list.append(point)

        filtered_points = np.array(filtered_points_list)

        # Plot lines connecting points
        plt.plot(filtered_points[:, 0], filtered_points[:, 1], marker="o", color=color, alpha=0.6)

        # Plot arrows on the lines
        for i in range(len(filtered_points) - 1):
            start_point = filtered_points[i]
            end_point = filtered_points[i + 1]
            dx, dy = end_point - start_point
            plt.quiver(
                start_point[0],
                start_point[1],
                dx,
                dy,
                angles="xy",
                scale_units="xy",
                scale=1,
                color=color,
                alpha=0.8,
                width=0.003,
                headwidth=3,
                headlength=5,
            )

    plt.xlabel("center")
    plt.ylabel("width")
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved as: {save_path}")
    if show:
        plt.show()
    plt.close()
