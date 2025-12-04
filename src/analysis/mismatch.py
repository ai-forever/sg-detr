"""Analysis of the mismatches."""

from typing import List, Optional, Tuple

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt

from src.analysis.data_models import VideoPrediction


def compute_mismatch_stats(video_predictions: List[VideoPrediction]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the number of matches and mismatches for each reference across all provided video predictions.

    This function analyzes both the final and auxiliary predictions (from intermediate model layers) for each video.
    A match is counted if a reference is consistently matched with the same ground truth span across all model levels.
    A mismatch is recorded if there's any inconsistency in matching with the ground truth span across the levels.

    Args:
        video_predictions (List[VideoPrediction]): A list of `VideoPrediction` objects, each containing the prediction
                                                details for a single video, including auxiliary predictions.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays:
                                        - match_for_ref: The number of matches for each reference (shape: n_references),
                                        - mismatch_for_ref: The number of mismatches for each reference
                                            (shape: n_references).
    """
    # Determine the number of references from the prediction probabilities shape of the first video prediction
    n_ref = video_predictions[0].pred_probs.shape[0]

    # Initialize counters for matches and mismatches for each reference
    mismatch_for_ref = [0 for _ in range(n_ref)]
    match_for_ref = [0 for _ in range(n_ref)]

    # Iterate through each video prediction to compute matches and mismatches
    for video_prediction in video_predictions:
        # Create a mapping from the main (final) predictions
        main_map = {
            value: [key]
            for key, value in zip(video_prediction.matching[0], video_prediction.matching[1])  # noqa: WPS221
        }

        # Create mappings from auxiliary (intermediate) predictions
        auxiliary_maps = [
            {value: key for key, value in zip(auxiliary.matching[0], auxiliary.matching[1])}  # noqa: WPS221
            for auxiliary in video_prediction.auxiliary
        ]

        # Update main_map with matches from auxiliary predictions
        for auxiliary_map in auxiliary_maps:
            for key, value in auxiliary_map.items():
                assert key in main_map
                main_map[key].append(value)

        # Determine matches and mismatches by analyzing consistency across all mappings
        for _, value in main_map.items():
            if len(np.unique(value)) > 1:  # If there are inconsistencies, count as a mismatch
                mismatch_for_ref[value[0]] += 1
            else:  # If consistent across all levels, count as a match
                match_for_ref[value[0]] += 1

    # Convert match and mismatch counters to numpy arrays before returning
    return np.array(match_for_ref), np.array(mismatch_for_ref)


def plot_matching_per_reference(  # noqa: WPS213
    video_predictions: List[VideoPrediction],
    title: str = "Match and Mismatch Counts for Each Reference",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Visualize the matching and mismatching statistics for each reference in a bar chart.

    This function takes a list of `VideoPrediction` objects, which contain predictions for each video,
    and displays a bar chart where each bar represents a reference index divided into matches (blue)
    and mismatches (red). Additionally, it calculates and displays the percentage of mismatches above each column.

    Args:
        video_predictions (List[VideoPrediction]): Predictions for each video.
        title (str): Plot title. Defaults to 'Match and Mismatch Counts for Each Reference'
        figsize (Tuple[int, int]): Figure size as a tuple (width, height). Defaults to (12, 6).
        save_path (Optional[str]): The path to save the graph (if None, then the graph is not saved). Defaults to None.
        show (bool): display a graph or not. Defaults to True.
    """
    match_for_ref, mismatch_for_ref = compute_mismatch_stats(video_predictions=video_predictions)

    indices = np.arange(len(match_for_ref))  # Indices for the columns

    plt.figure(figsize=figsize)

    # Drawing columns for match_for_ref
    plt.bar(indices, match_for_ref, width=0.4, label="Matches", color="blue", edgecolor="black")  # noqa: WPS432

    # Adding columns for mismatch_for_ref on top of the previous ones, using the bottom parameter
    bars2 = plt.bar(
        indices,
        mismatch_for_ref,
        width=0.4,  # noqa: WPS432
        bottom=match_for_ref,
        label="Mismatches",
        color="red",
        edgecolor="black",
    )

    plt.xlabel("Reference Index")
    plt.ylabel("Count")
    plt.title(title)
    plt.xticks(indices)  # Setting ticks on the X axis for each index
    plt.legend()

    # Calculating and displaying the percentage of mismatches above each column
    total_counts = match_for_ref + mismatch_for_ref
    percent_mismatches = (mismatch_for_ref / total_counts) * 100

    for idx, rect in enumerate(bars2):
        height = total_counts[idx]  # Using the total height of the column to position the text
        plt.text(
            rect.get_x() + rect.get_width() / 2.0,  # noqa: WPS432
            height,
            f"{percent_mismatches[idx]:.1f}%",
            ha="center",
            va="bottom",
            color="black",
        )

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved as: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_matching_pie_chart(
    video_predictions: List[VideoPrediction],
    title: str = "Matches vs. Mismatches",
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Visualizes the total counts of matches and mismatches across all video predictions as a pie chart.

    The pie chart displays the proportion of matches to mismatches, including the counts and percentages for each
    category. This visualization helps to quickly assess the overall performance and accuracy of the predictions.


    Args:
        video_predictions (List[VideoPrediction]): Predictions for each video.
        title (str): Plot title. Defaults to 'Matches vs. Mismatches'.
        figsize (Tuple[int, int]): Figure size as a tuple (width, height). Defaults to (12, 6).
        save_path (Optional[str]): The path to save the graph (if None, then the graph is not saved).
        show (bool): display a graph or not. Defaults to True.
    """
    # Computing the total number of matches and mismatches
    match_for_ref, mismatch_for_ref = compute_mismatch_stats(video_predictions=video_predictions)
    # Counts for matches and mismatches
    matches_count = np.sum(match_for_ref)
    mismatches_count = np.sum(mismatch_for_ref)

    # Data for the pie chart
    sizes = [matches_count, mismatches_count]
    labels = ["Matches", "Mismatches"]
    colors = ["skyblue", "orange"]  # Цвета для каждой категории

    # Creating the pie chart
    plt.figure(figsize=figsize)
    plt.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct=lambda p: f"{p * (sum(sizes)) / 100:.0f}\n({p:.1f})",  # noqa: WPS111,WPS221,WPS237
        startangle=90,  # noqa: WPS432
    )

    plt.title(title)

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved as: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()
