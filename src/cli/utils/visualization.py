"""Metrics visualization."""

from typing import Dict

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

ROTATION_ANGLE: int = 45


def visualize_metrics(metrics_dict: Dict[str, float], path_to_save: str):  # noqa: WPS213
    """Visualize metrics.

    Args:
        metrics_dict (Dict[str, float]): metrics
        path_to_save (str): path to output dir.
    """
    # Convert the dictionary into a DataFrame
    df = pd.DataFrame(list(metrics_dict.items()), columns=["Metric", "Value"])

    metric_names = df["Metric"]

    # Extract additional categorization for plotting
    df["Category"] = metric_names.apply(lambda name: name.split("/")[1].split("_")[0].split("@")[0])  # noqa: WPS221
    df["Subcategory"] = metric_names.apply(
        lambda name: "-".join(name.split("-")[1:]) if "-" in name else "Overall",  # noqa: WPS221
    )

    # Create subplots
    sns.color_palette("Set2")
    _, axs = plt.subplots(1, 2, figsize=(15, 10))

    # Plot HL-HIT metrics
    hl_hit_mask = df["Category"].str.contains("HL-HIT")
    sns.barplot(data=df[hl_hit_mask], x="Subcategory", y="Value", ax=axs[0])  # noqa: WPS204, WPS221
    axs[0].set_title("HL-HIT Metrics")

    # Plot HL-mAP metrics
    hl_map_mask = df["Category"].str.contains("HL-mAP")
    sns.barplot(data=df[hl_map_mask], x="Subcategory", y="Value", ax=axs[1])  # noqa: WPS204, WPS221
    axs[1].set_title("HL-mAP Metrics")

    plt.tight_layout()
    plt.savefig(f"{path_to_save}/hl-metrics.png")
    plt.close()

    # Create subplots
    _, axs = plt.subplots(2, 1, figsize=(15, 10))

    # Plot MR-R1-Full metrics
    filter_mr_r1_full = metric_names.str.contains("MR-R1-Full") & ~metric_names.str.contains("mIoU")  # noqa: WPS465
    sns.barplot(data=df[filter_mr_r1_full], x="Subcategory", y="Value", ax=axs[0])  # noqa: WPS221
    axs[0].set_title("MR-R1-Full Metrics")
    axs[0].tick_params(axis="x", rotation=ROTATION_ANGLE)

    # Plot MR-mAP-Full metrics
    filter_mr_map_full = metric_names.str.contains("MR-mAP-Full") & ~metric_names.str.contains("Avg")  # noqa: WPS465
    sns.barplot(data=df[filter_mr_map_full], x="Subcategory", y="Value", ax=axs[1])  # noqa: WPS221
    axs[1].set_title("MR-mAP-Full Metrics")
    axs[1].tick_params(axis="x", rotation=ROTATION_ANGLE)

    plt.tight_layout()
    plt.savefig(f"{path_to_save}/mr-metrics.png")
    plt.close()
