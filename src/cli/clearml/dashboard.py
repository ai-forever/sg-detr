"""Dashboard."""

from typing import Any, Dict, Tuple

import click
import numpy as np
import pandas as pd
from plotly import graph_objects as go  # noqa: WPS111
from plotly.colors import n_colors
from plotly.subplots import make_subplots
from scipy import stats

from src.utils.rw_utils import load_json

CHART_HEIGHT: int = 500
CONFIDENCE_INTERVAL_DIVISOR: float = 2.0


def calculate_confidence_interval(std_dev: float, sample_size: int, confidence_level: float) -> Tuple[float, float]:
    """
    Calculate the confidence interval for a given standard deviation, sample size, and confidence level.

    Args:
        std_dev (float): The standard deviation of the sample.
        sample_size (int): The size of the sample.
        confidence_level (float): The confidence level for the interval.

    Returns:
        Tuple[float, float]: The lower and upper bounds of the confidence interval.
    """
    t_value = stats.t.ppf((1 + confidence_level) / CONFIDENCE_INTERVAL_DIVISOR, sample_size - 1)
    return std_dev * t_value / np.sqrt(sample_size)


def prepare_test_metrics(test_data: Dict[str, Any], confidence_level: float = 0.95) -> pd.DataFrame:
    """
    Prepare test metrics from the given test data and calculate confidence intervals for each metric.

    Args:
        test_data (Dict[str, Any]): A dict where keys are experiment names and values are dictionaries of test data.
        confidence_level (float): The confidence level for the confidence interval calculation. Defaults to 0.95.

    Returns:
        pd.DataFrame: A DataFrame containing the results for all exps with means, std, and ci for each metric.
    """
    combined_results = {}

    # Process each experiment
    for experiment_name, experiment_data in test_data.items():
        experiment_df = pd.DataFrame(experiment_data)
        experiment_results = {}

        for metric_name in experiment_df.columns:
            mean_value = experiment_df[metric_name].mean()
            std_dev = experiment_df[metric_name].std(ddof=1)
            sample_size = len(experiment_df[metric_name])
            margin_of_error = calculate_confidence_interval(std_dev, sample_size, confidence_level)
            confidence_interval = (mean_value - margin_of_error, mean_value + margin_of_error)

            experiment_results[metric_name] = {
                "mean": mean_value,
                "std": std_dev,
                "ci_lower": confidence_interval[0],
                "ci_upper": confidence_interval[1],
            }

        combined_results[experiment_name] = pd.DataFrame(experiment_results).T

    # Combine results from all experiments into a single DataFrame for each metric
    return pd.concat(combined_results, axis=1)


@click.command()
@click.option("--json_file", type=click.Path(exists=True), default="data_local/metrics/metrics_for_exp_1_baseline.json")
def generate_metrics_dashboard(json_file: str) -> None:
    """Generate validation and test metrics dashboard from JSON file.

    Args:
        json_file (str): path to json file.
    """
    # Load the JSON file
    data = load_json(json_file)

    # Extract all experiment names
    experiments = data.keys()

    # Create a dictionary to hold validation metrics dataframes for each experiment
    val_metrics = {exp: pd.DataFrame(data[exp]["val"]) for exp in experiments}

    # Combine test metrics for all experiments into a single DataFrame
    test_part = {exp: data[exp]["test"] for exp in experiments}
    test_metrics_combined = prepare_test_metrics(test_part)

    # Define metric pairs if their names do not correspond directly
    metrics = list(val_metrics[next(iter(experiments))].columns)
    val_metrics_titles = [f"Validation {metric}" for metric in metrics]
    test_metrics_titles = [f"Test {metric}" for metric in metrics]
    titles = [elem for pair in zip(val_metrics_titles, test_metrics_titles) for elem in pair]
    num_metrics = len(metrics)

    # Define a color palette
    colors = n_colors("rgb(0, 100, 255)", "rgb(255, 65, 54)", len(experiments) + 1, colortype="rgb")
    experiment_colors = dict(zip(experiments, colors[: len(experiments)]))

    fig = make_subplots(
        rows=num_metrics,
        cols=2,
        shared_xaxes=False,
        vertical_spacing=0.025,  # noqa: WPS432
        horizontal_spacing=0.1,
        subplot_titles=titles,
    )

    # Add validation metrics to the first column and test metrics to the second column
    for idx, metric in enumerate(metrics):
        for exp in experiments:
            # Add validation metric
            fig.add_trace(
                go.Scatter(
                    x=val_metrics[exp].index,
                    y=val_metrics[exp][metric].round(4),
                    mode="lines+markers",
                    line={"width": 1.5, "color": experiment_colors[exp]},
                    marker={"size": 2.5},
                    name=exp,
                    legendgroup=f"group{idx}",  # Group legends by metric index
                    showlegend=True if idx == 0 else False,  # noqa: WPS502
                ),
                row=idx + 1,
                col=1,
            )

        # Add test metric
        exp_results = test_metrics_combined.loc[metric].reset_index(1)
        exp_results_mean = exp_results.loc[exp_results["level_1"] == "mean"].drop(columns=["level_1"])
        exp_results_ci_upper = exp_results.loc[exp_results["level_1"] == "ci_upper"].drop(columns=["level_1"])
        exp_results_ci_lower = exp_results.loc[exp_results["level_1"] == "ci_lower"].drop(columns=["level_1"])
        fig.add_trace(
            go.Bar(
                x=exp_results_mean.index,
                y=exp_results_mean[metric].round(4),
                text=exp_results_mean[metric].round(4),
                error_y={
                    "type": "data",
                    "array": (exp_results_ci_upper[metric] - exp_results_mean[metric]).tolist(),
                    "visible": True,
                },
                textposition="auto",
                marker={"color": [experiment_colors[exp] for exp in exp_results_mean.index]},  # noqa: WPS441
                name=f"{metric} (test)",
                showlegend=False,  # Show legend only once per column
            ),
            row=idx + 1,
            col=2,
        )

        # Update y-axis range for test metrics
        fig.update_yaxes(
            range=(exp_results_ci_lower[metric].min() - 2, exp_results_ci_upper[metric].max() + 2),  # noqa: WPS221
            row=idx + 1,
            col=2,
        )

        # Adjust x-axis tick angle
        fig.update_xaxes(tickangle=10, row=idx + 1, col=2)

    fig.update_layout(
        height=CHART_HEIGHT * num_metrics,  # Adjust height based on the number of rows
        title_text="Validation and Test Metrics Dashboard",
        template="plotly_white",
        showlegend=True,
        margin={"t": 50, "b": 50},
        title_x=0.5,
    )
    fig.update_yaxes(showgrid=True)
    fig.show()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    generate_metrics_dashboard()
