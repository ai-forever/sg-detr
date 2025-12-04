"""Fetch metrics from ClearML."""

import json
import os
from typing import Optional

import click
import numpy as np
from clearml import Task
from dotenv import load_dotenv

load_dotenv(".env")

useful_metrics = (
    "HL-mAP-VeryGood",
    "HL-HIT@1-VeryGood",
    "MR-R1-Full_0.5",
    "MR-R1-Full_0.5-AUX",
    "MR-R1-Full_0.7",
    "MR-R1-Full_0.7-AUX",
    "MR-mAP-Full_0.5",
    "MR-mAP-Full_0.5-AUX",
    "MR-mAP-Full_0.75",
    "MR-mAP-Full_0.75-AUX",
    "MR-mAP-Full_Avg",
    "MR-mAP-Full_Avg-AUX",
)


@click.command()
@click.option("--tag", type=str, help="Task tag")
@click.option("--task_stem", default=None, help="Task stem to filter tasks by name")
@click.option("--project_name", default="MomentRetrieval", help="Project name to filter tasks by project")
@click.option("--outputdir", type=str, default="data_local/metrics", help="Dir to save metrics.")
def retrieve_metrics_by_tag(  # noqa:C901,WPS231
    tag: str,
    task_stem: Optional[str] = None,
    project_name: str = "MomentRetrieval",
    outputdir: str = "data_local",
) -> None:
    """Retrieve metrics by tag and save to a JSON file.

    Args:
        tag (str): task tag.
        task_stem (Optional[str]): task stem. Defaults to None.
        project_name (str): project name. Defaults to "MomentRetrieval".
        outputdir (str): dir to save metrics.
    """
    completed_experiments = Task.get_tasks(
        project_name=project_name,
        task_name=task_stem,
        tags=[tag],
        task_filter={"status": ["completed"]},
        allow_archived=False,
    )

    # Step 2: Download the metrics from each experiment
    metrics_dict = {}
    for experiment in completed_experiments:
        random_state, experiment_name = experiment.name.split("_", 1)
        metrics = experiment.get_reported_scalars()
        val_metrics = {
            key: (np.array(value["y"]) * 100 if key == "HL-mAP-VeryGood" else np.array(value["y"]))  # noqa: WPS221
            for key, value in metrics["val"].items()
            if key in useful_metrics  # noqa: WPS221
        }
        test_metrics = (
            {
                key: (
                    np.array(value["y"][0]) * 100  # noqa: WPS509
                    if key == "HL-mAP-VeryGood"
                    else np.array(value["y"][0])
                )
                for key, value in metrics["test"].items()
                if key in useful_metrics
            }
            if metrics["test"] is not None
            else None
        )

        if experiment_name not in metrics_dict:
            metrics_dict[experiment_name] = {"val": {}, "test": {}, "random_states": []}  # noqa: WPS204

        metrics_dict[experiment_name]["random_states"].append(random_state)

        for key, value in val_metrics.items():
            if key not in metrics_dict[experiment_name]["val"]:
                metrics_dict[experiment_name]["val"][key] = value
            else:
                metrics_dict[experiment_name]["val"][key] += value

        for key, value in test_metrics.items():
            if key not in metrics_dict[experiment_name]["test"]:
                metrics_dict[experiment_name]["test"][key] = [value]
            else:
                metrics_dict[experiment_name]["test"][key].append(value)

    # Compute the mean of the metrics by experiment name
    for _, metrics in metrics_dict.items():
        num_exps = len(metrics["random_states"])
        for key in metrics["val"]:
            metrics["val"][key] /= num_exps

    # Save the metrics_dict to a JSON file
    pathname = os.path.join(outputdir, f"metrics_for_{tag}.json")
    with open(pathname, "w", encoding="utf8") as file:
        json.dump(
            metrics_dict,
            file,
            indent=4,
            default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x,  # noqa: WPS221,WPS111
        )
    print(f"Saved to {pathname}")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    retrieve_metrics_by_tag()
