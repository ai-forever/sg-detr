import json
import os
from collections import defaultdict
from typing import Dict, List

import click
import numpy as np


def parse_metrics(logs_dir: str, metric_name: str = 'val/HL-mAP-Binary') -> Dict[str, List[float]]:
    """
    Parse metrics for experiments where there are many domains and seeds.

    Args:
        logs_dir (str): log dir of the experiment
        metric_name (str, optional): name of the main metric. Defaults to 'val/HL-mAP-Binary'.

    Returns:
        Dict[str, List[float]]: metrics, in format domain: list of metrics(different seed)
    """
    metrics_dict = defaultdict(list)
    main_log_dir = os.path.basename(os.path.normpath(logs_dir))

    # Going through all folders in the logs_dir directory
    for root, _, files in os.walk(logs_dir):
        # looking for the metrics.json file
        if 'metrics.json' in files:
            metrics_file = os.path.join(root, 'metrics.json')
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                # Getting the `metric_name` value
                if metric_name in metrics:
                    val_map = metrics[metric_name]

                    # Defining domain from the folder name
                    path_parts = root.split(os.sep)
                    domain_folder = path_parts[path_parts.index(main_log_dir) + 1]
                    domain = domain_folder.split('_')[1]

                    metrics_dict[domain].append(val_map)

    return dict(metrics_dict)


@click.command()
@click.option("--logs_directory", type=str)
@click.option("--metric_name", type=str)
def main(logs_directory: str, metric_name: str) -> None:
    """
    Calculates metrics for a series of experiments for different domains and different seeds.

    Args:
        logs_directory (str): log dir of the experiment
        metric_name (str): name of the main metric
    """
    results = parse_metrics(logs_directory, metric_name)
    domain_keys = list(results.keys())
    macro_metrics = []
    macro_std = []
    for key in domain_keys:
        value = results[key]
        macro_metrics.append(np.mean(value))
        macro_std.append(np.std(value))
        results[f"{key}_mean"] = np.round(np.mean(value) * 100, 1)
        results[f"{key}_std"] = np.round(np.std(value) * 100, 1)
    results['macro_mean'] = np.round(np.mean(macro_metrics) * 100, 1)
    results['macro_std'] = np.round(np.mean(macro_std) * 100, 1)
    with open(os.path.join(logs_directory, "metrics.json"), "w") as file:
        json.dump(results, file)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
