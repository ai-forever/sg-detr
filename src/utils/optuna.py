"""Optuna helper."""

from copy import deepcopy
from typing import Optional

import pandas as pd
import psycopg2
from loguru import logger
from omegaconf import DictConfig


def check_sampled_params(config: DictConfig) -> Optional[float]:
    """
    Check the sampled parameters from a configuration against a metrics table.

    The function dynamically navigates through the nested dictionary structure of
    the given configuration, filtering a pandas DataFrame (`metrics_table`) based on
    the hyperparameters specified. If any set of hyperparameters results in an empty
    DataFrame after filtering, the function returns None. If the filtering is successful
    for all hyperparameters, the function returns the first entry in the 'value' column
    of the filtered DataFrame as a float.

    Args:
        config (DictConfig): A dictionary containing the configuration of hyperparameters

    Returns:
        Optional[float]: The first value in the 'value' column of the filtered DataFrame,
        if any, as a float. Returns None if the DataFrame is empty after filtering for any
        of the hyperparameters.
    """
    metrics_table = extract_data_from_postgres(config)
    if metrics_table is None:
        return None
    metrics_table.dropna(inplace=True)

    for hp_key in metrics_table.columns[1:-1]:
        sub_keys = hp_key.split(".")
        selected_value = deepcopy(config)
        for key in sub_keys:
            selected_value = selected_value[key]  # Dynamically access nested dictionary values
        # Apply the filter to the DataFrame and reassign
        metrics_table = metrics_table.loc[metrics_table[hp_key] == selected_value]

        # If filtering results in an empty DataFrame, return None
        if metrics_table.empty:
            return None

    return float(metrics_table["value"].max())


def extract_data_from_postgres(config: DictConfig) -> Optional[pd.DataFrame]:  # noqa: WPS231
    """Extract data from sql db.

    Args:
        config (DictConfig): A dictionary containing the configuration of hyperparameters

    Returns:
        Optional[pd.DataFrame]: extracted data in pd.DataFrame format.
    """
    try:  # noqa: WPS229
        # Connect to the PostgreSQL database
        connection = psycopg2.connect(
            user=config["db_cred"],
            password=config["db_cred"],
            host="localhost",
            database=config["database"],
        )

        # Create a cursor object using the cursor() method
        cursor = connection.cursor()

        # Extract params and its IDs
        cursor.execute("SELECT * FROM trial_params;")
        rows = cursor.fetchall()
        col_names = [desc[0] for desc in cursor.description]
        params = pd.DataFrame(rows, columns=col_names)
        params = params.pivot(index="trial_id", columns="param_name", values="param_value").reset_index()

        # Extract IDS and its metrics
        cursor.execute("SELECT * FROM trial_values;")
        rows = cursor.fetchall()
        col_names = [desc[0] for desc in cursor.description]
        metrics = pd.DataFrame(rows, columns=col_names)
        metrics = metrics.loc[:, ["trial_id", "value"]]
        return pd.merge(left=params, right=metrics, on="trial_id")

    except (Exception, psycopg2.Error) as error:  # pylint: disable=broad-exception-caught
        logger.error(f"Error while connecting to PostgreSQL: {error}")
        return None

    finally:
        # Close the cursor and connection
        if connection:
            cursor.close()
            connection.close()
