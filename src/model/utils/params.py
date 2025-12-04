"""Select params."""

from typing import List, Optional

from torch import nn


def get_params_by_name(
    model: nn.Module,
    include_prefixes: Optional[List[str]] = None,
    exclude_prefixes: Optional[List[str]] = None,
) -> List[nn.Parameter]:
    """
    Retrieve model parameters based on name prefixes.

    Args:
        model (nn.Module): The model from which parameters are retrieved.
        include_prefixes (Optional[List[str]]): List of prefixes to include.
        exclude_prefixes (Optional[List[str]]): List of prefixes to exclude.

    Returns:
        List[nn.Parameter]: A list of parameters that match the include and exclude criteria.
    """
    params = []
    for name, param in model.named_parameters():
        if include_prefixes and not any(name.startswith(prefix) for prefix in include_prefixes):
            continue
        if exclude_prefixes and any(name.startswith(prefix) for prefix in exclude_prefixes):
            continue
        params.append(param)
    return params
