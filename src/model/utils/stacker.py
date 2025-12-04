"""Module to create stack of modules."""
import copy

from torch import nn


def get_clones(module: nn.Module, number_layers: int) -> nn.ModuleList:
    """Create stack of modules.

    Args:
        module (nn.Module): The module to stack.
        number_layers (int): Number of layer to stack.

    Returns:
        nn.ModuleList: The stacked modules.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(number_layers)])
