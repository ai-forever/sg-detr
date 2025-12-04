"""Error handler."""

from functools import wraps
from typing import Any, Callable, TypeVar, cast

from loguru import logger

# Creating a type variable for a "universal" function
FuncType = TypeVar("FuncType", bound=Callable[..., Any])  # pylint: disable=invalid-name


def ignore_errors(default_value: Any = None, log_errors: bool = True) -> Callable[[FuncType], FuncType]:  # noqa:WPS212
    """
    Catche exceptions in the decorated function, logs them, and returns a default value.

    Args:
        default_value: The value to return if an exception occurs (default is None).
        log_errors: Boolean indicating whether to log the exception (default is True).

    Returns:
        The original function result or a default value if an exception occurred.
    """

    def decorator(func: FuncType) -> FuncType:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)

            # pylint: disable=broad-exception-caught
            except Exception as exp:
                if log_errors:
                    logger.error(f"Error in function {func.__name__}: {exp}", exc_info=True)
                return default_value

        return cast(FuncType, wrapper)

    return decorator
