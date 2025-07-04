# src/agent_engine/utils/config_utils.py

from typing import Any, Dict

# A sentinel object to detect if a key is missing, distinguishing it from a value of None.
_sentinel = object()


def get_config_value(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Safely retrieves a value from a nested dictionary using a dot-separated path.
    """
    # Handle empty path edge case
    if not path:
        return default

    keys = path.split(".")
    value: Any = config
    for key in keys:
        if isinstance(value, dict):
            # Use the sentinel to distinguish between a missing key and a key with a None value.
            value = value.get(key, _sentinel)
            if value is _sentinel:
                return default
        else:
            # The path tries to go deeper, but the current value is not a dictionary.
            return default

    return value
