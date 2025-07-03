# src/agent_engine/utils/config_utils.py

from typing import Any, Dict


def get_config_value(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Safely retrieves a value from a nested dictionary using a dot-separated path.
    """
    keys = path.split(".")
    value: Any = config
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return default
        if value is None:
            return default
    return value
