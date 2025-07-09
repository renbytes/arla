# src/agent_engine/utils/config_utils.py

from typing import Any

from omegaconf import DictConfig, OmegaConf

# A sentinel object to detect if a key is missing, distinguishing it from a value of None.
_sentinel = object()


def get_config_value(config: Any, path: str, default: Any = None) -> Any:
    """
    Safely retrieves a value from a nested dictionary or OmegaConf object using a dot-separated path.
    """
    # Handle OmegaConf objects
    if isinstance(config, DictConfig):
        return OmegaConf.select(config, path, default=default)

    # Handle empty path edge case
    if not path:
        return default

    # Existing logic for regular dicts
    keys = path.split(".")
    value: Any = config
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key, _sentinel)
            if value is _sentinel:
                return default
        else:
            return default

    return value
