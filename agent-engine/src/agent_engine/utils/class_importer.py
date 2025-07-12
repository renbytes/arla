# agent-engine/src/agent_engine/utils/class_importer.py
import importlib
from typing import Type, cast

from agent_core.core.ecs.component import Component


def import_class(class_path: str) -> Type[Component]:
    """
    Helper to dynamically import a component class from its full path string.

    """
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        # Cast the dynamically loaded class to the expected type to satisfy mypy.
        # This tells the type checker to trust that the loaded attribute will be a
        # subclass of Component, resolving `no-any-return` error.
        component_class = getattr(module, class_name)
        return cast(Type[Component], component_class)
    except (ImportError, AttributeError, ValueError) as e:
        print(f"[ERROR] Failed to import class at path '{class_path}': {e}")
        raise
