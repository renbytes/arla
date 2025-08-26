# src/agent_core/agents/actions/action_registry.py
"""
Defines the ActionRegistry, a singleton object that discovers and manages
all available actions in the simulation. This version is designed to be
"override-friendly" for testing purposes.
"""

import importlib
from typing import Dict, List, Type

from agent_core.agents.actions.action_interface import ActionInterface


class ActionRegistry:
    """
    A registry for discovering, storing, and retrieving action classes.

    In its default mode (`strict=False`), it allows new actions to override
    existing actions with the same `action_id`. This is useful in testing
    environments where multiple simulations define their own version of a
    common action (e.g., 'move').

    In `strict=True` mode, it will raise a ValueError if a duplicate
    action_id is registered, which is useful for production to prevent
    accidental naming collisions.
    """

    def __init__(self, strict: bool = False) -> None:
        self._actions: Dict[str, Type[ActionInterface]] = {}
        self.strict = strict
        print(
            f"ActionRegistry initialized in {'strict' if self.strict else 'override'} mode."
        )

    def load_actions_from_paths(self, module_paths: List[str]) -> None:
        """
        Dynamically imports Python modules from a list of string paths.
        """
        print(f"Dynamically loading actions from: {module_paths}")
        for path in module_paths:
            try:
                importlib.import_module(path)
                print(f"  Successfully loaded action module: {path}")
            except ImportError as e:
                print(
                    f"WARNING: Could not import action module at '{path}'. Error: {e}"
                )

    def register(self, action_class: Type[ActionInterface]) -> Type[ActionInterface]:
        """
        A decorator to register any class that implements the ActionInterface.
        """
        if not issubclass(action_class, ActionInterface):
            raise TypeError(
                f"Class {action_class.__name__} must implement ActionInterface to be registered."
            )

        try:
            instance = action_class()
            action_id = instance.action_id
            action_name = instance.name
        except Exception as e:
            raise TypeError(
                f"Could not instantiate action class {action_class.__name__} to read properties. Error: {e}"
            ) from e

        if not isinstance(action_id, str) or not action_id:
            raise TypeError(
                f"Action class {action_class.__name__} has an invalid 'action_id' property."
            )

        # Only raise an error if the registry is in strict mode.
        if action_id in self._actions and self.strict:
            raise ValueError(f"Action with ID '{action_id}' is already registered.")

        # Silently override the existing action if not in strict mode.
        self._actions[action_id] = action_class
        print(f"Action '{action_name}' registered with ID '{action_id}'.")
        return action_class

    def get_action(self, action_id: str) -> Type[ActionInterface]:
        """Retrieves an action class by its ID."""
        action = self._actions.get(action_id)
        if not action:
            raise ValueError(f"No action with ID '{action_id}' is registered.")
        return action

    def get_all_actions(self) -> List[Type[ActionInterface]]:
        """Returns a list of all registered action classes."""
        return list(self._actions.values())

    @property
    def action_ids(self) -> List[str]:
        """Returns a sorted list of all registered action IDs."""
        return sorted(self._actions.keys())


# Create a global singleton instance of the registry in non-strict mode.
action_registry = ActionRegistry(strict=False)
