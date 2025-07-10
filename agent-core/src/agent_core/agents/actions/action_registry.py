# src/agent_core/agents/actions/action_registry.py
"""
Defines the ActionRegistry, a singleton object that discovers and manages
all available actions in the simulation.
"""

import importlib
from typing import Dict, List, Type

from agent_core.agents.actions.action_interface import ActionInterface


class ActionRegistry:
    """
    A registry for discovering, storing, and retrieving action classes.
    """

    def __init__(self) -> None:
        self._actions: Dict[str, Type[ActionInterface]] = {}
        print("ActionRegistry initialized.")

    def load_actions_from_paths(self, module_paths: List[str]) -> None:
        """
        Dynamically imports Python modules from a list of string paths.

        This is the core of the plugin system. Importing a module that
        contains an @action_registry.register decorator will cause that
        action to be registered.

        Args:
            module_paths: A list of module paths, e.g.,
                          ["simulations.soul_sim.actions.move_action"].
        """
        print(f"Dynamically loading actions from: {module_paths}")
        for path in module_paths:
            try:
                importlib.import_module(path)
                print(f"  Successfully loaded action module: {path}")
            except ImportError as e:
                print(f"WARNING: Could not import action module at '{path}'. Error: {e}")

    def register(self, action_class: Type[ActionInterface]) -> Type[ActionInterface]:
        """
        A decorator to register any class that implements the ActionInterface.
        """
        if not issubclass(action_class, ActionInterface):
            raise TypeError(f"Class {action_class.__name__} must implement ActionInterface to be registered.")

        try:
            instance = action_class()
            action_id = instance.action_id
            action_name = instance.name
        except Exception as e:
            raise TypeError(
                f"Could not instantiate action class {action_class.__name__} to read properties. Error: {e}"
            ) from e

        if not isinstance(action_id, str) or not action_id:
            raise TypeError(f"Action class {action_class.__name__} has an invalid 'action_id' property.")

        if action_id in self._actions:
            raise ValueError(f"Action with ID '{action_id}' is already registered.")

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


# Create a global singleton instance of the registry.
# All other parts of the application will import this instance.
action_registry = ActionRegistry()
