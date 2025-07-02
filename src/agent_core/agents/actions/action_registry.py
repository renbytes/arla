# src/agent_core/agents/actions/action_registry.py
"""
Defines the ActionRegistry, a singleton object that discovers and manages
all available actions in the simulation.
"""

from typing import Dict, List, Type, cast

from src.agent_core.agents.actions.action_interface import ActionInterface


class ActionRegistry:
    """
    A registry for discovering, storing, and retrieving action classes.
    """

    def __init__(self) -> None:
        self._actions: Dict[str, Type[ActionInterface]] = {}
        print("ActionRegistry initialized.")

    def register(self, action_class: Type[ActionInterface]) -> Type[ActionInterface]:
        """
        A decorator to register any class that implements the ActionInterface.
        """
        if not issubclass(action_class, ActionInterface):
            raise TypeError(f"Class {action_class.__name__} must implement ActionInterface to be registered.")

        action_id = cast(str, action_class.action_id)
        if action_id in self._actions:
            raise ValueError(f"Action with ID '{action_id}' is already registered.")

        self._actions[action_id] = action_class
        action_name = cast(str, action_class.name)
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
