"""
This file now only contains simple data structures used across the action system.
The CoreActionType enum and static Action class have been replaced by the
new plugin-based registry system.
"""

# src/agent_core/agents/actions/base_action.py

from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, List

from agent_core.agents.actions.action_interface import ActionInterface
from agent_core.agents.actions.action_outcome import ActionOutcome


class Intent(Enum):
    """Enumeration of high-level modifiers or motivations for actions."""

    SOLITARY = "SOLITARY"
    COOPERATE = "COOPERATE"
    COMPETE = "COMPETE"


class Action(ActionInterface):
    """
    A concrete base class for all actions that implements the ActionInterface.
    This class can be extended by specific actions like MoveAction, CombatAction, etc.
    """

    @property
    @abstractmethod
    def action_id(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    def get_base_cost(self, simulation_state: Any) -> float:
        """The base time budget cost to perform the action."""
        return 1.0  # Default cost, can be overridden

    @abstractmethod
    def generate_possible_params(
        self, entity_id: str, simulation_state: Any, current_tick: int
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def execute(
        self,
        entity_id: str,
        simulation_state: Any,
        params: Dict[str, Any],
        current_tick: int,
    ) -> ActionOutcome:
        raise NotImplementedError

    @abstractmethod
    def get_feature_vector(
        self, entity_id: str, simulation_state: Any, params: Dict[str, Any]
    ) -> List[float]:
        raise NotImplementedError

    @staticmethod
    def initialize_action_registry() -> None:
        """
        A helper method to ensure all action modules are imported,
        which triggers their registration with the action_registry.
        This method is now a placeholder, as loading is handled by the
        specific simulation's entry point.
        """
        # This is intentionally left empty. The simulation runner is
        # now responsible for dynamically loading action modules.
        pass
