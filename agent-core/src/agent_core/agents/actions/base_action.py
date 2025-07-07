"""
This file now only contains simple data structures used across the action system.
The CoreActionType enum and static Action class have been replaced by the
new plugin-based registry system.
"""
# src/agent_core/agents/actions/base_action.py

from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

from agent_core.agents.actions.action_interface import ActionInterface


class Intent(Enum):
    """Enumeration of high-level modifiers or motivations for actions."""

    SOLITARY = "SOLITARY"
    COOPERATE = "COOPERATE"
    COMPETE = "COMPETE"


class ActionOutcome:
    """Standardized structure for the result of executing an action."""

    def __init__(
        self,
        success: bool,
        message: str,
        base_reward: float,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.success = success
        self.message = message
        self.base_reward = base_reward
        self.details = details if details is not None else {}
        self.reward: float = base_reward


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
        self, entity_id: str, simulation_state: Any, params: Dict[str, Any], current_tick: int
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_feature_vector(self, entity_id: str, simulation_state: Any, params: Dict[str, Any]) -> List[float]:
        raise NotImplementedError

    @staticmethod
    def initialize_action_registry() -> None:
        """
        A helper method to ensure all action modules are imported,
        which triggers their registration with the action_registry.
        """
        # In a real application, you would import your action modules here.
        # For example: from simulations.soul_sim.actions import MoveAction, CombatAction
        # Since your actions are not defined yet, this can be empty for now.
        print("Action registry initialized (placeholder).")
        pass
