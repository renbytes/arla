# src/agent_core/agents/actions/action_interface.py
"""
Defines the interface (the contract) that all action classes must implement.
This ensures that any new action, regardless of its specific logic, will
behave in a way the rest of the simulation can understand.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from src.agent_core.core.ecs.component import SimulationState


class ActionInterface(ABC):
    """
    Abstract Base Class for all actions. Any new action created for the simulation
    must inherit from this class and implement its abstract methods and properties.
    """

    @property
    @abstractmethod
    def action_id(self) -> str:
        """A unique string identifier for the action, e.g., 'move', 'extract_resources'."""
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """A human-readable name for the action, e.g., 'Move'."""
        raise NotImplementedError

    @abstractmethod
    def get_base_cost(self, simulation_state: "SimulationState") -> float:
        """The base time budget cost to perform the action."""
        raise NotImplementedError

    @abstractmethod
    def generate_possible_params(self, entity_id: str, simulation_state: "SimulationState", current_tick: int) -> List[Dict[str, Any]]:
        """
        Generates all possible valid parameter combinations for this action
        for a given entity.
        """
        raise NotImplementedError

    @abstractmethod
    def execute(self, entity_id: str, simulation_state: "SimulationState", params: Dict[str, Any], current_tick: int) -> Dict[str, Any]:
        """
        Executes the action's logic and modifies the simulation state.
        """
        raise NotImplementedError

    @abstractmethod
    def get_feature_vector(self, entity_id: str, simulation_state: "SimulationState", params: Dict[str, Any]) -> List[float]:
        """
        Generates the feature vector for this specific action variant.
        """
        raise NotImplementedError
