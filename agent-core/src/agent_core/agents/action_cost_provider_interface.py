# src/agent_core/agents/action_cost_provider_interface.py
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_core.core.ecs.abstractions import SimulationState


class ActionCostProviderInterface(ABC):
    """
    Abstract Base Class for an action cost provider.

    The concrete implementation of this interface will live in the final
    simulation application. It is responsible for applying the cost of
    an action to the appropriate agent component (e.g., TimeBudgetComponent,
    EnergyComponent, etc.).
    """

    @abstractmethod
    def apply_action_cost(
        self,
        entity_id: str,
        cost: float,
        simulation_state: "SimulationState",
    ) -> None:
        """
        Deducts the cost of an action from the relevant agent component.

        Args:
            entity_id: The ID of the agent performing the action.
            cost: The calculated cost of the action.
            simulation_state: The current state of the simulation.
        """
        raise NotImplementedError
