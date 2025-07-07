# src/agent_core/agents/action_generator_interface.py

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from agent_core.core.ecs.abstractions import SimulationState
    from agent_core.core.ecs.component import ActionPlanComponent


class ActionGeneratorInterface(ABC):
    """
    Abstract Base Class for an action generator.

    The concrete implementation of this interface will live in the final
    simulation application (e.g., agent-soul-sim). It is responsible for
    querying the global action registry and generating all possible, valid
    parameter combinations for each action for a given entity at a specific moment.
    """

    @abstractmethod
    def generate(
        self, simulation_state: "SimulationState", entity_id: str, current_tick: int
    ) -> List["ActionPlanComponent"]:
        """
        Generates all valid ActionPlanComponent instances for a given entity.

        Args:
            simulation_state: The current state of the simulation.
            entity_id: The ID of the agent for whom to generate actions.
            current_tick: The current simulation tick.

        Returns:
            A list of all possible ActionPlanComponent objects for the entity.
        """
        raise NotImplementedError
