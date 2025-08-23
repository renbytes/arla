# FILE: agent-core/src/agent_core/environment/perception_provider_interface.py

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Type

if TYPE_CHECKING:
    from agent_core.core.ecs.abstractions import SimulationState
    from agent_core.core.ecs.component import Component


class PerceptionProviderInterface(ABC):
    """

    An abstract interface for a class that provides world-specific
    sensory information to an agent's PerceptionComponent.
    """

    @abstractmethod
    def update_perception(
        self,
        entity_id: str,
        components: Dict[Type["Component"], "Component"],
        simulation_state: "SimulationState",
        current_tick: int,
    ) -> None:
        """
        Populates the PerceptionComponent of an entity with data about
        visible entities in the environment.

        Args:
            entity_id: The ID of the agent whose perception is being updated.
            components: The components of the perceiving agent.
            simulation_state: The current state of the entire simulation.
            current_tick: The current simulation tick.
        """
        raise NotImplementedError
