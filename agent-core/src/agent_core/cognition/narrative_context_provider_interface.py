# src/agent_core/cognition/narrative_context_provider_interface.py
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Type

if TYPE_CHECKING:
    from agent_core.core.ecs.abstractions import SimulationState
    from agent_core.core.ecs.component import Component


class NarrativeContextProviderInterface(ABC):
    """
    Abstract Base Class for narrative context providers.

    This interface defines the contract for how a simulation's state (including
    world-specific components) is transformed into a narrative string
    suitable for LLM-based reflection.

    The concrete implementation of this interface will live in the final
    simulation application (e.g., agent-soul-sim) and will be responsible
    for extracting relevant information from various components (both
    cognitive and world-specific) to build a coherent narrative.
    """

    @abstractmethod
    def get_narrative_context(
        self,
        entity_id: str,
        components: Dict[Type["Component"], "Component"],
        simulation_state: "SimulationState",
        current_tick: int,
        config: Any,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Constructs a detailed narrative context from an entity's components
        and the overall simulation state. This narrative is intended to be
        fed to an LLM for metacognition.

        Args:
            entity_id: The ID of the agent whose context is being generated.
            components: A dictionary of the entity's components.
            simulation_state: The current overall simulation state, allowing
                              access to global information or other entities.
            current_tick: The current simulation tick.

        Returns:
            A string representing the narrative context.
        """
        raise NotImplementedError
