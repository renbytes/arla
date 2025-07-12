# src/agent_core/environment/controllability_provider_interface.py
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Type

if TYPE_CHECKING:
    from agent_core.core.ecs.abstractions import SimulationState
    from agent_core.core.ecs.component import Component


class ControllabilityProviderInterface(ABC):
    """
    Abstract Base Class for controllability providers.

    This interface defines the contract for how a specific simulation's
    world state (e.g., failed attempts, environmental difficulty) is
    transformed into a generalized 'controllability' score for affective appraisal.

    The concrete implementation will reside in the final simulation application
    (e.g., agent-soul-sim) and will extract data from world-specific components
    (like FailedStatesComponent, PositionComponent, etc.) to provide
    a world-agnostic controllability metric.
    """

    @abstractmethod
    def get_controllability_score(
        self,
        entity_id: str,
        components: Dict[Type["Component"], "Component"],
        simulation_state: "SimulationState",
        current_tick: int,
        config: Dict[str, Any],
    ) -> float:
        """
        Calculates a normalized controllability score (0.0 to 1.0) for the agent
        based on world-specific factors.

        A higher score indicates more perceived control or easier environment.

        Args:
            entity_id: The ID of the entity.
            components: A dictionary of the entity's components.
            simulation_state: The current overall simulation state.
            current_tick: The current simulation tick.
            config: The simulation configuration.

        Returns:
            A float between 0.0 and 1.0 representing the controllability.
        """
        raise NotImplementedError
