# src/agent_core/environment/state_node_encoder_interface.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Type

# Forward reference for Component from agent_core
if "TYPE_CHECKING" not in globals():
    from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agent_core.core.ecs.component import Component


class StateNodeEncoderInterface(ABC):
    """
    Abstract Base Class for state node encoders.

    This interface defines the contract for how a specific simulation's
    world state (represented by an entity's components) is transformed
    into a generalized, symbolic tuple for the CausalGraphSystem.

    The concrete implementation of this interface will live in the final
    simulation application (e.g., agent-soul-sim) and will be responsible
    for extracting relevant information from world-specific components
    (like HealthComponent, PositionComponent, etc.) and converting it
    into a world-agnostic representation for the causal graph.
    """

    @abstractmethod
    def encode_state_for_causal_graph(
        self,
        entity_id: str,
        components: Dict[Type["Component"], "Component"],
        current_tick: int,
        config: Dict[str, Any],
    ) -> Tuple[Any, ...]:
        """
        Encodes the current state of an entity into a tuple that represents
        a node in the causal graph. This tuple should be generalized and
        not contain direct references to world-specific component types
        (e.g., 'HealthComponent', 'PositionComponent').

        Instead, it should derive abstract concepts like 'health_status',
        'location_type', 'danger_level', etc., from the world-specific components
        provided in 'components'.

        Args:
            entity_id: The ID of the entity whose state is being encoded.
            components: A dictionary of the entity's components (world-specific included).
            current_tick: The current simulation tick.
            config: The simulation configuration.

        Returns:
            A tuple representing the abstract state node for the causal graph.
            Example: ("STATE", "health_good", "at_base", "no_danger", "goal_explore")
        """
        raise NotImplementedError
