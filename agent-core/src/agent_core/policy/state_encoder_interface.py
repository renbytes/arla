# src/agent_core/policy/state_encoder_interface.py
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Type

import numpy as np

if TYPE_CHECKING:
    from agent_core.core.ecs.abstractions import SimulationState
    from agent_core.core.ecs.component import Component


class StateEncoderInterface(ABC):
    """
    Abstract Base Class for a state encoder.
    This interface allows the Q-learning system to be decoupled from the specific
    state representation of any given simulation. The concrete implementation will
    be provided by the final application (e.g., agent-soul-sim) and will contain
    the logic for converting that world's state into a feature vector.
    """

    @abstractmethod
    def encode_state(
        self,
        simulation_state: "SimulationState",
        entity_id: str,
        config: Dict[str, Any],
        target_entity_id: Optional[str] = None,
    ) -> np.ndarray:
        """
        Encodes the current simulation state into a fixed-size numerical
        feature vector from the perspective of a given entity.

        Args:
            simulation_state: The main state object of the simulation.
            entity_id: The ID of the entity for whom the state is being encoded.
            config: The simulation configuration dictionary.
            target_entity_id: An optional ID of another entity that is the
                              focus of the current action, for social contexts.
        Returns:
            A 1D numpy array representing the encoded state features.
        """
        raise NotImplementedError

    @abstractmethod
    def encode_internal_state(
        self, components: Dict[Type["Component"], "Component"], config: Any
    ) -> np.ndarray:
        """
        Encodes an agent's internal, cognitive components into a fixed-size
        numerical feature vector.

        Args:
            components: The dictionary of components for the specific agent.
            config: The simulation configuration object.

        Returns:
            A 1D numpy array representing the encoded internal state features.
        """
        raise NotImplementedError
