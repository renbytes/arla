# src/agent_engine/policy/state_encoder.py

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class StateEncoder(ABC):
    """
    Abstract Base Class for a state encoder.

    This interface allows the Q-learning system to be decoupled from the specific
    state representation of any given simulation. The concrete implementation will
    be provided by the final application (e.g., agent-soul-sim) and will contain
    the logic for converting that world's state into a feature vector.
    """

    @abstractmethod
    def encode(self, simulation_state: Any, entity_id: str, target_entity_id: Optional[str] = None) -> np.ndarray:
        """
        Encodes the current simulation state into a fixed-size numerical
        feature vector from the perspective of a given entity.

        Args:
            simulation_state: The main state object of the simulation.
            entity_id: The ID of the entity for whom the state is being encoded.
            target_entity_id: An optional ID of another entity that is the
                              focus of the current action, for social contexts.

        Returns:
            A 1D numpy array representing the encoded state features.
        """
        raise NotImplementedError
