# src/agent_engine/policy/state_encoder.py

from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np


class StateEncoder(ABC):
    @abstractmethod
    def encode(self, simulation_state: Any, entity_id: str, target_entity_id: Optional[str] = None) -> np.ndarray:
        """Encodes the world state into a feature vector for the Q-learning model."""
        raise NotImplementedError
