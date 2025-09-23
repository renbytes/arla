# simulations/disease_sim/environment.py
"""
Defines the environment for the Disease simulation.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from agent_core.environment.interface import EnvironmentInterface


class DiseaseEnvironment(EnvironmentInterface):
    """Manages the grid for agent positions."""

    def __init__(self, width: int, height: int, rng: np.random.Generator):
        self.width = width
        self.height = height
        self.rng = rng
        self.agent_positions: Dict[str, Tuple[int, int]] = {}

    def update_entity_position(
        self, entity_id: str, old_pos: Optional[Any], new_pos: Any
    ) -> None:
        self.agent_positions[entity_id] = new_pos

    def remove_entity(self, entity_id: str) -> None:
        if entity_id in self.agent_positions:
            del self.agent_positions[entity_id]

    def is_valid_position(self, position: Any) -> bool:
        return 0 <= position[0] < self.width and 0 <= position[1] < self.height

    def get_valid_positions(self) -> List[Any]:
        return [(x, y) for x in range(self.width) for y in range(self.height)]

    # --- Other required methods from the interface (can be placeholders) ---
    def get_neighbors(self, position: Any) -> List[Any]:
        return []

    def distance(self, pos1: Any, pos2: Any) -> float:
        return 0.0

    def can_move(self, from_pos: Any, to_pos: Any) -> bool:
        return True

    def get_entities_at_position(self, position: Any) -> Set[str]:
        return set()

    def get_entities_in_radius(self, center: Any, radius: int) -> List[Tuple[str, Any]]:
        return []

    def to_dict(self) -> Dict[str, Any]:
        return {"width": self.width, "height": self.height}

    def restore_from_dict(self, data: Dict[str, Any]) -> None:
        pass
