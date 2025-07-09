# src/simulations/soul_sim/world.py
"""
Defines the concrete environment for the 'soul_sim' simulation.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from agent_core.environment.interface import EnvironmentInterface


# A placeholder for a spatial index if you choose to implement one
class SpatialIndex:
    def __init__(self, width: int, height: int):
        self._grid: List[List[Set[str]]] = [[set() for _ in range(width)] for _ in range(height)]

    def add_entity(self, entity_id: str, position: Tuple[int, int]):
        y, x = position
        self._grid[y][x].add(entity_id)

    def remove_entity(self, entity_id: str, position: Tuple[int, int]):
        y, x = position
        if entity_id in self._grid[y][x]:
            self._grid[y][x].remove(entity_id)

    def get_entities_at_position(self, position: Tuple[int, int]) -> Set[str]:
        y, x = position
        return self._grid[y][x]


class Grid2DEnvironment(EnvironmentInterface):
    """A 2D grid environment for the simulation."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self._spatial_index = SpatialIndex(width, height)

    def get_valid_positions(self) -> List[Tuple[int, int]]:
        return [(r, c) for r in range(self.height) for c in range(self.width)]

    def get_neighbors(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        # Implementation for getting neighbors
        return []

    def distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def can_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        return True  # Simplified

    def is_valid_position(self, position: Tuple[int, int]) -> bool:
        return 0 <= position[0] < self.height and 0 <= position[1] < self.width

    def get_entities_at_position(self, position: Tuple[int, int]) -> Set[str]:
        return self._spatial_index.get_entities_at_position(position)

    def get_entities_in_radius(self, center: Tuple[int, int], radius: int) -> List[Tuple[str, Tuple[int, int]]]:
        return []  # Simplified

    def update_entity_position(self, entity_id: str, old_pos: Optional[Tuple[int, int]], new_pos: Tuple[int, int]):
        if old_pos:
            self._spatial_index.remove_entity(entity_id, old_pos)
        self._spatial_index.add_entity(entity_id, new_pos)

    def remove_entity(self, entity_id: str, position: Optional[Tuple[int, int]] = None):
        if position:
            self._spatial_index.remove_entity(entity_id, position)

    def to_dict(self) -> Dict[str, Any]:
        return {"width": self.width, "height": self.height, "type": "Grid2D"}


def init_resources(environment: Grid2DEnvironment, config: Dict, seed: int) -> Dict[str, Dict]:
    """Initializes resource nodes in the environment."""
    # Placeholder implementation
    print("Initializing resources...")
    return {}
