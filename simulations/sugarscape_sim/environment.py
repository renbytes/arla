# FILE: simulations/sugarscape_sim/environment.py
"""
Defines the Sugarscape simulation environment.

This class manages the 2D grid, the distribution and regeneration of the
'sugar' resource, and provides an interface for systems to query and
interact with the world state. It implements the EnvironmentInterface
from agent-core.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from agent_core.environment.interface import EnvironmentInterface


class SugarscapeEnvironment(EnvironmentInterface):
    """
    Manages the grid, sugar resources, and agent positions for the simulation.

    This version is updated to accept a seeded NumPy random number generator
    to ensure that the initial resource distribution is fully reproducible.
    """

    def __init__(
        self,
        width: int,
        height: int,
        rng: np.random.Generator,
        initial_sugar_distribution: str = "gaussian",
        sugar_regeneration_rate: int = 1,
        max_sugar_per_cell: int = 4,
    ) -> None:
        self.width = width
        self.height = height
        self.rng = rng
        self.sugar_regeneration_rate = sugar_regeneration_rate
        self.max_sugar_per_cell = max_sugar_per_cell

        self.sugar_map: np.ndarray = self._create_sugar_distribution(
            initial_sugar_distribution
        )
        self.agent_positions: Dict[str, Tuple[int, int]] = {}

    def _create_sugar_distribution(self, distribution_type: str) -> np.ndarray:
        """Creates the initial sugar map based on the specified distribution."""
        if distribution_type == "gaussian":
            x, y = np.meshgrid(np.arange(self.width), np.arange(self.height))

            peak1_x = self.rng.uniform(0.2, 0.3) * self.width
            peak1_y = self.rng.uniform(0.2, 0.3) * self.height
            peak2_x = self.rng.uniform(0.7, 0.8) * self.width
            peak2_y = self.rng.uniform(0.7, 0.8) * self.height

            dist1 = np.sqrt((x - peak1_x) ** 2 + (y - peak1_y) ** 2)
            dist2 = np.sqrt((x - peak2_x) ** 2 + (y - peak2_y) ** 2)
            gaussian1 = np.exp(-(dist1**2) / (2 * (self.width / 6) ** 2))
            gaussian2 = np.exp(-(dist2**2) / (2 * (self.width / 6) ** 2))
            sugar = (gaussian1 + gaussian2) * self.max_sugar_per_cell
            # CHANGED: The sugar map is now an array of floats
            return sugar.astype(float)
        else:
            # CHANGED: The sugar map is now an array of floats
            return self.rng.integers(
                0, self.max_sugar_per_cell + 1, size=(self.width, self.height)
            ).astype(float)

    def regenerate_sugar(self) -> None:
        """Increments the sugar in all cells up to the maximum."""
        # This operation now works correctly because the dtypes match.
        self.sugar_map += self.sugar_regeneration_rate
        np.clip(self.sugar_map, 0, self.max_sugar_per_cell, out=self.sugar_map)

    def get_sugar_at(self, position: Tuple[int, int]) -> int:
        """Returns the integer amount of sugar at a given position."""
        return int(self.sugar_map[position[1], position[0]])

    def consume_sugar(self, position: Tuple[int, int]) -> int:
        """Consumes the integer portion of sugar and returns the amount."""
        sugar_amount = self.get_sugar_at(position)
        # The agent harvests the integer amount, leaving the fraction.
        self.sugar_map[position[1], position[0]] -= sugar_amount
        return sugar_amount

    def get_all_empty_cells(self) -> List[Tuple[int, int]]:
        """
        Returns a list of all unoccupied cells. This is a deterministic
        method required by the loader for reproducible agent spawning.
        """
        all_cells = set((x, y) for x in range(self.width) for y in range(self.height))
        occupied_cells = set(self.agent_positions.values())
        return list(all_cells - occupied_cells)

    def get_random_empty_cell(self) -> Optional[Tuple[int, int]]:
        """Finds a single random cell not occupied by another agent."""
        empty_cells = self.get_all_empty_cells()
        if not empty_cells:
            return None
        return empty_cells[self.rng.choice(len(empty_cells))]

    def update_entity_position(
        self, entity_id: str, old_pos: Optional[Any], new_pos: Any
    ) -> None:
        self.agent_positions[entity_id] = new_pos

    def remove_entity(self, entity_id: str) -> None:
        if entity_id in self.agent_positions:
            del self.agent_positions[entity_id]

    def is_valid_position(self, position: Any) -> bool:
        return 0 <= position[0] < self.width and 0 <= position[1] < self.height

    def get_entities_at_position(self, position: Any) -> Set[str]:
        for agent_id, pos in self.agent_positions.items():
            if pos == position:
                return {agent_id}
        return set()

    def get_neighbors(self, position: Any) -> List[Any]:
        x, y = position
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if self.is_valid_position((nx, ny)):
                    neighbors.append((nx, ny))
        return neighbors

    def distance(self, pos1: Any, pos2: Any) -> float:
        return float(abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "sugar_map": self.sugar_map.tolist(),
            "agent_positions": self.agent_positions,
        }

    def restore_from_dict(self, data: Dict[str, Any]) -> None:
        self.width = data["width"]
        self.height = data["height"]
        self.sugar_map = np.array(data["sugar_map"])
        self.agent_positions = {k: tuple(v) for k, v in data["agent_positions"].items()}

    def get_valid_positions(self) -> List[Any]:
        return [(x, y) for x in range(self.width) for y in range(self.height)]

    def can_move(self, from_pos: Any, to_pos: Any) -> bool:
        return not self.get_entities_at_position(to_pos)

    def get_entities_in_radius(self, center: Any, radius: int) -> List[Tuple[str, Any]]:
        return []
