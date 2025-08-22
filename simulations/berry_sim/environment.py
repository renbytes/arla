# simulations/berry_sim/environment.py

import random
from typing import Any, Dict, List, Optional, Set, Tuple

from agent_core.environment.interface import EnvironmentInterface


class BerryWorldEnvironment(EnvironmentInterface):
    """
    Manages the grid, berry spawning, and toxicity rules for the experiment.
    """

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.water_locations: Set[Tuple[int, int]] = set()
        self.rock_locations: Set[Tuple[int, int]] = set()
        self.berry_locations: Dict[Tuple[int, int], str] = {}
        self.agent_positions: Dict[str, Tuple[int, int]] = {}
        self._grid_entities: Dict[Tuple[int, int], str] = {}

    def is_occupied(self, position: Tuple[int, int]) -> bool:
        """Check if a cell is occupied by a blocking object (agent, rock, water)."""
        return (
            position in self._grid_entities
            or position in self.water_locations
            or position in self.rock_locations
        )

    def get_random_empty_cell(self) -> Optional[Tuple[int, int]]:
        """Finds a random unoccupied cell."""
        for _ in range(self.width * self.height):  # Avoid infinite loops
            pos = (
                random.randint(0, self.width - 1),
                random.randint(0, self.height - 1),
            )
            if not self.is_occupied(pos) and pos not in self.berry_locations:
                return pos
        return None

    def get_berry_toxicity(
        self, berry_type: str, position: Tuple[int, int], tick: int
    ) -> float:
        """Determines the health effect of eating a berry based on experiment rules."""
        if berry_type == "red":
            return 10.0
        elif berry_type == "blue":
            return (
                -20.0
                if self.is_near_feature(position, self.water_locations, 2)
                else 10.0
            )
        elif berry_type == "yellow":
            # Toxicity is random, but seeded by position and tick for reproducibility
            seed = hash((position, tick // 100))  # Stable toxicity for a period
            rng = random.Random(seed)
            return -20.0 if rng.random() < 0.5 else 10.0
        return 0.0

    def is_near_feature(
        self, pos: Tuple[int, int], features: Set[Tuple[int, int]], distance: int
    ) -> bool:
        """Checks if a position is within a certain Manhattan distance of any feature."""
        for feature_pos in features:
            dist = abs(pos[0] - feature_pos[0]) + abs(pos[1] - feature_pos[1])
            if dist <= distance:
                return True
        return False

    def get_environmental_context(self, position: Tuple[int, int]) -> Dict[str, bool]:
        """Provides the context used by the CausalGraphSystem's StateNodeEncoder."""
        return {
            "near_water": self.is_near_feature(position, self.water_locations, 2),
            "near_rocks": self.is_near_feature(position, self.rock_locations, 2),
        }

    # --- Interface Methods ---
    def add_entity(self, entity_id: str, position: Tuple[int, int]):
        self._grid_entities[position] = entity_id
        self.agent_positions[entity_id] = position

    def update_entity_position(
        self, entity_id: str, old_pos: Optional[Any], new_pos: Any
    ):
        if old_pos and old_pos in self._grid_entities:
            del self._grid_entities[old_pos]
        self._grid_entities[new_pos] = entity_id
        self.agent_positions[entity_id] = new_pos

    def remove_entity(self, entity_id: str):
        if entity_id in self.agent_positions:
            pos = self.agent_positions.pop(entity_id)
            if pos in self._grid_entities:
                del self._grid_entities[pos]

    def get_entities_at_position(self, position: Any) -> Set[str]:
        entity_id = self._grid_entities.get(position)
        return {entity_id} if entity_id else set()

    def get_valid_positions(self) -> List[Any]:
        return [(x, y) for x in range(self.width) for y in range(self.height)]

    def get_neighbors(self, position: Any) -> List[Any]:
        x, y = position
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    neighbors.append((nx, ny))
        return neighbors

    def distance(self, pos1: Any, pos2: Any) -> float:
        return float(abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]))

    def can_move(self, from_pos: Any, to_pos: Any) -> bool:
        return not self.is_occupied(to_pos)

    def is_valid_position(self, position: Any) -> bool:
        return 0 <= position[0] < self.width and 0 <= position[1] < self.height

    def get_entities_in_radius(self, center: Any, radius: int) -> List[Tuple[str, Any]]:
        return []  # Not needed for this simulation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "water_locations": [list(pos) for pos in self.water_locations],
            "rock_locations": [list(pos) for pos in self.rock_locations],
        }

    def restore_from_dict(self, data: Dict[str, Any]) -> None:
        self.width = data["width"]
        self.height = data["height"]
        self.water_locations = {tuple(pos) for pos in data["water_locations"]}
        self.rock_locations = {tuple(pos) for pos in data["rock_locations"]}
