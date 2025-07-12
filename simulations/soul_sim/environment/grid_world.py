import math
from typing import Any, Dict, List, Optional, Set, Tuple

from agent_core.environment.interface import EnvironmentInterface


class GridWorld(EnvironmentInterface):
    """
    A concrete implementation of a 2D grid-based environment.
    Manages entity positions and provides spatial query methods.
    """

    def __init__(
        self,
        width: int,
        height: int,
        impassable_tiles: Optional[Set[Tuple[int, int]]] = None,
    ):
        self.width = width
        self.height = height
        self.impassable_tiles = impassable_tiles or set()

        # Efficient lookups for entities at a specific coordinate
        self._spatial_index: Dict[Tuple[int, int], Set[str]] = {}
        # Efficient reverse lookup for an entity's current position
        self._entity_positions: Dict[str, Tuple[int, int]] = {}

    def is_valid_position(self, position: Tuple[int, int]) -> bool:
        """Checks if a position is within bounds and not on an impassable tile."""
        x, y = position
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        if position in self.impassable_tiles:
            return False
        return True

    def get_valid_positions(self) -> List[Tuple[int, int]]:
        """Returns all valid, passable positions in the grid."""
        all_positions = [(x, y) for x in range(self.width) for y in range(self.height)]
        return [pos for pos in all_positions if self.is_valid_position(pos)]

    def get_neighbors(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Gets all valid neighboring positions (including diagonals)."""
        x, y = position
        potential_neighbors = [
            (x - 1, y - 1),
            (x, y - 1),
            (x + 1, y - 1),
            (x - 1, y),
            (x + 1, y),
            (x - 1, y + 1),
            (x, y + 1),
            (x + 1, y + 1),
        ]
        return [pos for pos in potential_neighbors if self.is_valid_position(pos)]

    def distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculates the Euclidean distance between two points."""
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def can_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        """Checks if movement is possible (i.e., target is a valid neighbor)."""
        return to_pos in self.get_neighbors(from_pos)

    def get_entities_at_position(self, position: Tuple[int, int]) -> Set[str]:
        """Gets all entity IDs at a specific position."""
        return self._spatial_index.get(position, set())

    def update_entity_position(
        self,
        entity_id: str,
        old_pos: Optional[Tuple[int, int]],
        new_pos: Tuple[int, int],
    ) -> None:
        """Updates the internal spatial index with an entity's new position."""
        if old_pos and old_pos in self._spatial_index:
            self._spatial_index[old_pos].discard(entity_id)
            if not self._spatial_index[old_pos]:
                del self._spatial_index[old_pos]

        if new_pos not in self._spatial_index:
            self._spatial_index[new_pos] = set()
        self._spatial_index[new_pos].add(entity_id)

        self._entity_positions[entity_id] = new_pos

    def remove_entity(self, entity_id: str) -> None:
        """Removes an entity from all spatial tracking."""
        position = self._entity_positions.pop(entity_id, None)
        if position and position in self._spatial_index:
            self._spatial_index[position].discard(entity_id)
            if not self._spatial_index[position]:
                del self._spatial_index[position]

    def get_entities_in_radius(self, center: Tuple[int, int], radius: int) -> List[Tuple[str, Tuple[int, int]]]:
        """Finds all entities within a given radius of a center point."""
        entities_found = []
        # Simple square bounding box check first for efficiency
        for x in range(max(0, center[0] - radius), min(self.width, center[0] + radius + 1)):
            for y in range(max(0, center[1] - radius), min(self.height, center[1] + radius + 1)):
                pos = (x, y)
                # Check actual circular distance
                if self.distance(center, pos) <= radius:
                    for entity_id in self.get_entities_at_position(pos):
                        entities_found.append((entity_id, pos))
        return entities_found

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the environment's state for snapshots."""
        return {
            "width": self.width,
            "height": self.height,
            "impassable_tiles": list(self.impassable_tiles),
            "entity_positions": {eid: list(pos) for eid, pos in self._entity_positions.items()},
        }

    def restore_from_dict(self, data: Dict[str, Any]) -> None:
        """
        Restores the environment's state from a dictionary snapshot.
        """
        self.width = data.get("width", self.width)
        self.height = data.get("height", self.height)
        self.impassable_tiles = set(tuple(tile) for tile in data.get("impassable_tiles", []))

        # Clear existing state
        self._spatial_index.clear()
        self._entity_positions.clear()

        # Repopulate from snapshot
        entity_positions = data.get("entity_positions", {})
        for entity_id, pos_list in entity_positions.items():
            pos_tuple = tuple(pos_list)
            self.update_entity_position(entity_id, None, pos_tuple)

        print(f"Restored GridWorld state with {len(self._entity_positions)} entities.")
