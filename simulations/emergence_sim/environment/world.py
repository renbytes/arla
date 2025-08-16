# simulations/emergence_sim/environment/world.py

import random
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from agent_core.environment.interface import EnvironmentInterface


class EmergenceEnvironment(EnvironmentInterface):
    """A simple grid-based environment for the emergence simulation."""

    def __init__(self, width: int, height: int, num_objects: int):
        self.width = width
        self.height = height
        self.entity_positions: Dict[str, Tuple[int, int]] = {}
        # Stores objects for the "naming game"
        self.objects: Dict[str, Dict[str, Any]] = {}
        self._initialize_objects(num_objects)

    def _initialize_objects(self, num_objects: int):
        """Creates random objects with properties for agents to perceive."""
        if num_objects == 0:
            return
        shapes = ["sphere", "cube", "pyramid"]
        colors = ["red", "green", "blue"]
        for i in range(num_objects):
            obj_id = f"obj_{i}"
            pos = (
                random.randint(0, self.width - 1),
                random.randint(0, self.height - 1),
            )
            self.objects[obj_id] = {
                "id": obj_id,
                "position": pos,
                "shape": random.choice(shapes),
                "color": random.choice(colors),
            }

    def get_object_at(self, position: Tuple[int, int]) -> Optional[Dict[str, Any]]:
        """Finds an object at a specific coordinate."""
        for _obj_id, obj_data in self.objects.items():
            if obj_data["position"] == position:
                return obj_data
        return None

    def get_objects_in_radius(self, center: Tuple[int, int], radius: int) -> List[Tuple[str, Dict[str, Any]]]:
        """Gets all special objects within a given radius."""
        nearby_objects = []
        for obj_id, obj_data in self.objects.items():
            if self.distance(center, obj_data["position"]) <= radius:
                nearby_objects.append((obj_id, obj_data))
        return nearby_objects

    def get_valid_positions(self) -> List[Tuple[int, int]]:
        return [(x, y) for x in range(self.width) for y in range(self.height)]

    def is_valid_position(self, position: Tuple[int, int]) -> bool:
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height

    def distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def update_entity_position(
        self,
        entity_id: str,
        old_pos: Optional[Tuple[int, int]],
        new_pos: Tuple[int, int],
    ):
        self.entity_positions[entity_id] = new_pos

    def get_entities_in_radius(self, center: Tuple[int, int], radius: int) -> List[Tuple[str, Tuple[int, int]]]:
        nearby = []
        for eid, pos in self.entity_positions.items():
            if self.distance(center, pos) <= radius:
                nearby.append((eid, pos))
        return nearby

    def remove_entity(self, entity_id: str) -> None:
        if entity_id in self.entity_positions:
            del self.entity_positions[entity_id]

    def get_neighbors(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Gets all valid neighboring positions (cardinal directions)."""
        x, y = position
        potential_neighbors = [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        ]
        return [pos for pos in potential_neighbors if self.is_valid_position(pos)]

    def can_move(self, from_pos: Any, to_pos: Any) -> bool:
        # Placeholder
        return True

    def get_entities_at_position(self, position: Tuple[int, int]) -> Set[str]:
        """Gets all entities at a specific position."""
        return {eid for eid, pos in self.entity_positions.items() if pos == position}

    def to_dict(self) -> Dict[str, Any]:
        return {"entity_positions": self.entity_positions, "objects": self.objects}

    def restore_from_dict(self, data: Dict[str, Any]) -> None:
        self.entity_positions = data.get("entity_positions", {})
        self.objects = data.get("objects", {})
