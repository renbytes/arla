"""
Defines the environment for the Tragedy of the Commons simulation.

This class manages the 2D grid, the grass resources, and provides an
interface for systems to query and interact with the world state.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from agent_core.environment.interface import EnvironmentInterface

from .components import ResourceComponent


class CommonsEnvironment(EnvironmentInterface):
    """
    Manages the grid, grass resources, and agent positions for the simulation.
    """

    def __init__(self, width: int, height: int, rng: np.random.Generator):
        self.width = width
        self.height = height
        self.rng = rng
        self.agent_positions: Dict[str, Tuple[int, int]] = {}
        # A grid to hold the entity IDs of the grass patches
        self.resource_grid: Dict[Tuple[int, int], str] = {}

    def get_resource_at(self, position: Tuple[int, int]) -> float:
        """Returns the amount of grass at a given position."""
        resource_id = self.resource_grid.get(position)
        if resource_id and "simulation_state" in self.__dict__:
            res_comp = self.simulation_state.get_component(
                resource_id, ResourceComponent
            )
            if res_comp:
                return res_comp.current_resource
        return 0.0

    def consume_resource(self, position: Tuple[int, int], amount: float) -> float:
        """Consumes grass from a patch and returns the amount consumed."""
        resource_id = self.resource_grid.get(position)
        if resource_id and "simulation_state" in self.__dict__:
            res_comp = self.simulation_state.get_component(
                resource_id, ResourceComponent
            )
            if res_comp:
                return res_comp.consume(amount)
        return 0.0

    def get_all_empty_cells(self) -> List[Tuple[int, int]]:
        """Returns a list of all cells not occupied by an agent."""
        all_cells = set((x, y) for x in range(self.width) for y in range(self.height))
        occupied_cells = set(self.agent_positions.values())
        return list(all_cells - occupied_cells)

    def is_occupied_by_agent(self, position: Tuple[int, int]) -> bool:
        """Checks if a cell is occupied by an agent."""
        return position in self.agent_positions.values()

    # --- Interface Methods ---

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
            "agent_positions": self.agent_positions,
        }

    def restore_from_dict(self, data: Dict[str, Any]) -> None:
        self.width = data["width"]
        self.height = data["height"]
        self.agent_positions = {k: tuple(v) for k, v in data["agent_positions"].items()}

    def get_valid_positions(self) -> List[Any]:
        return [(x, y) for x in range(self.width) for y in range(self.height)]

    def can_move(self, from_pos: Any, to_pos: Any) -> bool:
        return not self.is_occupied_by_agent(to_pos)

    def get_entities_in_radius(self, center: Any, radius: int) -> List[Tuple[str, Any]]:
        return []
