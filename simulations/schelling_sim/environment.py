# simulations/schelling_sim/environment.py

from typing import Any, Dict, List, Optional, Set, Tuple

from agent_core.environment.interface import EnvironmentInterface

from .components import PositionComponent


class SchellingGridEnvironment(EnvironmentInterface):
    """
    A grid-based environment for the Schelling segregation model.

    This environment implements toroidal wrapping, meaning the grid's edges
    connect to each other.

    Args:
        width (int): The width of the grid.
        height (int): The height of the grid.

    Sample Usage:
        env = SchellingGridEnvironment(width=20, height=20)
        env.add_entity("agent_1", (5, 5))
        neighbors = env.get_neighbors((5, 5))
    """

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.grid: Dict[Tuple[int, int], str] = {}
        self.agent_positions: Dict[str, Tuple[int, int]] = {}

    def initialize_from_state(
        self, simulation_state: Any, agent_components: Dict[str, Dict[Any, Any]]
    ) -> None:
        """Populates the grid based on the initial state of agent components."""
        self.grid.clear()
        self.agent_positions.clear()
        for agent_id, components in agent_components.items():
            pos_comp = components.get(PositionComponent)
            if pos_comp:
                self.add_entity(agent_id, pos_comp.position)

    def add_entity(self, entity_id: str, position: Tuple[int, int]) -> None:
        """Adds an entity to the grid at a given position."""
        self.grid[position] = entity_id
        self.agent_positions[entity_id] = position

    def move_entity(
        self, entity_id: str, from_pos: Tuple[int, int], to_pos: Tuple[int, int]
    ) -> bool:
        """Moves an entity from one position to another."""
        if self.grid.get(to_pos) is not None:
            return False  # Target cell is occupied
        if self.grid.get(from_pos) != entity_id:
            return False  # Entity not at the specified starting position

        del self.grid[from_pos]
        self.grid[to_pos] = entity_id
        self.agent_positions[entity_id] = to_pos
        return True

    def get_neighbors_of_position(
        self, position: Tuple[int, int]
    ) -> Dict[Tuple[int, int], str]:
        """
        Gets all neighboring entities for a given position with toroidal wrapping.
        """
        x, y = position
        neighbors = {}
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue

                # Apply toroidal wrapping
                nx, ny = (x + dx) % self.width, (y + dy) % self.height
                neighbor_pos = (nx, ny)

                if neighbor_pos in self.grid:
                    neighbors[neighbor_pos] = self.grid[neighbor_pos]
        return neighbors

    def get_empty_cells(self) -> List[Tuple[int, int]]:
        """Returns a list of all unoccupied cells on the grid."""
        all_cells = set((x, y) for x in range(self.width) for y in range(self.height))
        occupied_cells = set(self.grid.keys())
        return list(all_cells - occupied_cells)

    def distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculates the toroidal distance between two points."""
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])

        # Account for wrapping
        wrapped_dx = min(dx, self.width - dx)
        wrapped_dy = min(dy, self.height - dy)

        return float(wrapped_dx + wrapped_dy)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the environment's state to a dictionary."""
        return {
            "width": self.width,
            "height": self.height,
            # Convert tuple keys to string for JSON compatibility
            "grid": {str(k): v for k, v in self.grid.items()},
            "agent_positions": self.agent_positions,
        }

    def restore_from_dict(self, data: Dict[str, Any]) -> None:
        """Restores the environment's state from a dictionary."""
        self.width = data["width"]
        self.height = data["height"]
        # Convert string keys back to tuple
        self.grid = {eval(k): v for k, v in data["grid"].items()}
        self.agent_positions = data["agent_positions"]

    # --- Other required methods from the interface ---
    def get_valid_positions(self) -> List[Any]:
        return list((x, y) for x in range(self.width) for y in range(self.height))

    def get_neighbors(self, position: Any) -> List[Any]:
        return list(self.get_neighbors_of_position(position).keys())

    def can_move(self, from_pos: Any, to_pos: Any) -> bool:
        return self.grid.get(to_pos) is None

    def is_valid_position(self, position: Any) -> bool:
        return 0 <= position[0] < self.width and 0 <= position[1] < self.height

    def get_entities_at_position(self, position: Any) -> Set[str]:
        agent_id = self.grid.get(position)
        return {agent_id} if agent_id else set()

    def get_entities_in_radius(self, center: Any, radius: int) -> List[Tuple[str, Any]]:
        # Not required for this simulation, but must be implemented
        return []

    def update_entity_position(
        self, entity_id: str, old_pos: Optional[Any], new_pos: Any
    ) -> None:
        if old_pos:
            self.move_entity(entity_id, old_pos, new_pos)
        else:
            self.add_entity(entity_id, new_pos)

    def remove_entity(self, entity_id: str) -> None:
        if entity_id in self.agent_positions:
            pos = self.agent_positions.pop(entity_id)
            if pos in self.grid:
                del self.grid[pos]
