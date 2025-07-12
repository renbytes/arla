# src/agent_core/environment/interface.py
"""
Defines the abstract interface for all environments in the simulation,
allowing for different world structures (e.g., 2D Grid, Graph, etc.).
"""

from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)


# Placeholder for SpatialIndex if it's not a separate, shared component.
# If SpatialIndex is a specific implementation detail of a GridWorld,
# it should NOT be in agent_core. For now, assuming it's a general concept
# that environments might use, or it needs to be defined in the application layer.
# For the purpose of making EnvironmentInterface importable, we'll make a simple placeholder.
class SpatialIndex:
    """
    A placeholder for a spatial indexing structure.
    Concrete implementations would live in the application layer.
    """

    def __init__(self) -> None:
        pass

    def add_entity(self, entity_id: str, position: Tuple[int, int]) -> None:
        pass

    def remove_entity(self, entity_id: str) -> None:
        pass

    def update_entity_position(
        self,
        entity_id: str,
        old_pos: Optional[Tuple[int, int]],
        new_pos: Tuple[int, int],
    ) -> None:
        pass

    def get_entities_at_position(self, position: Tuple[int, int]) -> Set[str]:
        return set()

    def get_entities_in_radius(self, center: Tuple[int, int], radius: int) -> List[Tuple[str, Tuple[int, int]]]:
        return []


class EnvironmentInterface(ABC):
    """
    Abstract Base Class for all simulation environments.
    """

    @abstractmethod
    def get_valid_positions(
        self,
    ) -> List[Any]:  # Return type could be more specific, e.g., List[Tuple[int, int]]
        """Return all valid positions in this environment."""
        pass

    @abstractmethod
    def get_neighbors(self, position: Any) -> List[Any]:  # Position type can be abstract
        """Get all valid neighboring positions for a given position."""
        pass

    @abstractmethod
    def distance(self, pos1: Any, pos2: Any) -> float:  # Position type can be abstract
        """Calculate the distance between two positions."""
        pass

    @abstractmethod
    def can_move(self, from_pos: Any, to_pos: Any) -> bool:  # Position type can be abstract
        """Check if movement between two positions is valid."""
        pass

    @abstractmethod
    def is_valid_position(self, position: Any) -> bool:  # Position type can be abstract
        """Checks if a given position is valid within the environment bounds."""
        pass

    @abstractmethod
    def get_entities_at_position(self, position: Any) -> Set[str]:  # Position type can be abstract
        """Get all entities at a specific position."""
        pass

    @abstractmethod
    def get_entities_in_radius(
        self, center: Any, radius: int
    ) -> List[Tuple[str, Any]]:  # Position type can be abstract
        """Get all entities within a radius of a position."""
        pass

    @abstractmethod
    def update_entity_position(
        self, entity_id: str, old_pos: Optional[Any], new_pos: Any
    ) -> None:  # Position type can be abstract
        """Update an entity's position in any underlying spatial data structures."""
        pass

    @abstractmethod
    def remove_entity(self, entity_id: str) -> None:
        """Remove an entity from any underlying spatial data structures."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serializes the environment's state to a dictionary."""
        raise NotImplementedError

    @abstractmethod
    def restore_from_dict(self, data: Dict[str, Any]) -> None:
        """
        Restores the environment's state from a dictionary snapshot.
        This is the counterpart to to_dict().
        """
        raise NotImplementedError
