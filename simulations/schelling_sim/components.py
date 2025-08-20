# simulations/schelling_sim/components.py

from typing import Any, Dict, List, Tuple

from agent_core.core.ecs.component import Component


class PositionComponent(Component):
    """Stores an entity's x, y coordinates in the grid world."""

    def __init__(self, x: int = 0, y: int = 0) -> None:
        self.x = x
        self.y = y
        self.previous_x = x
        self.previous_y = y

    def move_to(self, new_x: int, new_y: int) -> None:
        """Update position and track previous location."""
        self.previous_x, self.previous_y = self.x, self.y
        self.x, self.y = new_x, new_y

    @property
    def position(self) -> Tuple[int, int]:
        """Returns the current position as a tuple."""
        return (self.x, self.y)

    @property
    def previous_position(self) -> Tuple[int, int]:
        """Returns the previous position as a tuple."""
        return (self.previous_x, self.previous_y)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the component's data to a dictionary."""
        return {
            "x": self.x,
            "y": self.y,
            "previous_x": self.previous_x,
            "previous_y": self.previous_y,
        }

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        """Validates the component's internal state."""
        errors: List[str] = []
        if not isinstance(self.x, int) or not isinstance(self.y, int):
            errors.append("Position coordinates must be integers")
        return len(errors) == 0, errors


class GroupComponent(Component):
    """Stores the agent's group or type identifier."""

    def __init__(self, agent_type: int) -> None:
        self.agent_type = agent_type

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the component's data to a dictionary."""
        return {"agent_type": self.agent_type}

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        """Validates the component's internal state."""
        return True, []


class SatisfactionComponent(Component):
    """Stores an agent's satisfaction state and threshold."""

    def __init__(self, satisfaction_threshold: float) -> None:
        self.satisfaction_threshold = satisfaction_threshold
        self.is_satisfied: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the component's data to a dictionary."""
        return {
            "satisfaction_threshold": self.satisfaction_threshold,
            "is_satisfied": self.is_satisfied,
        }

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        """Validates the component's internal state."""
        errors: List[str] = []
        if not 0.0 <= self.satisfaction_threshold <= 1.0:
            errors.append("satisfaction_threshold must be between 0.0 and 1.0.")
        return len(errors) == 0, errors
