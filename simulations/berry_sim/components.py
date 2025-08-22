# simulations/berry_sim/components.py

from typing import Any, Dict, List, Tuple
from agent_core.core.ecs.component import Component


class PositionComponent(Component):
    """Stores an entity's x, y coordinates in the grid world."""

    def __init__(self, x: int = 0, y: int = 0) -> None:
        self.x = x
        self.y = y

    @property
    def position(self) -> Tuple[int, int]:
        return (self.x, self.y)

    def to_dict(self) -> Dict[str, Any]:
        return {"x": self.x, "y": self.y}

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        if not isinstance(self.x, int) or not isinstance(self.y, int):
            errors.append("Position coordinates must be integers")
        return len(errors) == 0, errors


class HealthComponent(Component):
    """Stores the health of an agent."""

    def __init__(self, current_health: float, initial_health: float) -> None:
        self.current_health = current_health
        self.initial_health = initial_health

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_health": self.current_health,
            "initial_health": self.initial_health,
        }

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        if self.current_health < 0:
            return False, ["Health cannot be negative."]
        return True, []


class BerryComponent(Component):
    """Represents a berry resource in the environment."""

    def __init__(self, berry_type: str) -> None:
        self.berry_type = berry_type  # "red", "blue", or "yellow"

    def to_dict(self) -> Dict[str, Any]:
        return {"berry_type": self.berry_type}

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        if self.berry_type not in ["red", "blue", "yellow"]:
            return False, [f"Invalid berry type: {self.berry_type}"]
        return True, []


class WaterComponent(Component):
    """A marker component for a water tile."""

    def to_dict(self) -> Dict[str, Any]:
        return {}

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        return True, []


class RockComponent(Component):
    """A marker component for a rock tile."""

    def to_dict(self) -> Dict[str, Any]:
        return {}

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        return True, []
