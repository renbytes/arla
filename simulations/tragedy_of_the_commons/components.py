"""
Defines the data components for the Tragedy of the Commons simulation.

These classes represent the "nouns" of our simulation—the data that defines
the state of agents (Herders) and the environment (Grass Patches). They are
intentionally simple and contain no logic.
"""

from typing import Any, Dict, List, Tuple

from agent_core.core.ecs.component import Component


class PositionComponent(Component):
    """Stores an entity's x, y coordinates in the grid world."""

    def __init__(self, x: int = 0, y: int = 0) -> None:
        self.x = x
        self.y = y

    @property
    def position(self) -> Tuple[int, int]:
        """Returns the current position as a tuple."""
        return (self.x, self.y)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the component's data to a dictionary."""
        return {"x": int(self.x), "y": int(self.y)}

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        """Validates the component's internal state."""
        errors: List[str] = []
        if not isinstance(self.x, int) or not isinstance(self.y, int):
            errors.append("Position coordinates must be integers")
        return len(errors) == 0, errors


class EnergyComponent(Component):
    """
    Stores the energy level of a Herder agent, which is essential for survival.
    Energy is consumed by metabolism and gained by grazing.
    """

    def __init__(self, current_energy: float, initial_energy: float) -> None:
        self.current_energy = current_energy
        self.initial_energy = initial_energy

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the component's data to a dictionary."""
        return {
            "current_energy": self.current_energy,
            "initial_energy": self.initial_energy,
        }

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        """Validates that energy is not negative."""
        if self.current_energy < 0:
            return False, ["Energy cannot be negative."]
        return True, []


class ResourceComponent(Component):
    """
    Represents a patch of grass, a shared and depletable resource.

    Attributes:
        current_resource (float): The current amount of grass available.
        max_resource (int): The maximum capacity of this patch.
        regeneration_rate (float): The amount of grass that regrows each tick.
        is_depleted (bool): A flag indicating if the resource is fully consumed.
    """

    def __init__(
        self,
        current_resource: float,
        max_resource: int,
        regeneration_rate: float,
    ) -> None:
        self.current_resource = float(current_resource)
        self.max_resource = max_resource
        self.regeneration_rate = regeneration_rate
        self.is_depleted = self.current_resource <= 0

    def consume(self, amount: float) -> float:
        """
        Consumes a specified amount of the resource.

        Args:
            amount: The amount of resource to consume.

        Returns:
            The actual amount of resource that was consumed, which may be less
            than requested if the resource is depleted.
        """
        consumed_amount = min(self.current_resource, amount)
        self.current_resource -= consumed_amount
        if self.current_resource <= 0:
            self.is_depleted = True
        return consumed_amount

    def regenerate(self) -> None:
        """Regenerates the resource by its regeneration rate."""
        if self.current_resource < self.max_resource:
            self.current_resource = min(
                self.max_resource, self.current_resource + self.regeneration_rate
            )
            if self.current_resource > 0:
                self.is_depleted = False

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the component's data."""
        return {
            "current_resource": self.current_resource,
            "max_resource": self.max_resource,
            "regeneration_rate": self.regeneration_rate,
            "is_depleted": self.is_depleted,
        }

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        """Validates the resource's state."""
        errors = []
        if self.current_resource < 0:
            errors.append("Resource level cannot be negative.")
        if self.max_resource <= 0:
            errors.append("Max resource must be positive.")
        return len(errors) == 0, errors
