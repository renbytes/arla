"""
Defines the data components for the Sugarscape simulation.

Components are pure data containers that store the state of an entity.
They should not contain any logic. This separation of data and logic
is a core principle of the Entity-Component-System (ECS) architecture.
"""

from typing import Any, Dict, List, Tuple

from agent_core.core.ecs.component import Component


class PositionComponent(Component):
    """
    Stores an entity's x, y coordinates in the grid world.

    Args:
        x (int): The x-coordinate of the entity.
        y (int): The y-coordinate of the entity.

    Sample Usage:
        pos_comp = PositionComponent(x=10, y=20)
        current_pos = pos_comp.position  # Returns (10, 20)
    """

    def __init__(self, x: int = 0, y: int = 0) -> None:
        self.x = x
        self.y = y

    @property
    def position(self) -> Tuple[int, int]:
        """Returns the current position as a tuple."""
        return (self.x, self.y)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the component's data to a dictionary."""
        return {"x": self.x, "y": self.y}

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        """Validates the component's internal state."""
        errors: List[str] = []
        if not isinstance(self.x, int) or not isinstance(self.y, int):
            errors.append("Position coordinates must be integers")
        return len(errors) == 0, errors


class EnergyComponent(Component):
    """
    Stores the energy level of an agent. Replaces HealthComponent from
    previous simulations.

    Args:
        current_energy (float): The agent's current energy level.
        initial_energy (float): The energy level the agent started with.

    Sample Usage:
        energy_comp = EnergyComponent(current_energy=50.0, initial_energy=100.0)
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


class MetabolismComponent(Component):
    """
    Stores agent-specific metabolic traits, allowing for heterogeneity.

    Args:
        metabolic_rate (int): Energy consumed per tick.
        vision_range (int): How far the agent can see in the grid.

    Sample Usage:
        metabolism_comp = MetabolismComponent(metabolic_rate=2, vision_range=5)
    """

    def __init__(self, metabolic_rate: int, vision_range: int) -> None:
        self.metabolic_rate = metabolic_rate
        self.vision_range = vision_range

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the component's data to a dictionary."""
        return {
            "metabolic_rate": self.metabolic_rate,
            "vision_range": self.vision_range,
        }

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        """Validates metabolic traits."""
        errors = []
        if self.metabolic_rate <= 0:
            errors.append("Metabolic rate must be positive.")
        if self.vision_range < 0:
            errors.append("Vision range cannot be negative.")
        return len(errors) == 0, errors


class CommunicationComponent(Component):
    """
    Stores an agent's incoming message queue for communication.

    Args:
        message_range (int): The range within which an agent can send/receive messages.

    Sample Usage:
        comm_comp = CommunicationComponent(message_range=7)
    """

    def __init__(self, message_range: int) -> None:
        self.message_range = message_range
        self.message_queue: List[Dict[str, Any]] = []

    def add_message(self, sender_id: str, message: str, tick: int) -> None:
        """Adds a new message to the agent's inbox."""
        self.message_queue.append({"from": sender_id, "message": message, "tick": tick})

    def get_messages(self) -> List[Dict[str, Any]]:
        """Retrieves all messages and clears the queue."""
        messages = self.message_queue[:]
        self.message_queue.clear()
        return messages

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the component's data to a dictionary."""
        return {
            "message_range": self.message_range,
            "message_queue": self.message_queue,
        }

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        """Validates the message range."""
        if self.message_range < 0:
            return False, ["Message range cannot be negative."]
        return True, []
