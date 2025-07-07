# src/simulations/soul_sim/components.py
"""
Defines the concrete, world-specific components for the 'soul_sim' simulation.
These components represent the physical and tangible aspects of entities in this world.
"""

from collections import deque
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

# Imports from the core library, defining the base class and interfaces
from agent_core.core.ecs.component import Component

if TYPE_CHECKING:
    from agent_core.environment.interface import EnvironmentInterface


class PositionComponent(Component):
    """Stores an entity's position and movement history within a given environment."""

    def __init__(self, position: Tuple[int, int], environment: "EnvironmentInterface") -> None:
        self.position: Tuple[int, int] = position
        self.environment = environment
        self.history: deque[Tuple[int, int]] = deque(maxlen=20)
        self.history.append(position)
        self.visited_positions: set[Tuple[int, int]] = {position}

    def to_dict(self) -> Dict[str, Any]:
        return {"position": self.position, "history_len": len(self.history)}

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        if not hasattr(self.environment, "is_valid_position") or not self.environment.is_valid_position(self.position):
            errors.append(f"Position {self.position} is invalid for the current environment.")
        return len(errors) == 0, errors


class HealthComponent(Component):
    """Stores the health state of an entity."""

    def __init__(self, initial_health: float) -> None:
        self.initial_health = initial_health
        self.current_health: float = initial_health

    def to_dict(self) -> Dict[str, Any]:
        return {"current_health": self.current_health, "initial_health": self.initial_health}

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        if np.isnan(self.current_health) or np.isinf(self.current_health):
            errors.append(f"current_health has invalid value: {self.current_health}")
        elif self.current_health < 0:
            errors.append(f"current_health cannot be negative, got {self.current_health}")
        return len(errors) == 0, errors

    def auto_fix(self, entity_id: str, config: Dict[str, Any]) -> bool:
        if np.isnan(self.current_health) or np.isinf(self.current_health) or self.current_health < 0:
            self.current_health = 0.0
            return True
        return False


class InventoryComponent(Component):
    """Stores an entity's resource inventory and farming status."""

    def __init__(self, initial_resources: float) -> None:
        self.current_resources: float = initial_resources
        self.farming_mode: bool = False
        self.farm_location: Optional[Tuple[int, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"current_resources": self.current_resources, "is_farming": self.farming_mode}

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        if np.isnan(self.current_resources) or np.isinf(self.current_resources):
            errors.append(f"current_resources has invalid value: {self.current_resources}")
        elif self.current_resources < 0:
            errors.append(f"current_resources cannot be negative, got {self.current_resources}")
        return len(errors) == 0, errors


class CombatComponent(Component):
    """Stores an entity's combat-related attributes."""

    def __init__(self, attack_power: float) -> None:
        self.attack_power = attack_power

    def to_dict(self) -> Dict[str, Any]:
        return {"attack_power": self.attack_power}

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        if np.isnan(self.attack_power) or np.isinf(self.attack_power) or self.attack_power < 0:
            return False, [f"attack_power has invalid value: {self.attack_power}"]
        return True, []


class ResourceComponent(Component):
    """Stores the state of a resource node in the environment."""

    def __init__(
        self,
        resource_type: str,
        initial_health: float,
        min_agents: int,
        max_agents: int,
        mining_rate: float,
        reward_per_mine: float,
        resource_yield: float,
        respawn_time: int,
    ) -> None:
        self.type = resource_type
        self.initial_health = initial_health
        self.current_health: float = initial_health
        self.min_agents_needed = min_agents
        self.max_agents_allowed = max_agents
        self.mining_rate = mining_rate
        self.reward_per_mine_action = reward_per_mine
        self.resource_yield = resource_yield
        self.resource_respawn_time = respawn_time
        self.depleted_timer: int = 0
        self.is_depleted: bool = False
        self.mined_by_agents: List[str] = []

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "health": self.current_health, "is_depleted": self.is_depleted}

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        if self.current_health < 0:
            errors.append("current_health cannot be negative.")
        if self.is_depleted and self.current_health > 0:
            errors.append("Inconsistent state: is_depleted is True but health > 0.")
        return len(errors) == 0, errors


class NestComponent(Component):
    """Stores the locations of an agent's nests."""

    def __init__(self) -> None:
        self.locations: List[Tuple[int, int]] = []

    def to_dict(self) -> Dict[str, Any]:
        return {"nest_count": len(self.locations)}

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        if not isinstance(self.locations, list):
            return False, ["locations is not a list."]
        return True, []


class TelepathyComponent(Component):
    """A marker component enabling perfect social information access."""

    def to_dict(self) -> Dict[str, Any]:
        return {"enabled": True}

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        return True, []
