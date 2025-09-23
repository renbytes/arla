# simulations/disease_sim/components.py
"""
Defines the data components for the Disease simulation.
"""

from typing import Any, Dict, List, Tuple
from enum import Enum
from agent_core.core.ecs.component import Component


class DiseaseStateEnum(str, Enum):
    SUSCEPTIBLE = "S"
    EXPOSED = "E"
    INFECTIOUS = "I"
    REMOVED = "R"


class DiseaseStateComponent(Component):
    """Stores the agent's current epidemiological status."""

    state: DiseaseStateEnum = DiseaseStateEnum.SUSCEPTIBLE
    incubation_timer: int = 0
    infection_timer: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "incubation_timer": self.incubation_timer,
            "infection_timer": self.infection_timer,
        }

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        return True, []


class NeighborhoodComponent(Component):
    """Stores aggregate population counts for the neighborhood-agent."""

    total_population: int = 1000
    susceptible_count: int = 1000
    exposed_count: int = 0
    infectious_count: int = 0
    removed_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_population": self.total_population,
            "susceptible_count": self.susceptible_count,
            "exposed_count": self.exposed_count,
            "infectious_count": self.infectious_count,
            "removed_count": self.removed_count,
        }

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        return True, []


class SocialNetworkComponent(Component):
    """Stores the agent's connections in the social network."""

    short_range_contacts: List[str] = []
    long_range_contacts: List[str] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "short_range_contacts": self.short_range_contacts,
            "long_range_contacts": self.long_range_contacts,
        }

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        return True, []


class DiseaseParametersComponent(Component):
    """Stores the epidemiological parameters for an agent."""

    infection_prob_i: float = 0.05  # Beta
    infection_prob_e: float = 0.01  # Theta
    incubation_period: int = 8  # Alpha
    infection_period: int = 14  # Gamma

    def to_dict(self) -> Dict[str, Any]:
        return {
            "infection_prob_i": self.infection_prob_i,
            "infection_prob_e": self.infection_prob_e,
            "incubation_period": self.incubation_period,
            "infection_period": self.infection_period,
        }

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        return True, []


# --- Standard Components for Engine Compatibility ---


class PositionComponent(Component):
    """Stores an entity's x, y coordinates."""

    def __init__(self, x: int = 0, y: int = 0) -> None:
        self.x = x
        self.y = y

    @property
    def position(self) -> Tuple[int, int]:
        return (self.x, self.y)

    def to_dict(self) -> Dict[str, Any]:
        return {"x": self.x, "y": self.y}

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        return True, []
