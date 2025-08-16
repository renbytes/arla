# simulations/emergence_sim/components.py
"""
Defines the concrete, world-specific components for the 'emergence_sim' simulation.
These components are designed to support the emergence of complex social phenomena
like language, rituals, and economies from the bottom up.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np
from agent_core.core.ecs.component import Component

if TYPE_CHECKING:
    from agent_core.environment.interface import EnvironmentInterface


class OpinionComponent(Component):
    """Stores the agent's current opinion (e.g., 'Blue' or 'Orange')."""

    def __init__(self, initial_opinion: str):
        self.opinion: str = initial_opinion

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the component's state."""
        return {"opinion": self.opinion}

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        """Validates that the opinion is a non-empty string."""
        if not isinstance(self.opinion, str) or not self.opinion:
            return False, ["'opinion' attribute must be a non-empty string."]
        return True, []


class ConceptualSpaceComponent(Component):
    """
    Represents an agent's internal semantic space based on Gärdenfors' theory.
    Concepts are stored as convex regions within multi-dimensional quality spaces,
    allowing for the grounding of symbols to perceptual properties.
    """

    def __init__(self, quality_dimensions: Dict[str, int]):
        """
        Initializes the conceptual space with predefined quality dimensions.
        Args:
            quality_dimensions: A dictionary mapping dimension names (e.g., "color")
                                to their dimensionality (e.g., 3 for RGB).
        """
        self.quality_dimensions = quality_dimensions
        # Maps a symbol (str) to a dictionary of its conceptual region's properties.
        # e.g., {"red": {"dimension": "color", "centroid": np.array([...]), "boundary": [...]}}
        self.concepts: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        """Serializes key information about the agent's learned concepts."""
        return {
            "concept_count": len(self.concepts),
            "concepts_known": list(self.concepts.keys()),
        }

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        """Validates the internal consistency of the component's data."""
        if not isinstance(self.concepts, dict):
            return False, ["'concepts' attribute must be a dictionary."]
        return True, []


class RitualComponent(Component):
    """
    Stores codified ritualistic behaviors learned by an agent. These rituals
    are action sequences that have been found to reliably reduce cognitive
    dissonance in response to specific trigger events.
    """

    def __init__(self):
        """Initializes the component with an empty dictionary for rituals."""
        # Maps a trigger event (e.g., "post_combat_loss") to a sequence of action IDs.
        self.codified_rituals: Dict[str, List[str]] = {}

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the names of the rituals the agent has learned."""
        return {"rituals_known": list(self.codified_rituals.keys())}

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        """Validates that the ritual storage is a dictionary."""
        if not isinstance(self.codified_rituals, dict):
            return False, ["'codified_rituals' attribute must be a dictionary."]
        return True, []


class DebtLedgerComponent(Component):
    """
    Stores a memory of informal, non-quantified social obligations, forming
    the basis of a "human economy" as described by David Graeber. This tracks
    who owes what to whom in a social, rather than a purely transactional, sense.
    """

    def __init__(self):
        """Initializes the component with an empty list of obligations."""
        # List of dicts, e.g.,
        # {'creditor': 'agent_A', 'debtor': 'agent_B', 'description': 'food_given', 'tick': 150}
        self.obligations: List[Dict[str, Any]] = []

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the number of outstanding obligations."""
        return {"obligation_count": len(self.obligations)}

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        """Validates that the obligations are stored in a list."""
        if not isinstance(self.obligations, list):
            return False, ["'obligations' attribute must be a list."]
        return True, []


class SocialCreditComponent(Component):
    """
    Holds a single floating-point value representing an agent's reputation or
    trustworthiness within the social group. This score is updated dynamically
    by the SocialCreditSystem based on pro-social or anti-social behaviors.
    """

    def __init__(self, initial_credit: float = 0.5):
        """
        Initializes the social credit score.
        Args:
            initial_credit: The starting social credit score, typically between 0 and 1.
        """
        self.score: float = initial_credit

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the agent's current social credit score."""
        return {"social_credit_score": self.score}

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        """Validates that the score is a finite number."""
        if not isinstance(self.score, float) or not np.isfinite(self.score):
            return False, ["'score' attribute must be a finite float."]
        return True, []


class InventoryComponent(Component):
    """
    Stores an entity's resource inventory for the emergence simulation.
    This is a minimal component focused purely on the quantity of a generic resource.
    """

    def __init__(self, initial_resources: float = 0.0):
        """Initializes the component with a starting resource amount."""
        self.current_resources: float = initial_resources

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the component's state."""
        return {
            "current_resources": self.current_resources,
        }

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        """Validates the internal consistency of the component's data."""
        errors: List[str] = []
        if np.isnan(self.current_resources) or np.isinf(self.current_resources):
            errors.append(f"current_resources has an invalid value: {self.current_resources}")
        elif self.current_resources < 0:
            errors.append(f"current_resources cannot be negative, got {self.current_resources}")
        return len(errors) == 0, errors


class PositionComponent(Component):
    """Stores an entity's position in the world and a reference to the environment."""

    def __init__(self, position: Tuple[int, int], environment: "EnvironmentInterface"):
        """
        Initializes the component.
        Args:
            position: The (x, y) coordinates of the entity.
            environment: A reference to the simulation's environment object.
        """
        self.position: Tuple[int, int] = position
        # This reference allows actions to easily query the world from the agent's perspective
        self.environment: "EnvironmentInterface" = environment

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the component's state."""
        return {
            "position_x": self.position[0],
            "position_y": self.position[1],
        }

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        """Validates that the position is a tuple of two integers."""
        if (
            not isinstance(self.position, tuple)
            or len(self.position) != 2
            or not all(isinstance(x, int) for x in self.position)
        ):
            return False, [f"Position must be a tuple of two integers, but got {self.position}"]
        return True, []


class SynergyTrackerComponent(Component):
    """Stores a record of agents who have given this agent resources."""

    def __init__(self):
        # Maps giver_id -> tick_of_gift
        self.synergy_partners: Dict[str, int] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {"synergy_partners_count": len(self.synergy_partners)}

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        return isinstance(self.synergy_partners, dict), []
