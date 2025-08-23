# src/agent_core/core/ecs/component.py

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from agent_core.agents.actions.base_action import Intent
from agent_core.core.ecs.base import CognitiveComponent

if TYPE_CHECKING:
    from dowhy import CausalModel

    from agent_core.core.schemas import (
        Belief,
        CounterfactualEpisode,
        Episode,
        RelationalSchema,
    )


# Define an interface to break the circular dependency with agent-engine.
class MultiDomainIdentityInterface(ABC):
    @abstractmethod
    def get_global_identity_embedding(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_identity_coherence(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_identity_stability(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def update_domain_identity(
        self,
        domain: Any,
        new_traits: np.ndarray,
        context: Dict[str, Any],
        current_tick: int,
    ) -> Tuple[bool, float, Dict[str, float]]:
        raise NotImplementedError

    @abstractmethod
    def get_domain_embedding(self, domain: Any) -> np.ndarray:
        raise NotImplementedError


class ComponentValidationError(Exception):
    """Raised when component validation fails."""

    def __init__(self, component_type: str, entity_id: str, errors: List[str]):
        self.component_type = component_type
        self.entity_id = entity_id
        self.errors = errors
        super().__init__(
            f"{component_type} validation failed for {entity_id}: {', '.join(errors)}"
        )


class Component(CognitiveComponent):
    """
    Base class for all components with validation interface.
    """

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Converts the component's data to a dictionary for serialization/logging."""
        raise NotImplementedError

    @abstractmethod
    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        """
        Validates component state and returns (is_valid, error_list).
        """
        raise NotImplementedError

    def auto_fix(self, entity_id: str, config: Dict[str, Any]) -> bool:
        """
        Attempts to automatically fix validation errors.
        """
        return False  # Default: no auto-fix


# -----------------------------------------------------------------------
# Core Cognitive Components (World-Agnostic)
# These components define the internal, psychological state of an agent.
# -----------------------------------------------------------------------


class MemoryComponent(Component):
    """
    Stores episodic, short-term, and causal memories. This component has been
    updated to support formal causal reasoning with the `dowhy` library.
    """

    def __init__(self) -> None:
        self.episodic_memory: List[Dict[str, Any]] = []
        self.short_term_memory: deque[Dict[str, Any]] = deque(maxlen=10)
        self.last_llm_reflection_summary: str = ""
        self.counterfactual_memories: List["CounterfactualEpisode"] = []

        # Attributes for formal causal reasoning
        self.causal_data: List[Dict[str, Any]] = []
        self.causal_model: Optional["CausalModel"] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the component's state, reflecting the new causal structure."""
        return {
            "episodic_memory_count": len(self.episodic_memory),
            "causal_data_points": len(self.causal_data),
            "has_causal_model": self.causal_model is not None,
            "counterfactual_count": len(self.counterfactual_memories),
        }

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        """Validates the new memory structures."""
        if not isinstance(self.episodic_memory, list) or not isinstance(
            self.causal_data, list
        ):
            return False, ["Memory structures have incorrect types."]
        return True, []


class IdentityComponent(Component):
    """Stores the agent's multi-domain identity."""

    def __init__(self, multi_domain_identity: "MultiDomainIdentityInterface") -> None:
        self.multi_domain_identity = multi_domain_identity
        self.embedding: np.ndarray = (
            self.multi_domain_identity.get_global_identity_embedding()
        )
        self.salient_traits_cache: Dict[str, float] = {}
        self.learned_concepts: Dict[str, np.ndarray] = {}
        self.identity_coherence_history: List[float] = []
        self.last_identity_update_tick: int = 0
        self.identity_change_resistance: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "salient_traits_cache": self.salient_traits_cache,
            "identity_coherence": self.multi_domain_identity.get_identity_coherence(),
            "identity_stability": self.get_identity_stability(),
        }

    def get_identity_stability(self) -> float:
        """Delegates the call to the underlying MultiDomainIdentity object."""
        return self.multi_domain_identity.get_identity_stability()

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        if self.embedding is None or not isinstance(self.embedding, np.ndarray):
            errors.append("embedding is None or not a numpy array")
        return len(errors) == 0, errors


class ValidationComponent(Component):
    """
    Stores confidence scores for an agent's reflections and its causal model.
    """

    def __init__(self) -> None:
        self.reflection_confidence_scores: Dict[int, float] = {}
        self.causal_model_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence_scores": self.reflection_confidence_scores,
            "causal_model_confidence": self.causal_model_confidence,
        }

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        if not isinstance(self.reflection_confidence_scores, dict):
            return False, ["reflection_confidence_scores is not a dict"]
        if not isinstance(self.causal_model_confidence, float):
            return False, ["causal_model_confidence is not a float"]
        return True, []


class GoalComponent(Component):
    """Stores agent goals."""

    def __init__(self, embedding_dim: int) -> None:
        self.current_symbolic_goal: Optional[str] = None
        self.symbolic_goals_data: Dict[str, Dict[str, Any]] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {"current_symbolic_goal": self.current_symbolic_goal}

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        if (
            self.current_symbolic_goal
            and self.current_symbolic_goal not in self.symbolic_goals_data
        ):
            return False, [
                f"Current goal '{self.current_symbolic_goal}' not in goal data."
            ]
        return True, []


class EmotionComponent(Component):
    """Stores the agent's current emotional state (valence, arousal)."""

    def __init__(
        self,
        valence: float = 0.0,
        arousal: float = 0.5,
        current_emotion_category: str = "neutral",
    ) -> None:
        self.valence: float = valence
        self.arousal: float = arousal
        self.current_emotion_category: str = current_emotion_category

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "current_emotion_category": self.current_emotion_category,
        }

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        errors = []
        if not -1.0 <= self.valence <= 1.0:
            errors.append("Valence out of bounds.")
        if not 0.0 <= self.arousal <= 1.0:
            errors.append("Arousal out of bounds.")
        return len(errors) == 0, errors


class AffectComponent(Component):
    """Stores affective state, including prediction errors and dissonance."""

    def __init__(self, affective_buffer_maxlen: int) -> None:
        self.prediction_delta_magnitude: float = 0.0
        self.predictive_delta_smooth: float = 0.5
        self.cognitive_dissonance: float = 0.0
        self.affective_experience_buffer: deque[Any] = deque(
            maxlen=affective_buffer_maxlen
        )
        self.prev_reward: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction_delta_magnitude": self.prediction_delta_magnitude,
            "cognitive_dissonance": self.cognitive_dissonance,
            "prev_reward": self.prev_reward,
        }

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        if np.isnan(self.cognitive_dissonance) or np.isinf(self.cognitive_dissonance):
            return False, ["Cognitive dissonance is not a finite number."]
        return True, []


class CompetenceComponent(Component):
    """
    Stores an objective record of an agent's actions to track skill development.
    """

    def __init__(self) -> None:
        self.action_counts: Dict[str, int] = defaultdict(int)

    def to_dict(self) -> Dict[str, Any]:
        return {"action_counts": self.action_counts}

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        if not isinstance(self.action_counts, defaultdict):
            errors.append(
                f"action_counts is not a defaultdict but a {type(self.action_counts)}"
            )
        return len(errors) == 0, errors

    def auto_fix(self, entity_id: str, config: Dict[str, Any]) -> bool:
        if not isinstance(self.action_counts, defaultdict):
            self.action_counts = defaultdict(int)
            return True
        return False


class EpisodeComponent(Component):
    """Stores narrative arcs (episodes) chunked from the agent's experiences."""

    def __init__(self) -> None:
        self.episodes: List["Episode"] = []

    def to_dict(self) -> Dict[str, Any]:
        return {"episode_count": len(self.episodes)}

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        return isinstance(self.episodes, list), []


class BeliefSystemComponent(Component):
    """Holds the agent's core beliefs and actionable rules."""

    def __init__(self) -> None:
        self.belief_base: Dict[str, "Belief"] = {}
        self.rule_base: List[str] = []
        self.social_norms: Dict[str, float] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "belief_count": len(self.belief_base),
            "rule_count": len(self.rule_base),
            "social_norms": self.social_norms,
        }

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        """Validates that all attributes have the correct type."""
        errors: List[str] = []
        if not isinstance(self.belief_base, dict):
            errors.append("'belief_base' attribute must be a dictionary.")
        if not isinstance(self.rule_base, list):
            errors.append("'rule_base' attribute must be a list.")
        if not isinstance(self.social_norms, dict):
            errors.append("'social_norms' attribute must be a dictionary.")

        # Returns True if the errors list is empty, along with the list itself.
        return len(errors) == 0, errors


class SocialMemoryComponent(Component):
    """Stores relational schemas about other agents."""

    def __init__(self, schema_embedding_dim: int, device: Any) -> None:
        self.schemas: Dict[str, "RelationalSchema"] = {}
        self.schema_embedding_dim = schema_embedding_dim
        self.device = device

    def to_dict(self) -> Dict[str, Any]:
        return {"num_schemas": len(self.schemas)}

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        return isinstance(self.schemas, dict), []


class ValueSystemComponent(Component):
    """Holds personal multipliers for subjective rewards and costs."""

    def __init__(self) -> None:
        self.combat_victory_multiplier: float = 1.0
        self.collaboration_multiplier: float = 1.0
        self.resource_yield_multiplier: float = 1.0
        self.exploration_multiplier: float = 1.0
        self.risk_tolerance: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "collaboration_multiplier": self.collaboration_multiplier,
            "risk_tolerance": self.risk_tolerance,
        }

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        return True, []


# -----------------------------------------------------------------------
# Agnostic Utility Components
# These components are necessary for the engine to function but are not
# part of the agent's core cognitive state.
# -----------------------------------------------------------------------


class PerceptionComponent(Component):
    """
    A generic container for an agent's sensory input.

    This component is world-agnostic and stores a dictionary of entities
    that the agent can currently perceive, along with their observed features.
    It is populated by a world-specific PerceptionProvider.
    """

    def __init__(self, vision_range: int) -> None:
        """
        Initializes the PerceptionComponent.

        Args:
            vision_range: The maximum distance the agent can perceive.
        """
        self.vision_range = vision_range
        self.visible_entities: Dict[str, Dict[str, Any]] = {}

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the component's data."""
        return {
            "vision_range": self.vision_range,
            "visible_entities_count": len(self.visible_entities),
        }

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        """Validates the component's data."""
        if self.vision_range < 0:
            return False, ["Vision range cannot be negative."]
        return True, []


class TimeBudgetComponent(Component):
    """
    Manages an agent's time budget (or lifespan/energy) within the simulation.
    This is a foundational utility component for the simulation engine,
    representing an agent's capacity to perform actions.
    """

    def __init__(
        self, initial_time_budget: float, lifespan_std_dev_percent: float = 0.0
    ) -> None:
        self.initial_time_budget = initial_time_budget
        self.max_time_budget = initial_time_budget * 2  # Max capacity for time/energy
        self.current_time_budget: float = initial_time_budget
        self.is_active: bool = True  # Whether the agent is currently able to act
        self.action_counts: Dict[str, int] = defaultdict(
            int
        )  # Tracks how many times each action was performed

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        if self.initial_time_budget <= 0:
            errors.append(
                f"initial_time_budget must be > 0, got {self.initial_time_budget}"
            )
        if self.current_time_budget < 0:
            errors.append(
                f"current_time_budget cannot be negative, got {self.current_time_budget}"
            )
        if self.max_time_budget <= 0:
            errors.append(f"max_time_budget must be > 0, got {self.max_time_budget}")
        if self.is_active and self.current_time_budget <= 0:
            errors.append(
                f"Entity marked active but has no time budget ({self.current_time_budget})"
            )
        if not self.is_active and self.current_time_budget > 0:
            errors.append(
                f"Entity marked inactive but has time budget ({self.current_time_budget})"
            )
        if (
            self.current_time_budget > self.max_time_budget * 1.1
        ):  # Allow slight overshoot for float precision
            errors.append(
                f"current_time_budget ({self.current_time_budget}) exceeds max ({self.max_time_budget})"
            )
        return len(errors) == 0, errors

    def auto_fix(self, entity_id: str, config: Dict[str, Any]) -> bool:
        fixed: bool = False
        if self.current_time_budget < 0:
            self.current_time_budget = 0
            self.is_active = False
            fixed = True
        if self.is_active and self.current_time_budget <= 0:
            self.is_active = False
            fixed = True
        if (
            not self.is_active
            and self.current_time_budget > 0
            and self.current_time_budget < self.initial_time_budget * 0.1
        ):
            self.current_time_budget = 0
            fixed = True
        if self.current_time_budget > self.max_time_budget:
            self.current_time_budget = self.max_time_budget
            fixed = True
        return fixed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_time_budget": self.initial_time_budget,
            "current_time_budget": self.current_time_budget,
            "is_active": self.is_active,
            "action_counts": self.action_counts,
        }


class ActionPlanComponent(Component):
    """A placeholder for the agent's chosen action for the current tick."""

    def __init__(
        self,
        action_type: Optional[Any] = None,
        intent: Optional[Intent] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.action_type = action_type
        self.intent = intent
        self.params = params if params is not None else {}

    def to_dict(self) -> Dict[str, Any]:
        return {"action_type": str(self.action_type), "intent": str(self.intent)}

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        return True, []


class ActionOutcomeComponent(Component):
    """Stores the result of the last executed action."""

    def __init__(self) -> None:
        self.success: bool = False
        self.reward: float = 0.0
        self.details: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {"success": self.success, "reward": self.reward}

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        return True, []
