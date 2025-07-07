# src/agent_engine/systems/affect_system.py
"""
Manages agent emotion and affect based on action outcomes.
"""

from typing import Any, Dict, List, Type, cast

import numpy as np

# Imports from agent_core
from agent_core.agents.actions.action_registry import action_registry
from agent_core.agents.actions.base_action import ActionOutcome
from agent_core.core.ecs.component import (
    AffectComponent,
    Component,
    EmotionComponent,
    GoalComponent,
)
from agent_core.core.ecs.event_bus import EventBus
from agent_core.environment.controllability_provider_interface import (
    ControllabilityProviderInterface,
)
from agent_core.environment.vitality_metrics_provider_interface import (
    VitalityMetricsProviderInterface,
)

# Imports from agent_engine
from agent_engine.cognition.emotions.affect_base import AffectiveExperience
from agent_engine.cognition.emotions.affect_learning import (
    discover_emotions,
    get_emotion_from_affect,
)
from agent_engine.cognition.emotions.model import EmotionalDynamics
from agent_engine.simulation.simulation_state import SimulationState
from agent_engine.simulation.system import System


class AffectSystem(System):
    """
    Relies on injected providers for world-specific data, improving decoupling.
    """

    REQUIRED_COMPONENTS: List[Type[Component]] = [
        AffectComponent,
        EmotionComponent,
        GoalComponent,
    ]

    def __init__(
        self,
        simulation_state: SimulationState,
        config: Dict[str, Any],
        cognitive_scaffold: Any,
        vitality_metrics_provider: VitalityMetricsProviderInterface,
        controllability_provider: ControllabilityProviderInterface,
    ):
        super().__init__(simulation_state, config, cognitive_scaffold)

        event_bus = simulation_state.event_bus
        if not event_bus:
            raise ValueError("EventBus not initialized in SimulationState")
        self.event_bus: EventBus = event_bus

        self.vitality_metrics_provider = vitality_metrics_provider
        self.controllability_provider = controllability_provider
        self.event_bus.subscribe("action_executed", self.on_action_executed)
        self.emotional_dynamics = EmotionalDynamics(config)
        self.config = config

    def on_action_executed(self, event_data: Dict[str, Any]) -> None:
        """
        Event handler that fetches dependencies and injects them into the core logic.
        """
        entity_id = event_data["entity_id"]
        components = self.simulation_state.entities.get(entity_id, {})

        if not all(comp_type in components for comp_type in self.REQUIRED_COMPONENTS):
            missing = [t.__name__ for t in self.REQUIRED_COMPONENTS if t not in components]
            print(f"WARNING: AffectSystem missing components for {entity_id}: {missing}")
            return

        self._process_affective_response(
            entity_id,
            event_data["action_outcome"],
            event_data["action_plan"],
            components,
            event_data["current_tick"],
        )

    def _process_affective_response(
        self,
        entity_id: str,
        action_outcome: ActionOutcome,
        action_plan: Any,
        components: Dict[Type[Component], Component],
        current_tick: int,
    ) -> None:
        """Pure logic for updating an agent's emotional and affective state."""
        affect_comp = cast(AffectComponent, components[AffectComponent])
        emotion_comp = cast(EmotionComponent, components[EmotionComponent])
        goal_comp = cast(GoalComponent, components[GoalComponent])

        prediction_error = action_outcome.reward - getattr(affect_comp, "prev_reward", 0.0)
        affect_comp.prev_reward = action_outcome.reward
        affect_comp.prediction_delta_magnitude = abs(prediction_error)
        affect_comp.predictive_delta_smooth = 0.8 * affect_comp.predictive_delta_smooth + 0.2 * abs(prediction_error)

        social_context = self._get_social_context(entity_id, action_plan, action_outcome)

        controllability = self.controllability_provider.get_controllability_score(
            entity_id=entity_id,
            components=components,
            simulation_state=self.simulation_state,
            current_tick=current_tick,
            config=self.config,
        )

        updated_emotion = self.emotional_dynamics.update_emotion_with_appraisal(
            emotion_comp.to_dict(),
            prediction_error,
            goal_comp.current_symbolic_goal,
            action_outcome.success,
            social_context,
            controllability,
        )
        emotion_comp.valence = updated_emotion["valence"]
        emotion_comp.arousal = updated_emotion["arousal"]

        vitality_metrics = self.vitality_metrics_provider.get_normalized_vitality_metrics(
            entity_id=entity_id, components=components, config=self.config
        )

        exp = self._create_affective_experience(
            components, action_plan, action_outcome, prediction_error, vitality_metrics
        )
        affect_comp.affective_experience_buffer.append(exp)

        min_data = self.config.get("learning", {}).get("memory", {}).get("emotion_cluster_min_data", 50)
        if len(affect_comp.affective_experience_buffer) >= min_data:
            discover_emotions(
                affect_comp=affect_comp,
                cognitive_scaffold=self.cognitive_scaffold,
                agent_id=entity_id,
                current_tick=current_tick,
                config=self.config,
            )

        emotion_comp.current_emotion_category = get_emotion_from_affect(
            **exp.to_dict(), learned_emotion_clusters=getattr(affect_comp, "learned_emotion_clusters", {})
        )

    def _create_affective_experience(
        self,
        components: Dict[Type[Component], Component],
        action_plan: Any,
        outcome: ActionOutcome,
        error: float,
        vitality_metrics: Dict[str, float],
    ) -> AffectiveExperience:
        """Helper to create an AffectiveExperience object."""
        emotion = cast(EmotionComponent, components[EmotionComponent])
        affect = cast(AffectComponent, components[AffectComponent])

        health_norm = vitality_metrics.get("health_norm", 0.5)
        time_norm = vitality_metrics.get("time_norm", 0.5)
        res_norm = vitality_metrics.get("resources_norm", 0.5)

        action_ids = action_registry.action_ids
        action_type_oh = np.zeros(len(action_ids), dtype=np.float32)
        if hasattr(action_plan.action_type, "action_id"):
            try:
                idx = action_ids.index(action_plan.action_type.action_id)
                action_type_oh[idx] = 1.0
            except ValueError:
                pass  # action_id not in registry

        return AffectiveExperience(
            valence=emotion.valence,
            arousal=emotion.arousal,
            prediction_delta_magnitude=affect.prediction_delta_magnitude,
            predictive_delta_smooth=affect.predictive_delta_smooth,
            health_norm=health_norm,
            time_norm=time_norm,
            res_norm=res_norm,
            action_type_one_hot=action_type_oh,
            outcome_reward=outcome.reward,
            prediction_error=error,
            is_positive_outcome=outcome.reward > 0,
        )

    def _get_social_context(self, entity_id: str, plan: Any, outcome: ActionOutcome) -> Dict[str, Any]:
        """Pure helper to construct the social context for appraisal."""
        # Simplified for now, but `env` is available via `self.simulation_state.environment`
        return {}

    async def update(self, current_tick: int) -> None:
        """Applies passive decay to cognitive dissonance for all relevant entities."""
        target_entities = self.simulation_state.get_entities_with_components(self.REQUIRED_COMPONENTS)
        for components in target_entities.values():
            affect_comp = cast(AffectComponent, components.get(AffectComponent))
            if affect_comp:
                affect_comp.cognitive_dissonance = max(0.0, affect_comp.cognitive_dissonance * 0.99)
