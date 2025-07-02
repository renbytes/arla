# src/systems/affect_system.py

from typing import Any, Dict, List, Tuple, Type, cast

import numpy as np

from src.agents.actions.action_registry import action_registry
from src.agents.actions.base_action import ActionOutcome
from src.cognition.emotions.affect_base import AffectiveExperience
from src.cognition.emotions.affect_learning import discover_emotions, get_emotion_from_affect
from src.cognition.emotions.model import EmotionalDynamics
from src.core.ecs.abstractions import CognitiveComponent
from src.core.ecs.component import (
    AffectComponent,
    Component,  # Import Component for casting
    EmotionComponent,
    FailedStatesComponent,
    GoalComponent,
    HealthComponent,
    InventoryComponent,
    PositionComponent,
    TimeBudgetComponent,
)
from src.core.ecs.event_bus import EventBus
from src.core.ecs.system import System
from src.environment.interface import EnvironmentInterface
from src.utils.config_utils import get_config_value
from src.utils.math_utils import safe_divide


class AffectSystem(System):
    """
    Manages agent emotion and affect. Refactored to use dependency injection.
    """

    # The per-tick update loop requires entities with an AffectComponent.
    REQUIRED_COMPONENTS: List[Type[CognitiveComponent]] = [AffectComponent]

    def __init__(self, simulation_state: Any, config: Dict[str, Any], cognitive_scaffold: Any):
        super().__init__(simulation_state, config, cognitive_scaffold)
        self.event_bus: EventBus = simulation_state.event_bus
        self.event_bus.subscribe("action_executed", self.on_action_executed)
        self.llm_client = simulation_state.llm_client
        self.emotional_dynamics = EmotionalDynamics(config)
        self.config = config

    def on_action_executed(self, event_data: Dict[str, Any]) -> None:
        """
        Event handler that fetches all dependencies from the simulation state and
        injects them into the core affect logic function.
        """
        entity_id = event_data["entity_id"]
        current_tick = event_data["current_tick"]
        entities = self.simulation_state.entities
        environment = self.simulation_state.environment

        if not environment:
            return

        components = entities.get(entity_id, {})
        affect_comp = components.get(AffectComponent)
        if not isinstance(affect_comp, AffectComponent):
            return
        emotion_comp = components.get(EmotionComponent)
        if not isinstance(emotion_comp, EmotionComponent):
            return
        goal_comp = components.get(GoalComponent)
        if not isinstance(goal_comp, GoalComponent):
            return
        health_comp = components.get(HealthComponent)
        if not isinstance(health_comp, HealthComponent):
            return
        time_comp = components.get(TimeBudgetComponent)
        if not isinstance(time_comp, TimeBudgetComponent):
            return
        inv_comp = components.get(InventoryComponent)
        if not isinstance(inv_comp, InventoryComponent):
            return
        failed_states_comp = components.get(FailedStatesComponent)
        if not isinstance(failed_states_comp, FailedStatesComponent):
            return
        pos_comp = components.get(PositionComponent)
        if not isinstance(pos_comp, PositionComponent):
            return

        self._process_affective_response(
            entity_id,
            event_data["action_outcome"],
            event_data["action_plan"],
            affect_comp,
            emotion_comp,
            goal_comp,
            health_comp,
            time_comp,
            inv_comp,
            failed_states_comp,
            pos_comp,
            entities,
            environment,
            current_tick=current_tick,
        )

    def _process_affective_response(
        self,
        entity_id: str,
        action_outcome: ActionOutcome,
        action_plan: Any,
        affect_comp: AffectComponent,
        emotion_comp: EmotionComponent,
        goal_comp: GoalComponent,
        health_comp: HealthComponent,
        time_comp: TimeBudgetComponent,
        inv_comp: InventoryComponent,
        failed_states_comp: FailedStatesComponent,
        pos_comp: PositionComponent,
        entities: Dict[str, Dict[Type[Component], Component]],  # Explicitly type entities dict
        environment: EnvironmentInterface,
        current_tick: int,
    ) -> None:
        """
        Pure logic for updating an agent's emotional and affective state.
        """
        learning_memory_config = self.config.get("learning", {}).get("memory", {})
        min_data = learning_memory_config.get("emotion_cluster_min_data", 50)

        enabled_globally = get_config_value(
            self.config, "agent.cognitive.architecture_flags.enable_emotion_clustering", default=True
        )

        prediction_error, affect_comp.prev_reward = self._calculate_prediction_error(affect_comp, action_outcome.reward)
        affect_comp.prediction_delta_magnitude = abs(prediction_error)
        affect_comp.predictive_delta_smooth = 0.8 * affect_comp.predictive_delta_smooth + 0.2 * abs(prediction_error)

        dissonance_increase = abs(prediction_error) * 0.1 + (0.2 if not action_outcome.success else 0)
        affect_comp.cognitive_dissonance += dissonance_increase
        affect_comp.dissonance_history.append(affect_comp.cognitive_dissonance)

        social_context = self._get_social_context(entity_id, action_plan, action_outcome, entities, environment)
        failed_count = failed_states_comp.tracker.get(pos_comp.position, 0) if failed_states_comp else 0
        controllability_estimate = max(0.1, 1.0 - failed_count * 0.1)
        current_goal = goal_comp.current_symbolic_goal if goal_comp else None

        updated_emotion = self.emotional_dynamics.update_emotion_with_appraisal(
            emotion_comp.to_dict(),
            prediction_error,
            current_goal or "no_goal",
            action_outcome.success,
            social_context,
            controllability_estimate,
        )
        emotion_comp.valence = updated_emotion["valence"]
        emotion_comp.arousal = updated_emotion["arousal"]
        emotion_comp.last_appraisal = updated_emotion.get("appraisal_dimensions")

        affective_exp = self._create_affective_experience(
            emotion_comp, affect_comp, health_comp, time_comp, inv_comp, action_plan, action_outcome, prediction_error
        )
        affect_comp.affective_experience_buffer.append(affective_exp)

        if len(affect_comp.affective_experience_buffer) >= min_data and enabled_globally:
            discover_emotions(
                affect_comp=affect_comp,
                cognitive_scaffold=self.cognitive_scaffold,
                agent_id=entity_id,
                current_tick=current_tick,
                config=self.config,
            )

        emotion_comp.current_emotion_category = get_emotion_from_affect(
            **affective_exp.to_dict(), learned_emotion_clusters=affect_comp.learned_emotion_clusters
        )

    def _calculate_prediction_error(self, affect_comp: AffectComponent, current_reward: float) -> Tuple[float, float]:
        """Calculates prediction error and returns the new previous_reward."""
        prediction_error = current_reward - affect_comp.prev_reward
        return prediction_error, current_reward

    def _get_social_context(
        self,
        entity_id: str,
        action_plan: Any,
        action_outcome: ActionOutcome,
        entities: Dict[str, Dict[Type[Component], Component]],
        environment: EnvironmentInterface,
    ) -> Dict[str, Any]:
        """Pure helper to construct the social context for appraisal."""
        action_id = action_plan.action_type.action_id if hasattr(action_plan.action_type, "action_id") else "unknown"
        intent_name = action_plan.intent.name if hasattr(action_plan, "intent") and action_plan.intent else "SOLITARY"
        return {
            "other_agents_present": self._check_other_agents_nearby(entity_id, entities, environment),
            "action_intent": intent_name,
            "action_type": action_id,
            "target_agent_id": action_outcome.details.get("target_agent_id"),
            "collaboration_level": action_outcome.details.get("num_collaborators_on_hit", 0),
        }

    def _create_affective_experience(
        self,
        emotion_comp: EmotionComponent,
        affect_comp: AffectComponent,
        health_comp: HealthComponent,
        time_comp: TimeBudgetComponent,
        inv_comp: InventoryComponent,
        action_plan: Any,
        action_outcome: ActionOutcome,
        prediction_error: float,
    ) -> AffectiveExperience:
        """Pure helper to create an AffectiveExperience object."""
        health_norm = safe_divide(health_comp.current_health, health_comp.initial_health)
        time_norm = safe_divide(time_comp.current_time_budget, time_comp.initial_time_budget)

        start_inventory = get_config_value(
            self.config, "agent.foundational.vitals.initial_resources", default=100.0
        )  # Added default value for initial_resources
        res_norm = safe_divide(inv_comp.current_resources, start_inventory * 2.0)

        action_ids = action_registry.action_ids
        action_type_oh = np.zeros(len(action_ids), dtype=np.float32)
        try:
            if hasattr(action_plan.action_type, "action_id"):
                action_index = action_ids.index(action_plan.action_type.action_id)
                action_type_oh[action_index] = 1.0
        except (ValueError, AttributeError):
            pass

        return AffectiveExperience(
            emotion_comp.valence,
            emotion_comp.arousal,
            affect_comp.prediction_delta_magnitude,
            affect_comp.predictive_delta_smooth,
            health_norm,
            time_norm,
            res_norm,
            action_type_oh,
            action_outcome.reward,
            prediction_error,
            action_outcome.reward > 0,
        )

    def _check_other_agents_nearby(
        self, entity_id: str, entities: Dict[str, Dict[Type[Component], Component]], environment: EnvironmentInterface
    ) -> bool:
        """Pure helper to check for nearby agents."""
        pos_comp = entities.get(entity_id, {}).get(PositionComponent)
        if not isinstance(pos_comp, PositionComponent):
            return False

        my_pos = pos_comp.position
        for other_id, other_comps in entities.items():
            if other_id == entity_id:
                continue

            other_pos_comp = other_comps.get(PositionComponent)
            other_time_comp = other_comps.get(TimeBudgetComponent)
            if (
                isinstance(other_pos_comp, PositionComponent)
                and isinstance(other_time_comp, TimeBudgetComponent)
                and other_time_comp.is_active
            ):
                if environment.distance(my_pos, other_pos_comp.position) <= 2:
                    return True
        return False

    def update(self, current_tick: int) -> None:
        """
        Applies passive decay to cognitive dissonance for all relevant entities.
        The method now fetches the entities it needs directly from the simulation state.
        """
        # Cast to List[Type[Component]] to match the expected type for get_entities_with_components
        target_entities = self.simulation_state.get_entities_with_components(
            cast(List[Type[Component]], self.REQUIRED_COMPONENTS)
        )

        for components in target_entities.values():
            # The component is guaranteed to be here because of the filter above.
            affect_comp = cast(AffectComponent, components.get(AffectComponent))
            if affect_comp:
                affect_comp.cognitive_dissonance = max(0.0, affect_comp.cognitive_dissonance * 0.99)
