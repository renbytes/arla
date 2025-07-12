"""
This module contains all the simulation-specific provider implementations for Soul-Sim.
These classes act as the bridge between the generic agent-engine systems and the
concrete components and game rules of this specific world.
"""

import random
from typing import Any, Dict, List, Optional, Tuple, Type, cast

import numpy as np
import torch
from agent_core.agents.action_generator_interface import ActionGeneratorInterface
from agent_core.agents.actions.action_interface import ActionInterface
from agent_core.agents.actions.action_registry import action_registry
from agent_core.agents.decision_selector_interface import DecisionSelectorInterface
from agent_core.cognition.narrative_context_provider_interface import NarrativeContextProviderInterface
from agent_core.core.ecs.abstractions import SimulationState
from agent_core.core.ecs.component import ActionPlanComponent, AffectComponent, Component, EmotionComponent, GoalComponent, IdentityComponent, TimeBudgetComponent, ValueSystemComponent
from agent_core.environment.controllability_provider_interface import ControllabilityProviderInterface
from agent_core.environment.state_node_encoder_interface import StateNodeEncoderInterface
from agent_core.environment.vitality_metrics_provider_interface import VitalityMetricsProviderInterface
from agent_core.policy.reward_calculator_interface import RewardCalculatorInterface
from agent_core.policy.state_encoder_interface import StateEncoderInterface
from agent_engine.systems.components import QLearningComponent

# Import your soul-sim specific components here
# These imports will need to be adjusted to their actual location in your project
from .components import (
    CombatComponent,
    FailedStatesComponent,
    HealthComponent,
    InventoryComponent,
    PositionComponent,
)


class SoulSimActionGenerator(ActionGeneratorInterface):
    """Generates all possible, valid actions for an agent at a given tick."""

    def generate(
        self, simulation_state: "SimulationState", entity_id: str, current_tick: int
    ) -> List["ActionPlanComponent"]:
        """
        Iterates through all registered actions and generates valid parameter combinations for each.
        """
        possible_actions: List[ActionPlanComponent] = []
        registered_action_classes = action_registry.get_all_actions()

        for action_class in registered_action_classes:
            action_instance = action_class()
            possible_params = action_instance.generate_possible_params(
                entity_id, simulation_state, current_tick
            )
            for params in possible_params:
                possible_actions.append(
                    ActionPlanComponent(
                        action_type=action_instance,
                        intent=params.get("intent"),
                        params=params
                    )
                )

        return possible_actions


class SoulSimDecisionSelector(DecisionSelectorInterface):
    """
    Selects the best action for an agent to execute using its learned Q-learning policy.
    """

    def __init__(self) -> None:
        """Initializes the decision selector with a reusable state encoder instance."""
        self.state_encoder = SoulSimStateEncoder()

    def select(
        self,
        simulation_state: "SimulationState",
        entity_id: str,
        possible_actions: List["ActionPlanComponent"],
    ) -> Optional["ActionPlanComponent"]:
        """
        Selects an action using an epsilon-greedy strategy based on the agent's utility network.
        """
        if not possible_actions:
            return None

        q_comp = simulation_state.get_component(entity_id, QLearningComponent)
        if not isinstance(q_comp, QLearningComponent):
            # Fallback to random choice for non-learning agents.
            return random.choice(possible_actions)

        # Epsilon-greedy exploration: with probability epsilon, choose a random action.
        if random.random() < q_comp.current_epsilon:
            return random.choice(possible_actions)

        # --- Exploitation: Find the action with the highest Q-value ---
        best_action: Optional[ActionPlanComponent] = None
        max_q_value = -float("inf")

        # Pre-calculate state vectors that are constant for this decision moment.
        state_features = self.state_encoder.encode_state(simulation_state, entity_id, simulation_state.config)

        internal_features = simulation_state.get_internal_state_features_for_entity(
            simulation_state.get_component(entity_id, IdentityComponent),
            simulation_state.get_component(entity_id, AffectComponent),
            simulation_state.get_component(entity_id, GoalComponent),
            simulation_state.get_component(entity_id, EmotionComponent),
        )

        # Convert to tensors once for use in the loop.
        state_tensor = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0)
        internal_tensor = torch.tensor(internal_features, dtype=torch.float32).unsqueeze(0)

        # We don't need to track gradients during inference, which is a performance optimization.
        with torch.no_grad():
            for action_plan in possible_actions:
                if not isinstance(action_plan.action_type, ActionInterface):
                    continue

                # The action feature vector is the only part that changes per iteration.
                action_features = action_plan.action_type.get_feature_vector(
                    entity_id, simulation_state, action_plan.params
                )
                action_tensor = torch.tensor(action_features, dtype=torch.float32).unsqueeze(0)

                # FIX: Call the actual utility network to get the Q-value.
                q_value = q_comp.utility_network(state_tensor, internal_tensor, action_tensor).item()

                if q_value > max_q_value:
                    max_q_value = q_value
                    best_action = action_plan

        return best_action


class SoulSimRewardCalculator(RewardCalculatorInterface):
    """Calculates final, subjective rewards for actions in the soul_sim world."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("learning", {}).get("rewards", {})

    def calculate_final_reward(
        self,
        base_reward: float,
        action_type: Any,
        action_intent: str,
        outcome_details: Dict[str, Any],
        entity_components: Dict[Type[Component], Component],
    ) -> Tuple[float, Dict[str, Any]]:
        action_id = getattr(action_type, "action_id", "unknown")

        # Start with the base reward
        current_reward = base_reward
        breakdown: Dict[str, Any] = {"base": base_reward}

        # 1. Add objective world bonuses FIRST
        bonus = 0.0
        if outcome_details.get("status") == "defeated_entity":
            bonus = self.config.get("combat_victory", 10.0)
        elif outcome_details.get("explored_new_tile", False):
            bonus = self.config.get("exploration_bonus", 0.5)

        if bonus != 0.0:
            current_reward += bonus
            breakdown["bonus"] = bonus

        # 2. Apply the agent's subjective value multiplier LAST
        # TODO: decouple this from agent-core, which should not be so coupled with world-specific details
        if isinstance(vs := entity_components.get(ValueSystemComponent), ValueSystemComponent):
            multiplier = 1.0
            if action_id == "combat":
                multiplier = vs.combat_victory_multiplier
            elif action_id == "extract":
                multiplier = vs.resource_yield_multiplier
            elif action_id == "move":
                multiplier = vs.exploration_multiplier
            elif action_id == "communicate" and action_intent == "COOPERATE":
                multiplier = vs.collaboration_multiplier

            if multiplier != 1.0:
                current_reward *= multiplier
                breakdown["value_multiplier"] = multiplier

        breakdown["final"] = current_reward

        return current_reward, breakdown


class SoulSimStateEncoder(StateEncoderInterface):
    """Encodes the simulation state into a feature vector for the Q-network."""

    def encode_state(
        self,
        simulation_state: "SimulationState",
        entity_id: str,
        config: Dict[str, Any],
        target_entity_id: Optional[str] = None,
    ) -> np.ndarray:
        """
        Creates a feature vector from soul-sim's specific components.
        """
        features = []

        # Self-related features
        health_comp = simulation_state.get_component(entity_id, HealthComponent)
        time_comp = simulation_state.get_component(entity_id, TimeBudgetComponent)
        inventory_comp = simulation_state.get_component(entity_id, InventoryComponent)

        features.append(health_comp.normalized if health_comp else 0.0)
        features.append(time_comp.current_time_budget / time_comp.max_time_budget if time_comp else 0.0)
        features.append(inventory_comp.current_resources / 100.0 if inventory_comp else 0.0) # Normalize by a reasonable max

        # Environment-related features (example)
        # In a real implementation, you would query the environment for nearby entities.
        features.extend([0.0] * 5) # Placeholder for nearby enemies
        features.extend([0.0] * 5) # Placeholder for nearby resources

        return np.array(features, dtype=np.float32)


class SoulSimVitalityMetricsProvider(VitalityMetricsProviderInterface):
    """Provides normalized (0-1) vitality scores from soul-sim components."""

    def get_normalized_vitality_metrics(
        self, entity_id: str, components: Dict[Type[Component], Component], config: Dict[str, Any]
    ) -> Dict[str, float]:

        health_comp = cast(HealthComponent, components.get(HealthComponent))
        time_comp = cast(TimeBudgetComponent, components.get(TimeBudgetComponent))
        inventory_comp = cast(InventoryComponent, components.get(InventoryComponent))

        health_norm = health_comp.normalized if health_comp else 0.0
        time_norm = (time_comp.current_time_budget / time_comp.max_time_budget) if time_comp and time_comp.max_time_budget > 0 else 0.0

        # Normalize resources based on a config value, e.g., max expected resources
        max_res = 100.0
        # Changed 'resources' to 'current_resources'
        resources_norm = (inventory_comp.current_resources / max_res) if inventory_comp else 0.0

        return {
            "health_norm": np.clip(health_norm, 0, 1),
            "time_norm": np.clip(time_norm, 0, 1),
            "resources_norm": np.clip(resources_norm, 0, 1),
        }


class SoulSimControllabilityProvider(ControllabilityProviderInterface):
    """Calculates perceived controllability based on agent's state in soul-sim."""

    def get_controllability_score(
        self, entity_id: str, components: Dict[Type[Component], Component], **kwargs
    ) -> float:
        health_comp = cast(HealthComponent, components.get(HealthComponent))
        inventory_comp = cast(InventoryComponent, components.get(InventoryComponent))
        health_factor = health_comp.normalized if health_comp else 0.5
        resource_factor = np.clip(inventory_comp.current_resources / 50.0, 0, 1) if inventory_comp else 0.5

        failed_states_comp = cast(FailedStatesComponent, components.get(FailedStatesComponent))

        failure_penalty = len(failed_states_comp.tracker) * 0.1 if failed_states_comp else 0.0

        score = (health_factor * 0.5) + (resource_factor * 0.5) - failure_penalty
        return np.clip(score, 0.0, 1.0)


class SoulSimStateNodeEncoder(StateNodeEncoderInterface):
    """Encodes the world state into a discrete tuple for the Causal Graph System."""

    def encode_state_for_causal_graph(
        self, entity_id: str, components: Dict[Type[Component], Component], **kwargs
    ) -> Tuple[Any, ...]:

        health_comp = cast(HealthComponent, components.get(HealthComponent))
        pos_comp = cast(PositionComponent, components.get(PositionComponent))

        # Discretize continuous values into categories
        if health_comp.normalized > 0.7: health_status = "high_health"
        elif health_comp.normalized > 0.3: health_status = "medium_health"
        else: health_status = "low_health"

        # Example of environmental context
        location_type = "wilderness"
        # In a real scenario, you would check if pos_comp.position is inside a "base" or "nest" area.

        return ("STATE", health_status, location_type)


class SoulSimNarrativeContextProvider(NarrativeContextProviderInterface):
    """Constructs a detailed narrative string from the agent's soul-sim state."""

    def get_narrative_context(
        self, entity_id: str, components: Dict[Type[Component], Component], **kwargs
    ) -> Dict[str, Any]:

        health_comp = cast(HealthComponent, components.get(HealthComponent))
        inventory_comp = cast(InventoryComponent, components.get(InventoryComponent))
        pos_comp = cast(PositionComponent, components.get(PositionComponent))

        narrative = (
            f"I am currently at position {pos_comp.position}. "
            f"My health is at {health_comp.current_health:.0f}, "
            f"and I have {inventory_comp.current_resources:.0f} resources. "
        )

        # In a real implementation, you would add summaries of recent events,
        # social interactions, and goal status.
        # The provider returns a dictionary, which will be passed to other systems
        return {
            "narrative": narrative,
            "resource_level": inventory_comp.current_resources,
            "health_level": health_comp.normalized,
        }
