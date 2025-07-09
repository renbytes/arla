"""
This module contains all the simulation-specific provider implementations for Soul-Sim.
These classes act as the bridge between the generic agent-engine systems and the
concrete components and game rules of this specific world.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, cast

import numpy as np
from agent_core.agents.action_generator_interface import ActionGeneratorInterface
from agent_core.agents.actions.action_interface import ActionInterface
from agent_core.agents.actions.action_registry import action_registry
from agent_core.agents.decision_selector_interface import DecisionSelectorInterface
from agent_core.cognition.narrative_context_provider_interface import NarrativeContextProviderInterface
from agent_core.core.ecs.abstractions import SimulationState
from agent_core.core.ecs.component import ActionPlanComponent, Component, TimeBudgetComponent, ValueSystemComponent
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
            action_instance = action_class()  # Instantiate the action
            # The action's own logic determines what moves are possible in the world
            possible_params = action_instance.generate_possible_params(
                entity_id, simulation_state, current_tick
            )
            for params in possible_params:
                # Here you could add more sophisticated logic to determine intent
                possible_actions.append(ActionPlanComponent(action_type=action_instance, params=params))

        return possible_actions


class SoulSimDecisionSelector(DecisionSelectorInterface):
    """
    Selects the best action for an agent to execute, typically using a Q-learning policy.
    """

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
            # If the agent isn't a Q-learner, fall back to a random choice.
            return np.random.choice(possible_actions)

        # Epsilon-greedy exploration
        if np.random.rand() < q_comp.current_epsilon:
            return np.random.choice(possible_actions)

        # --- Exploitation: Find the action with the highest Q-value ---
        best_action: Optional[ActionPlanComponent] = None
        max_q_value = -np.inf

        # The state encoder needs to be retrieved from the system manager or passed in
        # For simplicity, we assume it's accessible or re-instantiated here
        state_encoder = SoulSimStateEncoder()
        state_features = state_encoder.encode_state(simulation_state, entity_id, simulation_state.config)

        internal_features = simulation_state.get_internal_state_features_for_entity(
            simulation_state.get_component(entity_id, IdentityComponent),
            simulation_state.get_component(entity_id, AffectComponent),
            simulation_state.get_component(entity_id, GoalComponent),
            simulation_state.get_component(entity_id, EmotionComponent),
        )

        for action_plan in possible_actions:
            if not isinstance(action_plan.action_type, ActionInterface):
                continue

            action_features = action_plan.action_type.get_feature_vector(
                entity_id, simulation_state, action_plan.params
            )

            # This is a simplified representation. In a real scenario, you'd use torch tensors
            # and the utility network from the QLearningComponent.
            # q_value = q_comp.utility_network(...)
            # For this example, we'll simulate a random Q-value.
            q_value = np.random.rand()

            if q_value > max_q_value:
                max_q_value = q_value
                best_action = action_plan

        return best_action


class SoulSimRewardCalculator(RewardCalculatorInterface):
    """
    Calculates the final, subjective reward for an agent by applying multipliers
    from its internal value system to the base reward from an action's outcome.
    """

    def calculate_final_reward(
        self,
        base_reward: float,
        action_type: Any,
        action_intent: str,
        outcome_details: Dict[str, Any],
        entity_components: Dict[Type[Component], Component],
    ) -> Tuple[float, Dict[str, Any]]:
        """Applies subjective bonuses and penalties to the base reward."""
        final_reward = base_reward
        breakdown = {"base": base_reward}

        value_comp = cast(ValueSystemComponent, entity_components.get(ValueSystemComponent))
        if not value_comp:
            return final_reward, breakdown

        # Apply multiplier for collaboration
        if action_intent == "COOPERATE":
            bonus = base_reward * (value_comp.collaboration_multiplier - 1.0)
            final_reward += bonus
            breakdown["collaboration_bonus"] = bonus

        # Apply multiplier for combat victories
        if "victory" in outcome_details.get("status", ""):
             bonus = base_reward * (value_comp.combat_victory_multiplier - 1.0)
             final_reward += bonus
             breakdown["combat_bonus"] = bonus

        return final_reward, breakdown


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
        features.append(inventory_comp.resources / 100.0 if inventory_comp else 0.0) # Normalize by a reasonable max

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
        resources_norm = (inventory_comp.resources / max_res) if inventory_comp else 0.0

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
        # High health and resources imply high control over one's fate.
        health_comp = cast(HealthComponent, components.get(HealthComponent))
        inventory_comp = cast(InventoryComponent, components.get(InventoryComponent))

        health_factor = health_comp.normalized if health_comp else 0.5
        resource_factor = np.clip(inventory_comp.resources / 50.0, 0, 1) if inventory_comp else 0.5

        # If the agent has recently failed at actions, controllability is lower
        failed_states_comp = cast(FailedStatesComponent, components.get(FailedStatesComponent))
        failure_penalty = len(failed_states_comp.failed_actions) * 0.1 if failed_states_comp else 0.0

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
        # In a real scenario, you would check if pos_comp.pos is inside a "base" or "nest" area.

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
            f"I am currently at position {pos_comp.pos}. "
            f"My health is at {health_comp.current_health:.0f}, "
            f"and I have {inventory_comp.resources:.0f} resources. "
        )

        # In a real implementation, you would add summaries of recent events,
        # social interactions, and goal status.

        # The provider returns a dictionary, which will be passed to other systems
        return {
            "narrative": narrative,
            "health_level": health_comp.normalized,
            "resource_level": inventory_comp.resources,
        }
