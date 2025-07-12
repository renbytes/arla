# src/agent_engine/systems/q_learning_system.py
"""
Manages the Q-learning process for all agents, updating the utility network.
"""

from typing import Any, Dict, List, Type, cast

import numpy as np
import torch

# Imports from agent_core
from agent_core.agents.actions.action_interface import ActionInterface
from agent_core.agents.actions.action_registry import action_registry
from agent_core.agents.actions.base_action import ActionOutcome
from agent_core.core.ecs.component import (
    ActionPlanComponent,
    AffectComponent,
    Component,
    EmotionComponent,
    GoalComponent,
    IdentityComponent,
    TimeBudgetComponent,
)
from agent_core.core.ecs.event_bus import EventBus
from agent_core.policy.state_encoder_interface import StateEncoderInterface

from agent_engine.simulation.simulation_state import SimulationState
from agent_engine.simulation.system import System

# Imports from agent_engine
from agent_engine.systems.components import QLearningComponent


class QLearningSystem(System):
    """
    Manages the Q-learning process for all agents.
    """

    REQUIRED_COMPONENTS: List[Type[Component]] = [
        TimeBudgetComponent,
        QLearningComponent,
        IdentityComponent,
        AffectComponent,
        GoalComponent,
        EmotionComponent,
    ]

    def __init__(
        self,
        simulation_state: SimulationState,
        config: Dict[str, Any],
        cognitive_scaffold: Any,
        state_encoder: StateEncoderInterface,
    ):
        # NOTE: The super call is now valid due to the inheritance fix.
        super().__init__(simulation_state, config, cognitive_scaffold)

        event_bus = simulation_state.event_bus
        if event_bus is None:
            raise ValueError("EventBus cannot be None for QLearningSystem.")
        self.event_bus: EventBus = event_bus

        self.event_bus.subscribe("action_executed", self.on_action_executed)
        self.previous_states: Dict[str, np.ndarray] = {}
        self.state_encoder = state_encoder

    async def update(self, current_tick: int) -> None:
        """
        Caches the current state features for each learning agent.
        """
        # NOTE: This call is now valid because self.simulation_state is concrete.
        target_entities = self.simulation_state.get_entities_with_components(self.REQUIRED_COMPONENTS)
        for entity_id, components in target_entities.items():
            time_comp = cast(TimeBudgetComponent, components.get(TimeBudgetComponent))
            if time_comp and time_comp.is_active:
                current_state_features = self.state_encoder.encode_state(self.simulation_state, entity_id, self.config)
                self.previous_states[entity_id] = current_state_features

    def on_action_executed(self, event_data: Dict[str, Any]) -> None:
        """Event handler that triggers the Q-learning update step."""
        entity_id = event_data["entity_id"]
        action_plan = cast(ActionPlanComponent, event_data["action_plan"])
        action_outcome = cast(ActionOutcome, event_data["action_outcome"])
        current_tick = event_data["current_tick"]

        q_comp = cast(
            QLearningComponent,
            self.simulation_state.get_component(entity_id, QLearningComponent),
        )
        if not q_comp or not isinstance(action_plan.action_type, ActionInterface):
            return

        old_state_features = self.previous_states.get(entity_id)
        if old_state_features is None:
            print(f"Warning: No previous state for {entity_id}. Skipping Q-update.")
            return

        target_id = action_plan.params.get("target_agent_id")

        new_state_features = self.state_encoder.encode_state(self.simulation_state, entity_id, self.config, target_id)

        action_features = action_plan.action_type.get_feature_vector(
            entity_id, self.simulation_state, action_plan.params
        )

        id_comp = self.simulation_state.get_component(entity_id, IdentityComponent)
        affect_comp = self.simulation_state.get_component(entity_id, AffectComponent)
        goal_comp = self.simulation_state.get_component(entity_id, GoalComponent)
        emotion_comp = self.simulation_state.get_component(entity_id, EmotionComponent)

        if not all([id_comp, affect_comp, goal_comp, emotion_comp]):
            print(f"Warning: Missing cognitive components for {entity_id}.")
            return

        internal_features = self.simulation_state.get_internal_state_features_for_entity(
            cast(IdentityComponent, id_comp),
            cast(AffectComponent, affect_comp),
            cast(GoalComponent, goal_comp),
            cast(EmotionComponent, emotion_comp),
        )
        possible_next_actions = self._generate_possible_action_plans(entity_id, current_tick)

        self._perform_learning_step(
            entity_id,
            q_comp,
            old_state_features,
            new_state_features,
            action_features,
            internal_features,
            action_outcome.reward,
            possible_next_actions,
            current_tick,
        )

    def _perform_learning_step(
        self,
        entity_id: str,
        q_comp: QLearningComponent,
        old_state: np.ndarray,
        new_state: np.ndarray,
        action_features: List[float],
        internal_features: np.ndarray,
        reward: float,
        possible_next_actions: List[ActionPlanComponent],
        current_tick: int,
    ) -> None:
        """Pure Q-learning logic using the Bellman equation."""
        q_config = self.config.get("learning", {}).get("q_learning", {})
        device = self.simulation_state.device

        old_state_t = torch.tensor(old_state, dtype=torch.float32).unsqueeze(0).to(device)
        new_state_t = torch.tensor(new_state, dtype=torch.float32).unsqueeze(0).to(device)
        action_t = torch.tensor(action_features, dtype=torch.float32).unsqueeze(0).to(device)
        internal_t = torch.tensor(internal_features, dtype=torch.float32).unsqueeze(0).to(device)
        reward_t = torch.tensor(reward, dtype=torch.float32).to(device)

        q_comp.optimizer.zero_grad()
        current_q = q_comp.utility_network(old_state_t, internal_t, action_t)

        max_next_q = 0.0
        if possible_next_actions:
            with torch.no_grad():
                next_action_features_list = []
                for plan in possible_next_actions:
                    if isinstance(plan.action_type, ActionInterface):
                        feature_vec = plan.action_type.get_feature_vector(entity_id, self.simulation_state, plan.params)
                        next_action_features_list.append(feature_vec)

                if next_action_features_list:
                    next_action_tensors = torch.tensor(np.array(next_action_features_list), dtype=torch.float32).to(
                        device
                    )
                    num_next = next_action_tensors.shape[0]
                    next_q_values = q_comp.utility_network(
                        new_state_t.expand(num_next, -1),
                        internal_t.expand(num_next, -1),
                        next_action_tensors,
                    )
                    if next_q_values.numel() > 0:
                        max_next_q = torch.max(next_q_values).item()

        gamma = q_config.get("gamma", 0.99)
        target_q = reward_t + gamma * max_next_q

        loss = q_comp.loss_fn(current_q.squeeze(), target_q.detach())
        loss.backward()
        q_comp.optimizer.step()

        self.event_bus.publish(
            "q_learning_update",
            {
                "entity_id": entity_id,
                "q_loss": loss.item(),
                "current_tick": current_tick,
            },
        )

    def _generate_possible_action_plans(self, entity_id: str, current_tick: int) -> List[ActionPlanComponent]:
        """Helper to generate all possible actions for an agent."""
        all_plans: List[ActionPlanComponent] = []
        for action_class in action_registry.get_all_actions():
            action_instance = action_class()
            for params in action_instance.generate_possible_params(entity_id, self.simulation_state, current_tick):
                all_plans.append(ActionPlanComponent(action_type=action_instance, params=params))
        return all_plans
