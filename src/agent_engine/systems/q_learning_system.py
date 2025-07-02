from typing import Any, Dict, List, Type, cast

import numpy as np
import torch

from agent_core.agents.actions.action_interface import ActionInterface
from agent_core.agents.actions.action_registry import action_registry
from agent_core.core.ecs.component import (
    ActionPlanComponent,
    AffectComponent,
    Component,
    EmotionComponent,
    EnvironmentObservationComponent,
    GoalComponent,
    IdentityComponent,
    QLearningComponent,
    TimeBudgetComponent,
)
from agent_core.core.ecs.event_bus import EventBus
from agent_engine.policy.state_encoder import StateEncoder
from agent_engine.simulation.system import System


class QLearningSystem(System):
    """Manages the Q-learning process for all agents."""

    REQUIRED_COMPONENTS: List[Type[Component]] = [
        TimeBudgetComponent,
        QLearningComponent,
        EnvironmentObservationComponent,
    ]

    def __init__(
        self, simulation_state: Any, config: Dict[str, Any], cognitive_scaffold: Any, state_encoder: StateEncoder
    ):
        super().__init__(simulation_state, config, cognitive_scaffold)
        self.event_bus: EventBus = simulation_state.event_bus
        self.event_bus.subscribe("action_executed", self.on_action_executed)
        self.previous_states: Dict[str, np.ndarray] = {}
        self.state_encoder = state_encoder

    def update(self, current_tick: int):
        """Caches the current state features for each learning agent for the next learning step."""
        target_entities = self.simulation_state.get_entities_with_components(self.REQUIRED_COMPONENTS)

        for entity_id, components in target_entities.items():
            time_comp = cast(TimeBudgetComponent, components.get(TimeBudgetComponent))
            if not time_comp or not time_comp.is_active:
                continue

            # Use the injected state encoder. Note: target_entity_id is not needed for this caching step.
            current_state_features = self.state_encoder.encode(self.simulation_state, entity_id)
            self.previous_states[entity_id] = current_state_features

    def on_action_executed(self, event_data: Dict[str, Any]):
        """Event handler that triggers the learning step."""
        entity_id = event_data["entity_id"]
        action_plan = event_data["action_plan"]
        action_outcome = event_data["action_outcome"]
        current_tick = event_data["current_tick"]

        q_comp = self.simulation_state.get_component(entity_id, QLearningComponent)
        if not isinstance(q_comp, QLearningComponent) or not isinstance(action_plan.action_type, ActionInterface):
            return

        old_state_features = self.previous_states.get(entity_id)
        if old_state_features is None:
            return

        target_entity_id = action_plan.params.get("target_agent_id")
        new_state_features = self.state_encoder.encode(self.simulation_state, entity_id, target_entity_id)
        action_features = action_plan.action_type.get_feature_vector(self.simulation_state, action_plan.params)

        # Gather internal state features
        id_comp = cast(IdentityComponent, self.simulation_state.get_component(entity_id, IdentityComponent))
        affect_comp = cast(AffectComponent, self.simulation_state.get_component(entity_id, AffectComponent))
        goal_comp = cast(GoalComponent, self.simulation_state.get_component(entity_id, GoalComponent))
        emotion_comp = cast(EmotionComponent, self.simulation_state.get_component(entity_id, EmotionComponent))

        internal_features = self.simulation_state.get_internal_state_features_for_entity(
            id_comp, affect_comp, goal_comp, emotion_comp
        )

        # Generate possible next actions for the Bellman equation
        possible_next_actions = self._generate_possible_action_plans(entity_id, current_tick)

        # Perform the learning step
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
    ):
        """Pure Q-learning logic. Takes all necessary data as arguments and performs backpropagation."""
        q_learning_config = self.config.get("learning", {}).get("q_learning", {})
        device = self.simulation_state.device

        old_state_tensor = torch.tensor(old_state, dtype=torch.float32).unsqueeze(0).to(device)
        new_state_tensor = torch.tensor(new_state, dtype=torch.float32).unsqueeze(0).to(device)
        action_tensor = torch.tensor(action_features, dtype=torch.float32).unsqueeze(0).to(device)
        internal_tensor = torch.tensor(internal_features, dtype=torch.float32).unsqueeze(0).to(device)
        reward_tensor = torch.tensor(reward, dtype=torch.float32).to(device)

        # --- Q-Learning Update ---
        # 1. Get current Q-value
        q_comp.optimizer.zero_grad()
        current_q = q_comp.utility_network(old_state_tensor, internal_tensor, action_tensor)

        # 2. Get max Q-value for the next state
        max_next_q = 0.0
        if possible_next_actions:
            with torch.no_grad():
                next_action_features_list = [
                    plan.action_type.get_feature_vector(self.simulation_state, plan.params)
                    for plan in possible_next_actions
                    if hasattr(plan, "action_type") and plan.action_type
                ]
                if next_action_features_list:
                    next_action_tensors = torch.tensor(np.array(next_action_features_list), dtype=torch.float32).to(
                        device
                    )
                    num_next_actions = next_action_tensors.shape[0]

                    expanded_new_state = new_state_tensor.expand(num_next_actions, -1)
                    expanded_internal_state = internal_tensor.expand(num_next_actions, -1)

                    next_q_values = q_comp.utility_network(
                        expanded_new_state, expanded_internal_state, next_action_tensors
                    )
                    if next_q_values.numel() > 0:
                        max_next_q = torch.max(next_q_values).item()

        # 3. Calculate target Q-value (Bellman equation)
        gamma = q_learning_config.get("gamma", 0.99)
        target_q = reward_tensor + gamma * max_next_q

        # 4. Calculate loss and update the network
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
        """Helper to generate all possible actions for an agent in its current state."""
        all_possible_plans: List[ActionPlanComponent] = []
        for action_class in action_registry.get_all_actions():
            action_instance = action_class()
            for params in action_instance.generate_possible_params(entity_id, self.simulation_state, current_tick):
                all_possible_plans.append(ActionPlanComponent(action_type=action_instance, params=params))
        return all_possible_plans
