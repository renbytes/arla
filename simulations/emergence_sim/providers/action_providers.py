# simulations/emergence_sim/providers/action_providers.py
import random
from typing import List, Optional

import torch
from agent_core.agents.action_generator_interface import ActionGeneratorInterface
from agent_core.agents.actions.action_interface import ActionInterface
from agent_core.agents.actions.action_registry import action_registry
from agent_core.agents.decision_selector_interface import DecisionSelectorInterface
from agent_core.core.ecs.abstractions import SimulationState
from agent_core.core.ecs.component import ActionPlanComponent
from agent_core.policy.state_encoder_interface import StateEncoderInterface
from agent_engine.systems.components import QLearningComponent


class EmergenceActionGenerator(ActionGeneratorInterface):
    """Generates all possible, valid actions for an agent at a given moment."""

    def generate(
        self, simulation_state: SimulationState, entity_id: str, current_tick: int
    ) -> List[ActionPlanComponent]:
        all_plans: List[ActionPlanComponent] = []
        # Iterate through all globally registered actions
        for action_class in action_registry.get_all_actions():
            action_instance = action_class()
            # For each action, generate all valid parameter combinations
            for params in action_instance.generate_possible_params(entity_id, simulation_state, current_tick):
                all_plans.append(ActionPlanComponent(action_type=action_instance, params=params))
        return all_plans


class EmergenceDecisionSelector(DecisionSelectorInterface):
    """Selects the best action using an epsilon-greedy Q-learning policy."""

    def __init__(self, state_encoder: StateEncoderInterface):
        """Initializes the decision selector with a reusable state encoder."""
        self.state_encoder = state_encoder

    def select(
        self,
        simulation_state: SimulationState,
        entity_id: str,
        possible_actions: List[ActionPlanComponent],
    ) -> Optional[ActionPlanComponent]:
        q_comp = simulation_state.get_component(entity_id, QLearningComponent)
        if not possible_actions or not q_comp:
            return None

        # Epsilon-greedy: explore vs. exploit
        if random.random() < q_comp.current_epsilon:
            # Explore: choose a random action
            return random.choice(possible_actions)
        else:
            # Exploit: choose the best action based on the Q-network
            with torch.no_grad():
                best_action = None
                max_q_value = -float("inf")

                # Encode state vectors once before the loop
                entity_components = simulation_state.entities.get(entity_id, {})
                state_features = self.state_encoder.encode_state(simulation_state, entity_id, simulation_state.config)
                internal_features = self.state_encoder.encode_internal_state(entity_components, simulation_state.config)

                state_tensor = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0)
                internal_tensor = torch.tensor(internal_features, dtype=torch.float32).unsqueeze(0)

                for plan in possible_actions:
                    if not isinstance(plan.action_type, ActionInterface):
                        continue

                    # Get action-specific feature vector
                    action_features = plan.action_type.get_feature_vector(entity_id, simulation_state, plan.params)
                    action_tensor = torch.tensor(action_features, dtype=torch.float32).unsqueeze(0)

                    q_value = q_comp.utility_network(state_tensor, internal_tensor, action_tensor).item()

                    if q_value > max_q_value:
                        max_q_value = q_value
                        best_action = plan

                return best_action
