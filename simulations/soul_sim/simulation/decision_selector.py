import random
from typing import Any, List, Optional

import torch
from agent_core.agents.decision_selector_interface import DecisionSelectorInterface
from agent_core.core.ecs.component import (
    ActionPlanComponent,
    AffectComponent,
    EmotionComponent,
    GoalComponent,
    IdentityComponent,
)
from agent_engine.systems.components import QLearningComponent
from .state_encoder import SoulSimStateEncoder

class SoulSimDecisionSelector(DecisionSelectorInterface):
    """
    Selects an agent's action using a Q-learning policy.

    This selector uses an epsilon-greedy strategy. With a probability of
    epsilon, it **explores** by choosing a random action. Otherwise, it **exploits**
    its current knowledge by selecting the action with the highest predicted
    utility (Q-value) from its neural network.
    """

    def __init__(self) -> None:
        """Initializes the decision selector with a reusable state encoder."""
        # The state encoder is instantiated once for efficiency.
        self.state_encoder = SoulSimStateEncoder()

    def select(
        self,
        simulation_state: Any,
        entity_id: str,
        possible_actions: List[ActionPlanComponent],
    ) -> Optional[ActionPlanComponent]:
        """
        Evaluates possible actions and selects one based on the Q-learning policy.

        Args:
            simulation_state: The current state of the simulation.
            entity_id: The ID of the agent making the decision.
            possible_actions: A list of valid ActionPlanComponent objects.

        Returns:
            The chosen ActionPlanComponent, or None if no action is possible.
        """
        # --- 1. Pre-computation and Epsilon-Greedy Check ---
        if not possible_actions:
            return None

        q_comp = simulation_state.get_component(entity_id, QLearningComponent)

        # Fallback to random choice for non-learning agents or for exploration
        if not isinstance(q_comp, QLearningComponent) or random.random() < q_comp.current_epsilon:
            return random.choice(possible_actions)

        # --- 2. Feature Vector Calculation (Optimized) ---
        # The agent's internal state and the world state are constant during this
        # single decision moment. We calculate these feature vectors once before the loop.
        state_features = self.state_encoder.encode_state(
            simulation_state, entity_id, simulation_state.config
        )

        # Gather all cognitive components needed for the internal state vector
        id_comp, aff_comp, goal_comp, emo_comp = (
            simulation_state.get_component(entity_id, c)
            for c in [IdentityComponent, AffectComponent, GoalComponent, EmotionComponent]
        )
        internal_features = simulation_state.get_internal_state_features_for_entity(
            id_comp, aff_comp, goal_comp, emo_comp
        )

        # Convert to tensors once for use in the loop
        state_t = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0)
        internal_t = torch.tensor(internal_features, dtype=torch.float32).unsqueeze(0)

        # --- 3. Exploitation: Evaluate Actions and Find the Best One ---
        best_action: Optional[ActionPlanComponent] = None
        max_utility = -float("inf")

        with torch.no_grad():
            for action_plan in possible_actions:
                if not (action_plan.action_type and hasattr(action_plan.action_type, "get_feature_vector")):
                    continue

                # The action feature vector is the only one that changes per iteration
                action_features = action_plan.action_type.get_feature_vector(
                    entity_id, simulation_state, action_plan.params
                )
                action_t = torch.tensor(action_features, dtype=torch.float32).unsqueeze(0)

                # Predict utility using the agent's neural network
                utility = q_comp.utility_network(state_t, internal_t, action_t).item()

                if utility > max_utility:
                    max_utility, best_action = utility, action_plan

        return best_action
