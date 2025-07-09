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
from .state_encoder import SoulSimStateEncoder  # Import from local module


class SoulSimDecisionSelector(DecisionSelectorInterface):
    """Selects an action for an agent using a Q-learning policy."""

    def __init__(self):
        # Improvement: Instantiate the encoder once, not in a loop.
        self.state_encoder = SoulSimStateEncoder()

    def select(
        self, simulation_state: Any, entity_id: str, possible_actions: List[ActionPlanComponent]
    ) -> Optional[ActionPlanComponent]:
        q_comp = simulation_state.get_component(entity_id, QLearningComponent)
        if not possible_actions:
            return None
        if not isinstance(q_comp, QLearningComponent) or random.random() < q_comp.current_epsilon:
            return random.choice(possible_actions)

        best_action, max_utility = None, -float("inf")
        with torch.no_grad():
            for action_plan in possible_actions:
                if not (action_plan.action_type and hasattr(action_plan.action_type, "get_feature_vector")):
                    continue

                state_features = self.state_encoder.encode_state(
                    simulation_state, entity_id, simulation_state.config, action_plan.params.get("target_agent_id")
                )
                action_features = action_plan.action_type.get_feature_vector(
                    entity_id, simulation_state, action_plan.params
                )

                id_comp, aff_comp, goal_comp, emo_comp = (
                    simulation_state.get_component(entity_id, c)
                    for c in [IdentityComponent, AffectComponent, GoalComponent, EmotionComponent]
                )
                internal_features = simulation_state.get_internal_state_features_for_entity(
                    id_comp, aff_comp, goal_comp, emo_comp
                )

                state_t = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0)
                action_t = torch.tensor(action_features, dtype=torch.float32).unsqueeze(0)
                internal_t = torch.tensor(internal_features, dtype=torch.float32).unsqueeze(0)

                utility = q_comp.utility_network(state_t, internal_t, action_t).item()
                if utility > max_utility:
                    max_utility, best_action = utility, action_plan
        return best_action
