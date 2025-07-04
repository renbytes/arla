# src/agent_engine/systems/components.py

from typing import Any, Dict, List, Tuple
import torch

from agent_core.core.ecs.component import Component
from agent_engine.policy.learned_utility import UtilityNetwork


class QLearningComponent(Component):
    """
    Data container for the QLearningSystem, holding the agent's specific
    utility network, optimizer, and learning-related state.
    """

    def __init__(
        self,
        state_feature_dim: int,
        internal_state_dim: int,
        action_feature_dim: int,
        q_learning_alpha: float,
        device: torch.device,
    ) -> None:
        self.utility_network = UtilityNetwork(state_feature_dim, internal_state_dim, action_feature_dim).to(device)
        self.optimizer = torch.optim.Adam(self.utility_network.parameters(), lr=q_learning_alpha)
        self.loss_fn = torch.nn.MSELoss()
        self.current_epsilon: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"current_epsilon": self.current_epsilon}

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        """
        Performs health checks on the Q-learning model and its parameters.
        """
        errors: List[str] = []

        # 1. Check for numerical stability in the network's weights
        for name, param in self.utility_network.named_parameters():
            if torch.isnan(param).any():
                errors.append(f"Network parameter '{name}' contains NaN values.")
            if torch.isinf(param).any():
                errors.append(f"Network parameter '{name}' contains infinite values.")

        # 2. Check for logical consistency of epsilon
        if not (0.0 <= self.current_epsilon <= 1.0):
            errors.append(f"current_epsilon is out of bounds [0.0, 1.0]. Got: {self.current_epsilon}")

        if torch.isnan(torch.tensor(self.current_epsilon)):
            errors.append("current_epsilon is NaN.")

        # Return True if no errors were found
        return len(errors) == 0, errors
