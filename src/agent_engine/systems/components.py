# src/agent_engine/systems/components.py

from typing import Any, Dict, List, Tuple
import torch

from agent_core.core.ecs.component import Component
from agent_engine.policy.learned_utility import UtilityNetwork


class QLearningComponent(Component):  # type: ignore[misc]
    """
    Data container for the QLearningSystem, holding the network and optimizer.
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

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        return True, []

    def to_dict(self) -> Dict[str, Any]:
        return {"current_epsilon": self.current_epsilon}
