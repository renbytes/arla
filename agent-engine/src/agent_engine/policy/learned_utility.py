# src/agent_engine/policy/learned_utility.py

from typing import cast

import torch
import torch.nn as nn


class UtilityNetwork(nn.Module):
    """
    A neural network that takes state features, internal state features,
    and action features, and outputs a scalar utility value.
    """

    def __init__(
        self, state_feature_dim: int, internal_state_dim: int, action_feature_dim: int
    ):
        super(UtilityNetwork, self).__init__()
        self.input_dim = state_feature_dim + internal_state_dim + action_feature_dim
        self.fc1 = nn.Linear(self.input_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(
        self,
        state_features: torch.Tensor,
        internal_state_features: torch.Tensor,
        action_features: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(
            [state_features, internal_state_features, action_features], dim=-1
        )
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        output = self.fc3(x)

        return cast(torch.Tensor, output)
