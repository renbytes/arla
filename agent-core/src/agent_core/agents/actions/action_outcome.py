# agent-core/src/agent_core/agents/actions/action_outcome.py
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ActionOutcome:
    """A data structure to hold the results of an action's execution."""

    success: bool
    message: str
    base_reward: float
    details: Dict[str, Any] = field(default_factory=dict)
    reward: float = field(init=False)

    def __post_init__(self):
        # The final reward can be modified by other systems,
        # but it starts as the base reward.
        self.reward = self.base_reward
