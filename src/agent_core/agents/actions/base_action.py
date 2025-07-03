# src/agent_core/agents/actions/base_action.py
from enum import Enum
from typing import Any, Dict, Optional


class Intent(Enum):
    """Enumeration of high-level modifiers or motivations for actions."""

    SOLITARY = "SOLITARY"
    COOPERATE = "COOPERATE"
    COMPETE = "COMPETE"


class ActionOutcome:
    """
    Standardized structure for the result of executing an action.
    This is a simple data container.
    """

    def __init__(
        self,
        success: bool,
        message: str,
        base_reward: float,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.success = success
        self.message = message
        self.base_reward = base_reward
        self.details = details if details is not None else {}
        self.reward: float = base_reward
