# src/agent_engine/policy/reward_calculator.py

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Type
from agent_core.core.ecs.component import Component


class RewardCalculator(ABC):
    @abstractmethod
    def calculate_final_reward(
        self,
        base_reward: float,
        action_type: Any,
        action_intent: str,
        outcome_details: Dict[str, Any],
        entity_components: Dict[Type[Component], Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculates the final subjective reward for an agent."""
        raise NotImplementedError
