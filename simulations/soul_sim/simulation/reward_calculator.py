from typing import Any, Dict, Tuple, Type

from agent_core.core.ecs.component import Component, ValueSystemComponent
from agent_core.policy.reward_calculator_interface import RewardCalculatorInterface


class SoulSimRewardCalculator(RewardCalculatorInterface):
    """Calculates final, subjective rewards for actions in the soul_sim world."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("learning", {}).get("rewards", {})

    def calculate_final_reward(
        self,
        base_reward: float,
        action_type: Any,
        action_intent: str,
        outcome_details: Dict[str, Any],
        entity_components: Dict[Type[Component], Component],
    ) -> Tuple[float, Dict[str, Any]]:
        action_id = getattr(action_type, "action_id", "unknown")

        # Start with the base reward
        current_reward = base_reward
        breakdown: Dict[str, Any] = {"base": base_reward}

        # 1. Add objective world bonuses FIRST
        bonus = 0.0
        if outcome_details.get("status") == "defeated_entity":
            bonus = self.config.get("combat_victory", 10.0)
        elif outcome_details.get("explored_new_tile", False):
            bonus = self.config.get("exploration_bonus", 0.5)

        if bonus != 0.0:
            current_reward += bonus
            breakdown["bonus"] = bonus

        # 2. Apply the agent's subjective value multiplier LAST
        if isinstance(vs := entity_components.get(ValueSystemComponent), ValueSystemComponent):
            multiplier = 1.0
            if action_id == "combat":
                multiplier = vs.combat_victory_multiplier
            elif action_id == "extract":
                multiplier = vs.resource_yield_multiplier
            elif action_id == "move":
                multiplier = vs.exploration_multiplier
            elif action_id == "communicate" and action_intent == "COOPERATE":
                multiplier = vs.collaboration_multiplier

            if multiplier != 1.0:
                current_reward *= multiplier
                breakdown["value_multiplier"] = multiplier

        breakdown["final"] = current_reward
        return current_reward, breakdown
