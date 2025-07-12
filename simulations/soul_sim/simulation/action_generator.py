from typing import Any, List

from agent_core.agents.action_generator_interface import ActionGeneratorInterface
from agent_core.agents.actions.action_registry import action_registry
from agent_core.core.ecs.component import ActionPlanComponent


class SoulSimActionGenerator(ActionGeneratorInterface):
    """Generates all possible actions for an agent in the soul_sim world."""

    def generate(self, simulation_state: Any, entity_id: str, current_tick: int) -> List[ActionPlanComponent]:
        all_possible_plans: List[ActionPlanComponent] = []
        for action_class in action_registry.get_all_actions():
            action_instance = action_class()
            for params in action_instance.generate_possible_params(entity_id, simulation_state, current_tick):
                all_possible_plans.append(
                    ActionPlanComponent(
                        action_type=action_instance,
                        intent=params.get("intent"),
                        params=params,
                    )
                )
        return all_possible_plans
