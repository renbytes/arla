# simulations/schelling_sim/providers.py

from typing import Any, Dict, List, Optional, Tuple, Type

from agent_core.agents.action_generator_interface import ActionGeneratorInterface
from agent_core.agents.decision_selector_interface import DecisionSelectorInterface
from agent_core.core.ecs.component import ActionPlanComponent, Component
from agent_core.core.ecs.component_factory_interface import ComponentFactoryInterface
from agent_core.policy.reward_calculator_interface import RewardCalculatorInterface

from .actions import MoveToEmptyCellAction
from .components import GroupComponent, PositionComponent, SatisfactionComponent


class SchellingActionGenerator(ActionGeneratorInterface):
    """Generates possible moves for unsatisfied agents."""

    def generate(self, sim_state, entity_id, tick) -> List[ActionPlanComponent]:
        move_action = MoveToEmptyCellAction()
        params_list = move_action.generate_possible_params(entity_id, sim_state, tick)
        return [
            ActionPlanComponent(action_type=move_action, params=p) for p in params_list
        ]


class SchellingDecisionSelector(DecisionSelectorInterface):
    """A simple policy: if an agent can move, it will."""

    def select(self, sim_state, entity_id, actions) -> Optional[ActionPlanComponent]:
        return actions[0] if actions else None


class SchellingComponentFactory(ComponentFactoryInterface):
    """Creates component instances from saved data."""

    def create_component(self, component_type: str, data: Dict[str, Any]) -> Component:
        # Assumes the full class path is provided in a real restore scenario
        class_name = component_type.split(".")[-1]

        if class_name == "PositionComponent":
            return PositionComponent(**data)
        if class_name == "GroupComponent":
            return GroupComponent(**data)
        if class_name == "SatisfactionComponent":
            return SatisfactionComponent(**data)
        raise TypeError(f"Unknown component type: {component_type}")


class SchellingRewardCalculator(RewardCalculatorInterface):
    """A simple reward calculator for the Schelling model."""

    def calculate_final_reward(
        self,
        base_reward: float,
        action_type: Any,
        action_intent: str,
        outcome_details: Dict[str, Any],
        entity_components: Dict[Type[Component], "Component"],
    ) -> Tuple[float, Dict[str, Any]]:
        """Simply returns the base reward of the action without modification."""
        return base_reward, {"base_reward": base_reward}
