from typing import List, Optional

from agent_core.agents.decision_selector_interface import DecisionSelectorInterface
from agent_core.core.ecs.abstractions import SimulationState
from agent_core.core.ecs.component import ActionPlanComponent

# Import world-specific components to check the agent's state
from .components import SatisfactionComponent


class SchellingDecisionSelector(DecisionSelectorInterface):
    """
    A simple decision selector for the Schelling model.
    If an agent is unsatisfied, it chooses the 'relocate' action.
    If it is satisfied, it chooses no action and stays put.
    """

    def select(
        self,
        simulation_state: SimulationState,
        entity_id: str,
        possible_actions: List[ActionPlanComponent],
    ) -> Optional[ActionPlanComponent]:
        """
        Selects the best action for an agent based on its satisfaction level.
        """
        sat_comp = simulation_state.get_component(entity_id, SatisfactionComponent)

        if not sat_comp:
            return None

        # If the agent is unsatisfied, we assume it chooses to relocate if that action is available.
        if not sat_comp.is_satisfied:
            for action in possible_actions:
                if action.action_type and action.action_type.action_id == "relocate":
                    return action

        # If satisfied, or if no 'relocate' action is available when unsatisfied, the agent does nothing.
        # The core `run` loop will then simply advance to the next agent.
        return None
