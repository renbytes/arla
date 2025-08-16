# simulations/emergence_sim/actions/influence_action.py
"""Defines the sole action for the Cognitive Voter Model: Influence."""

from typing import TYPE_CHECKING, Any, Dict, List

from agent_core.agents.actions.action_interface import ActionInterface
from agent_core.agents.actions.action_registry import action_registry
from agent_core.agents.actions.base_action import Intent

from simulations.emergence_sim.actions.communication_actions import (
    _create_emergence_feature_vector,
)

if TYPE_CHECKING:
    from agent_core.core.ecs.abstractions import SimulationState


@action_registry.register
class InfluenceAction(ActionInterface):
    """An agent attempts to influence a neighbor, which results in the agent
    adopting the neighbor's opinion, modulated by cognitive factors."""

    action_id = "influence"
    name = "Influence"

    def get_base_cost(self, simulation_state: "SimulationState") -> float:
        """A very low-cost action."""
        return 1.0

    def generate_possible_params(
        self, entity_id: str, simulation_state: "SimulationState", current_tick: int
    ) -> List[Dict[str, Any]]:
        """This action is always possible and requires no specific target."""
        return [{"intent": Intent.SOLITARY}]

    def execute(
        self,
        entity_id: str,
        simulation_state: "SimulationState",
        params: Dict[str, Any],
        current_tick: int,
    ) -> Dict[str, Any]:
        """
        The logic is handled by the OpinionDynamicsSystem. This method simply
        confirms the action was dispatched.
        """
        return {"status": "influence_attempted"}

    def get_feature_vector(
        self,
        entity_id: str,
        simulation_state: "SimulationState",
        params: Dict[str, Any],
    ) -> List[float]:
        """Generates the feature vector for this action."""
        return _create_emergence_feature_vector(
            self.action_id,
            params.get("intent", Intent.SOLITARY),
            self.get_base_cost(simulation_state),
            params,
            {},
            config=simulation_state.config,
        )
