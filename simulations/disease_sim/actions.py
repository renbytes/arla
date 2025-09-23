# simulations/disease_sim/actions.py
"""
Defines agent actions for the Disease simulation. In this passive model,
the only action is to 'wait' as disease progression is system-driven.
"""

from typing import Any, Dict, List
from agent_core.agents.actions.action_interface import ActionInterface
from agent_core.agents.actions.action_registry import action_registry
from agent_core.agents.actions.action_outcome import ActionOutcome
from agent_core.core.ecs.abstractions import SimulationState as AbstractSimulationState


@action_registry.register
class WaitAction(ActionInterface):
    """Allows an agent to do nothing for a turn."""

    @property
    def action_id(self) -> str:
        return "wait"

    @property
    def name(self) -> str:
        return "Wait"

    def get_base_cost(self, simulation_state: AbstractSimulationState) -> float:
        return 0.0

    def generate_possible_params(
        self, entity_id: str, sim_state: AbstractSimulationState, tick: int
    ) -> List[Dict[str, Any]]:
        # This action is always possible for all agents.
        return [{}]

    def execute(
        self,
        entity_id: str,
        sim_state: AbstractSimulationState,
        params: Dict[str, Any],
        tick: int,
    ) -> ActionOutcome:
        return ActionOutcome(success=True, message="Agent waits.", base_reward=0.0)

    def get_feature_vector(
        self, entity_id: str, sim_state: AbstractSimulationState, params: Dict[str, Any]
    ) -> List[float]:
        return [1.0]
