# FILE: simulations/emergence_sim/actions/move_action.py
from typing import TYPE_CHECKING, Any, Dict, List

from agent_core.agents.actions.action_interface import ActionInterface
from agent_core.agents.actions.action_registry import action_registry
from agent_core.agents.actions.base_action import Intent

from simulations.emergence_sim.actions.economic_actions import (
    _create_emergence_feature_vector,
)
from simulations.emergence_sim.components import PositionComponent

if TYPE_CHECKING:
    from agent_core.core.ecs.abstractions import SimulationState


@action_registry.register
class MoveAction(ActionInterface):
    action_id = "move"
    name = "Move"

    def get_base_cost(self, simulation_state: "SimulationState") -> float:
        return 1.0

    def generate_possible_params(
        self, entity_id: str, simulation_state: "SimulationState", current_tick: int
    ) -> List[Dict[str, Any]]:
        params_list: List[Dict[str, Any]] = []
        pos_comp = simulation_state.get_component(entity_id, PositionComponent)
        if not pos_comp or not hasattr(pos_comp, "environment"):
            return []

        my_pos = pos_comp.position
        # Simple cardinal directions
        possible_moves = [
            ((my_pos[0] + 1, my_pos[1]), 0),  # Down
            ((my_pos[0] - 1, my_pos[1]), 1),  # Up
            ((my_pos[0], my_pos[1] + 1), 2),  # Right
            ((my_pos[0], my_pos[1] - 1), 3),  # Left
        ]

        for pos, direction in possible_moves:
            if pos_comp.environment.is_valid_position(pos):
                params_list.append({"direction": direction, "intent": Intent.SOLITARY, "new_pos": pos})
        return params_list

    def execute(
        self,
        entity_id: str,
        simulation_state: "SimulationState",
        params: Dict[str, Any],
        current_tick: int,
    ) -> Dict[str, Any]:
        return {"status": "move_attempted"}

    def get_feature_vector(
        self,
        entity_id: str,
        simulation_state: "SimulationState",
        params: Dict[str, Any],
    ) -> List[float]:
        return _create_emergence_feature_vector(
            self.action_id,
            params.get("intent", Intent.SOLITARY),
            self.get_base_cost(simulation_state),
            params,
            {"direction": (0, params.get("direction", 0), 3.0)},
            config=simulation_state.config,
        )
