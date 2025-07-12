# actions/action_definition.py

from typing import TYPE_CHECKING, Any, Dict, List

from agent_core.agents.actions.action_interface import ActionInterface
from agent_core.agents.actions.action_registry import action_registry
from agent_core.agents.actions.base_action import Intent

from simulations.soul_sim.actions.action_utils import create_standard_feature_vector
from simulations.soul_sim.components import PositionComponent

if TYPE_CHECKING:
    from agent_core.core.ecs.abstractions import SimulationState


@action_registry.register
class MoveAction(ActionInterface):
    action_id = "move"
    name = "Move"

    def get_base_cost(self, simulation_state: "SimulationState") -> float:
        return simulation_state.config.get("agent", {}).get("action_base_cost", 1.0)

    def generate_possible_params(
        self, entity_id: str, simulation_state: "SimulationState", current_tick: int
    ) -> List[Dict[str, Any]]:
        params_list: List[Dict[str, Any]] = []
        pos_comp = simulation_state.get_component(entity_id, PositionComponent)
        if not isinstance(pos_comp, PositionComponent) or not hasattr(pos_comp, "environment"):
            return []

        my_pos = pos_comp.position
        neighbors = pos_comp.environment.get_neighbors(my_pos)

        if (my_pos[0] - 1, my_pos[1]) in neighbors:
            params_list.append({"direction": 0, "intent": Intent.SOLITARY})
        if (my_pos[0] + 1, my_pos[1]) in neighbors:
            params_list.append({"direction": 1, "intent": Intent.SOLITARY})
        if (my_pos[0], my_pos[1] - 1) in neighbors:
            params_list.append({"direction": 2, "intent": Intent.SOLITARY})
        if (my_pos[0], my_pos[1] + 1) in neighbors:
            params_list.append({"direction": 3, "intent": Intent.SOLITARY})

        return params_list

    def execute(
        self,
        entity_id: str,
        simulation_state: "SimulationState",
        params: Dict[str, Any],
        current_tick: int,
    ) -> Dict[str, Any]:
        return {}  # Logic is handled in MovementSystem

    def get_feature_vector(
        self,
        entity_id: str,
        simulation_state: "SimulationState",
        params: Dict[str, Any],
    ) -> List[float]:
        return create_standard_feature_vector(
            self.action_id,
            Intent.SOLITARY,
            self.get_base_cost(simulation_state),
            params,
            {"direction": (0, params.get("direction", 0), 3.0)},
        )
