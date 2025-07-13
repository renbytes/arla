# actions/action_definition.py

from typing import TYPE_CHECKING, Any, Dict, List

from agent_core.agents.actions.action_interface import ActionInterface
from agent_core.agents.actions.action_registry import action_registry
from agent_core.agents.actions.base_action import Intent

from simulations.soul_sim.actions.action_utils import create_standard_feature_vector
from simulations.soul_sim.components import PositionComponent, TimeBudgetComponent

if TYPE_CHECKING:
    from agent_core.core.ecs.abstractions import SimulationState


@action_registry.register
class CommunicateAction(ActionInterface):
    action_id = "communicate"
    name = "Communicate"

    def get_base_cost(self, simulation_state: "SimulationState") -> float:
        return simulation_state.config.agent.costs.actions.communicate

    def generate_possible_params(
        self, entity_id: str, simulation_state: "SimulationState", current_tick: int
    ) -> List[Dict[str, Any]]:
        params_list: List[Dict[str, Any]] = []
        pos_comp = simulation_state.get_component(entity_id, PositionComponent)
        if not isinstance(pos_comp, PositionComponent) or not hasattr(pos_comp, "environment"):
            return []

        entity_pos = pos_comp.position
        nearby_entities = pos_comp.environment.get_entities_in_radius(entity_pos, 2)

        for other_id, _ in nearby_entities:
            if entity_id == other_id:
                continue
            other_time_comp = simulation_state.get_component(other_id, TimeBudgetComponent)
            if isinstance(other_time_comp, TimeBudgetComponent) and other_time_comp.is_active:
                params_list.append(
                    {
                        "target_agent_id": other_id,
                        "message": "information_exchange",
                        "intent": Intent.COOPERATE,
                    }
                )
                params_list.append(
                    {
                        "target_agent_id": other_id,
                        "message": "assert_dominance",
                        "intent": Intent.COMPETE,
                    }
                )

        return params_list

    def execute(
        self,
        entity_id: str,
        simulation_state: "SimulationState",
        params: Dict[str, Any],
        current_tick: int,
    ) -> Dict[str, Any]:
        return {}

    def get_feature_vector(
        self,
        entity_id: str,
        simulation_state: "SimulationState",
        params: Dict[str, Any],
    ) -> List[float]:
        intent = params.get("intent", Intent.COOPERATE)
        return create_standard_feature_vector(
            self.action_id,
            intent,
            self.get_base_cost(simulation_state),
            params,
            {"target_agent_id": (0, 1, 1)},
        )
