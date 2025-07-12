# actions/action_definition.py

from typing import TYPE_CHECKING, Any, Dict, List

from agent_core.agents.actions.action_interface import ActionInterface
from agent_core.agents.actions.action_registry import action_registry
from agent_core.agents.actions.base_action import Intent

from simulations.soul_sim.actions.action_utils import create_standard_feature_vector
from simulations.soul_sim.components import (
    InventoryComponent,
    PositionComponent,
    TimeBudgetComponent,
)
from simulations.soul_sim.environment.resources import get_resource_at_pos

if TYPE_CHECKING:
    from agent_core.core.ecs.abstractions import SimulationState


@action_registry.register
class ExtractAction(ActionInterface):
    action_id = "extract"
    name = "Extract"

    def get_base_cost(self, simulation_state: "SimulationState") -> float:
        return simulation_state.config.get("agent", {}).get("extract_base_cost", 5.0)

    def generate_possible_params(
        self, entity_id: str, simulation_state: "SimulationState", current_tick: int
    ) -> List[Dict[str, Any]]:
        params_list: List[Dict[str, Any]] = []
        pos_comp = simulation_state.get_component(entity_id, PositionComponent)
        inv_comp = simulation_state.get_component(entity_id, InventoryComponent)
        if not isinstance(pos_comp, PositionComponent) or not isinstance(inv_comp, InventoryComponent):
            return []

        entity_pos = pos_comp.position
        resource_result = get_resource_at_pos(simulation_state, entity_pos)

        if resource_result:
            resource_id, resource_comp = resource_result
            if hasattr(pos_comp, "environment"):
                nearby_entities = pos_comp.environment.get_entities_at_position(entity_pos)
                other_agents_present = any(
                    isinstance(
                        time_comp := simulation_state.get_component(other_id, TimeBudgetComponent),
                        TimeBudgetComponent,
                    )
                    and time_comp.is_active
                    for other_id in nearby_entities
                    if other_id != entity_id
                )

                if other_agents_present and resource_comp.min_agents_needed > 1:
                    params_list.append({"resource_id": resource_id, "intent": Intent.COOPERATE})
                    params_list.append({"resource_id": resource_id, "intent": Intent.COMPETE})
                else:
                    params_list.append({"resource_id": resource_id, "intent": Intent.SOLITARY})

        if not inv_comp.farming_mode and inv_comp.current_resources >= simulation_state.config.get(
            "farm_threshold_resources", 5.0
        ):
            params_list.append({"establish_farm": True, "intent": Intent.SOLITARY})

        if inv_comp.farming_mode and inv_comp.farm_location == entity_pos:
            params_list.append({"is_farm": True, "intent": Intent.SOLITARY})

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
        intent = params.get("intent", Intent.SOLITARY)
        return create_standard_feature_vector(
            self.action_id,
            intent,
            self.get_base_cost(simulation_state),
            params,
            {
                "resource_id": (0, 1, 1),
                "establish_farm": (1, 1, 1),
                "is_farm": (2, 1, 1),
            },
        )
