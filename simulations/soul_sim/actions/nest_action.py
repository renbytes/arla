# actions/action_definition.py

from typing import TYPE_CHECKING, Any, Dict, List

from agent_core.agents.actions.action_interface import ActionInterface
from agent_core.agents.actions.action_registry import action_registry
from agent_core.agents.actions.base_action import Intent

from simulations.soul_sim.actions.action_utils import create_standard_feature_vector
from simulations.soul_sim.components import InventoryComponent

if TYPE_CHECKING:
    from agent_core.core.ecs.abstractions import SimulationState


@action_registry.register
class NestAction(ActionInterface):
    action_id = "nest"
    name = "Nest"

    def get_base_cost(self, simulation_state: "SimulationState") -> float:
        return simulation_state.config.get("agent", {}).get("nest_base_cost", 25.0)

    def generate_possible_params(
        self, entity_id: str, simulation_state: "SimulationState", current_tick: int
    ) -> List[Dict[str, Any]]:
        inv_comp = simulation_state.get_component(entity_id, InventoryComponent)
        if isinstance(inv_comp, InventoryComponent) and inv_comp.current_resources >= simulation_state.config.get(
            "nest_resource_cost", 120.0
        ):
            return [{"intent": Intent.SOLITARY}]
        return []

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
        return create_standard_feature_vector(
            self.action_id,
            Intent.SOLITARY,
            self.get_base_cost(simulation_state),
            params,
            {},
        )
