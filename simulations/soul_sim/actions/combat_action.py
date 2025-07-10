# actions/action_definition.py

from typing import TYPE_CHECKING, Any, Dict, List

from simulations.soul_sim.actions.action_utils import create_standard_feature_vector
from simulations.soul_sim.components import PositionComponent, HealthComponent, TimeBudgetComponent
from agent_core.agents.actions.action_interface import ActionInterface
from agent_core.agents.actions.action_registry import action_registry
from agent_core.agents.actions.base_action import Intent

if TYPE_CHECKING:
    from agent_core.core.ecs.abstractions import SimulationState


@action_registry.register
class CombatAction(ActionInterface):
    action_id = "combat"
    name = "Combat"

    def get_base_cost(self, simulation_state: "SimulationState") -> float:
        return simulation_state.config.get("agent", {}).get("combat_base_cost", 10.0)

    def generate_possible_params(self, entity_id: str, simulation_state: "SimulationState", current_tick: int) -> List[Dict[str, Any]]:
        params_list: List[Dict[str, Any]] = []
        pos_comp = simulation_state.get_component(entity_id, PositionComponent)
        health_comp = simulation_state.get_component(entity_id, HealthComponent)

        if not isinstance(pos_comp, PositionComponent) or not isinstance(health_comp, HealthComponent) or health_comp.current_health <= 0:
            return []

        if not hasattr(pos_comp, "environment"):
            return []

        entity_pos = pos_comp.position
        nearby_entities = pos_comp.environment.get_entities_in_radius(entity_pos, 1)

        for other_id, _ in nearby_entities:
            if entity_id == other_id:
                continue
            other_time_comp = simulation_state.get_component(other_id, TimeBudgetComponent)
            if isinstance(other_time_comp, TimeBudgetComponent) and other_time_comp.is_active:
                params_list.append({"target_agent_id": other_id, "intent": Intent.COMPETE})

        return params_list

    def execute(self, entity_id: str, simulation_state: "SimulationState", params: Dict[str, Any], current_tick: int) -> Dict[str, Any]:
        return {}

    def get_feature_vector(self, entity_id: str, simulation_state: "SimulationState", params: Dict[str, Any]) -> List[float]:
        intent = params.get("intent", Intent.COMPETE)
        return create_standard_feature_vector(self.action_id, intent, self.get_base_cost(simulation_state), params, {"target_agent_id": (0, 1, 1)})
