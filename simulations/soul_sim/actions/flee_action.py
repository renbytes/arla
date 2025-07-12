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
class FleeAction(ActionInterface):
    action_id = "flee"
    name = "Flee"

    def get_base_cost(self, simulation_state: "SimulationState") -> float:
        return simulation_state.config.get("agent", {}).get("flee_base_cost", 2.0)

    def _find_nearby_threats(self, entity_id: str, my_pos: tuple, simulation_state: "SimulationState") -> List[tuple]:
        pos_comp = simulation_state.get_component(entity_id, PositionComponent)
        if not isinstance(pos_comp, PositionComponent) or not hasattr(pos_comp, "environment"):
            return []

        nearby_entities = pos_comp.environment.get_entities_in_radius(my_pos, 2)
        threats = []
        for other_id, other_pos in nearby_entities:
            if other_id == entity_id:
                continue
            other_time_comp = simulation_state.get_component(other_id, TimeBudgetComponent)
            if isinstance(other_time_comp, TimeBudgetComponent) and other_time_comp.is_active:
                threats.append(other_pos)
        return threats

    def _calculate_best_flee_direction(
        self,
        my_pos: tuple,
        threats: List[tuple],
        pos_comp: PositionComponent,
        simulation_state: "SimulationState",
    ) -> int:
        if not threats or not hasattr(pos_comp, "environment"):
            return -1
        avg_threat_pos = (
            sum(p[0] for p in threats) / len(threats),
            sum(p[1] for p in threats) / len(threats),
        )
        best_flee_pos, max_dist = None, -1.0
        possible_new_positions = pos_comp.environment.get_neighbors(my_pos)
        for new_pos in possible_new_positions:
            dist_from_threat = pos_comp.environment.distance(new_pos, avg_threat_pos)
            if dist_from_threat > max_dist:
                max_dist = dist_from_threat
                best_flee_pos = new_pos
        if best_flee_pos is None:
            return -1
        dx, dy = best_flee_pos[0] - my_pos[0], best_flee_pos[1] - my_pos[1]
        if dx == -1:
            return 0
        if dx == 1:
            return 1
        if dy == -1:
            return 2
        if dy == 1:
            return 3
        return -1

    def generate_possible_params(
        self, entity_id: str, simulation_state: "SimulationState", current_tick: int
    ) -> List[Dict[str, Any]]:
        pos_comp = simulation_state.get_component(entity_id, PositionComponent)
        if not isinstance(pos_comp, PositionComponent):
            return []
        my_pos = pos_comp.position
        threats = self._find_nearby_threats(entity_id, my_pos, simulation_state)
        if threats:
            best_direction = self._calculate_best_flee_direction(my_pos, threats, pos_comp, simulation_state)
            if best_direction != -1:
                return [{"direction": best_direction, "intent": Intent.SOLITARY}]
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
        intent = params.get("intent", Intent.SOLITARY)
        return create_standard_feature_vector(
            self.action_id,
            intent,
            self.get_base_cost(simulation_state),
            params,
            {"direction": (0, params.get("direction", 0), 3.0)},
        )
