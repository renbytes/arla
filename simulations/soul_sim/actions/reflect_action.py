# actions/action_definition.py

from typing import TYPE_CHECKING, Any, Dict, List

from agent_core.agents.actions.action_interface import ActionInterface
from agent_core.agents.actions.action_registry import action_registry
from agent_core.agents.actions.base_action import Intent

from simulations.soul_sim.actions.action_utils import create_standard_feature_vector
from simulations.soul_sim.components import AffectComponent

if TYPE_CHECKING:
    from agent_core.core.ecs.abstractions import SimulationState


@action_registry.register
class ReflectAction(ActionInterface):
    action_id = "reflect"
    name = "Reflect"

    def get_base_cost(self, simulation_state: "SimulationState") -> float:
        return simulation_state.config.agent.costs.actions.reflect

    def generate_possible_params(
        self, entity_id: str, simulation_state: "SimulationState", current_tick: int
    ) -> List[Dict[str, Any]]:
        total_steps = simulation_state.config.simulation.steps
        if current_tick >= total_steps - 1:
            return []

        dissonance_threshold = simulation_state.config.learning.memory.cognitive_dissonance_threshold

        affect_comp = simulation_state.get_component(entity_id, AffectComponent)
        if isinstance(affect_comp, AffectComponent) and affect_comp.cognitive_dissonance > dissonance_threshold:
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
