# simulations/berry_sim/actions.py

from typing import Any, Dict, List
from agent_core.agents.actions.action_interface import ActionInterface
from agent_core.agents.actions.action_registry import action_registry
from agent_core.agents.actions.base_action import ActionOutcome
from agent_core.core.ecs.abstractions import SimulationState
from .components import PositionComponent
from .environment import BerryWorldEnvironment


@action_registry.register
class MoveAction(ActionInterface):
    """Allows an agent to move to an adjacent grid cell."""

    @property
    def action_id(self) -> str:
        return "move"

    @property
    def name(self) -> str:
        return "Move"

    def get_base_cost(self, simulation_state: SimulationState) -> float:
        return 1.0

    def generate_possible_params(
        self, entity_id: str, sim_state: SimulationState, tick: int
    ) -> List[Dict[str, Any]]:
        pos_comp = sim_state.get_component(entity_id, PositionComponent)
        env = sim_state.environment
        if not pos_comp or not isinstance(env, BerryWorldEnvironment):
            return []

        valid_moves = []
        for dx, dy, direction in [(0, 1, "N"), (0, -1, "S"), (1, 0, "E"), (-1, 0, "W")]:
            new_pos = (pos_comp.position[0] + dx, pos_comp.position[1] + dy)
            if env.is_valid_position(new_pos) and not env.is_occupied(new_pos):
                valid_moves.append({"target_pos": new_pos, "direction": direction})
        return valid_moves

    def execute(
        self,
        entity_id: str,
        sim_state: SimulationState,
        params: Dict[str, Any],
        tick: int,
    ) -> ActionOutcome:
        """Logic is handled by MovementSystem."""
        return ActionOutcome(success=True, message="Move initiated.", base_reward=0.0)

    def get_feature_vector(
        self, entity_id: str, sim_state: SimulationState, params: Dict[str, Any]
    ) -> List[float]:
        return [1.0, 0.0, 0.0]  # [is_move, is_eat_red, is_eat_blue, ...]


@action_registry.register
class EatBerryAction(ActionInterface):
    """Allows an agent to eat a berry at its current location."""

    @property
    def action_id(self) -> str:
        return "eat_berry"

    @property
    def name(self) -> str:
        return "Eat Berry"

    def get_base_cost(self, simulation_state: SimulationState) -> float:
        return 1.0

    def generate_possible_params(
        self, entity_id: str, sim_state: SimulationState, tick: int
    ) -> List[Dict[str, Any]]:
        pos_comp = sim_state.get_component(entity_id, PositionComponent)
        env = sim_state.environment
        if not pos_comp or not isinstance(env, BerryWorldEnvironment):
            return []

        berry_type = env.berry_locations.get(pos_comp.position)
        if berry_type:
            return [{"berry_type": berry_type}]
        return []

    def execute(
        self,
        entity_id: str,
        sim_state: SimulationState,
        params: Dict[str, Any],
        tick: int,
    ) -> ActionOutcome:
        """Logic is handled by ConsumptionSystem."""
        return ActionOutcome(
            success=True, message="Consumption initiated.", base_reward=0.0
        )

    def get_feature_vector(
        self, entity_id: str, sim_state: SimulationState, params: Dict[str, Any]
    ) -> List[float]:
        berry_type = params.get("berry_type", "")
        return [
            0.0,
            1.0 if berry_type == "red" else 0.0,
            1.0 if berry_type == "blue" else 0.0,
            1.0 if berry_type == "yellow" else 0.0,
        ]
