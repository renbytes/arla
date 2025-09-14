"""
Defines the agent actions for the Tragedy of the Commons simulation.

Each class implements the ActionInterface, defining a possible behavior. The
logic for how these actions affect the world is handled by Systems, which
listen for events published when an action's 'execute' method is called.
"""

from typing import Any, Dict, List, cast

from agent_core.agents.actions.action_interface import ActionInterface
from agent_core.agents.actions.action_registry import action_registry
from agent_core.agents.actions.action_outcome import ActionOutcome
from agent_core.core.ecs.abstractions import SimulationState as AbstractSimulationState

from .components import PositionComponent
from .environment import CommonsEnvironment


@action_registry.register
class MoveAction(ActionInterface):
    """Allows an agent to move to an adjacent, unoccupied cell."""

    @property
    def action_id(self) -> str:
        return "move"

    @property
    def name(self) -> str:
        return "Move"

    def get_base_cost(self, simulation_state: AbstractSimulationState) -> float:
        """The energy cost of moving."""
        return 1.0

    def generate_possible_params(
        self, entity_id: str, sim_state: AbstractSimulationState, tick: int
    ) -> List[Dict[str, Any]]:
        """Generates parameters for all valid moves (N, S, E, W)."""
        if not hasattr(sim_state, "environment"):
            return []

        pos_comp = sim_state.get_component(entity_id, PositionComponent)
        env = sim_state.environment
        if not pos_comp or not isinstance(env, CommonsEnvironment):
            return []

        pos_comp = cast(PositionComponent, pos_comp)

        valid_moves = []
        for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
            new_pos = (pos_comp.position[0] + dx, pos_comp.position[1] + dy)
            if env.is_valid_position(new_pos) and not env.is_occupied_by_agent(new_pos):
                valid_moves.append({"target_pos": new_pos})
        return valid_moves

    def execute(
        self,
        entity_id: str,
        sim_state: AbstractSimulationState,
        params: Dict[str, Any],
        tick: int,
    ) -> ActionOutcome:
        """Signals the intent to move; logic is handled by MovementSystem."""
        return ActionOutcome(success=True, message="Move initiated.", base_reward=0.0)

    def get_feature_vector(
        self, entity_id: str, sim_state: AbstractSimulationState, params: Dict[str, Any]
    ) -> List[float]:
        """One-hot encoding for the action space."""
        return [1.0, 0.0, 0.0]  # [is_move, is_graze, is_wait]


@action_registry.register
class GrazeAction(ActionInterface):
    """Allows an agent to consume grass from its current cell."""

    @property
    def action_id(self) -> str:
        return "graze"

    @property
    def name(self) -> str:
        return "Graze"

    def get_base_cost(self, simulation_state: AbstractSimulationState) -> float:
        return 0.5  # Grazing has a small energy cost

    def generate_possible_params(
        self, entity_id: str, sim_state: AbstractSimulationState, tick: int
    ) -> List[Dict[str, Any]]:
        """Action is possible if there is grass at the agent's location."""
        if not hasattr(sim_state, "environment"):
            return []

        pos_comp = sim_state.get_component(entity_id, PositionComponent)
        env = sim_state.environment
        if not pos_comp or not isinstance(env, CommonsEnvironment):
            return []

        pos_comp = cast(PositionComponent, pos_comp)

        if env.get_resource_at(pos_comp.position) > 0:
            return [{}]  # No parameters needed
        return []

    def execute(
        self,
        entity_id: str,
        sim_state: AbstractSimulationState,
        params: Dict[str, Any],
        tick: int,
    ) -> ActionOutcome:
        """Signals intent to graze; logic is handled by GrazingSystem."""
        return ActionOutcome(
            success=True, message="Grazing initiated.", base_reward=0.0
        )

    def get_feature_vector(
        self, entity_id: str, sim_state: AbstractSimulationState, params: Dict[str, Any]
    ) -> List[float]:
        """One-hot encoding for the action space."""
        return [0.0, 1.0, 0.0]  # [is_move, is_graze, is_wait]


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
        """Waiting has no additional energy cost beyond metabolism."""
        return 0.0

    def generate_possible_params(
        self, entity_id: str, sim_state: AbstractSimulationState, tick: int
    ) -> List[Dict[str, Any]]:
        """This action is always possible."""
        return [{}]

    def execute(
        self,
        entity_id: str,
        sim_state: AbstractSimulationState,
        params: Dict[str, Any],
        tick: int,
    ) -> ActionOutcome:
        """Logic is handled by MetabolismSystem (passive energy decay)."""
        return ActionOutcome(success=True, message="Agent waits.", base_reward=0.0)

    def get_feature_vector(
        self, entity_id: str, sim_state: AbstractSimulationState, params: Dict[str, Any]
    ) -> List[float]:
        """One-hot encoding for the action space."""
        return [0.0, 0.0, 1.0]  # [is_move, is_graze, is_wait]
