# simulations/schelling_sim/actions.py

import random
from typing import Any, Dict, List, cast

from agent_core.agents.actions.action_interface import ActionInterface
from agent_core.agents.actions.action_registry import action_registry
from agent_core.agents.actions.action_outcome import ActionOutcome
from agent_core.core.ecs.abstractions import SimulationState as AbstractSimulationState

# --- FIX: Import the new, separated components ---
from .components import PositionComponent, SatisfactionComponent
from .environment import SchellingGridEnvironment


@action_registry.register
class MoveToEmptyCellAction(ActionInterface):
    """
    An action that allows an unsatisfied agent to move to a random empty cell.
    """

    @property
    def action_id(self) -> str:
        """A unique string identifier for the action."""
        return "move_to_empty_cell"

    @property
    def name(self) -> str:
        """A human-readable name for the action."""
        return "Move to Empty Cell"

    def get_base_cost(self, simulation_state: AbstractSimulationState) -> float:
        """The base time budget cost to perform the action."""
        return 1.0

    def generate_possible_params(
        self,
        entity_id: str,
        simulation_state: AbstractSimulationState,
        current_tick: int,
    ) -> List[Dict[str, Any]]:
        """
        Generates a move parameter if the agent is unsatisfied and there are
        empty cells available.
        """
        # FIX: Check for attribute existence instead of strict type for mock-friendliness.
        if not hasattr(simulation_state, "environment"):
            return []

        satisfaction_comp = simulation_state.get_component(
            entity_id, SatisfactionComponent
        )

        if not satisfaction_comp:
            return []

        satisfaction_comp = cast(SatisfactionComponent, satisfaction_comp)

        # Only generate a move action if the agent is unsatisfied.
        if satisfaction_comp.is_satisfied:
            return []

        env = simulation_state.environment
        if not isinstance(env, SchellingGridEnvironment):
            return []

        empty_cells = env.get_empty_cells()
        if not empty_cells:
            return []

        # The agent will move to a random empty cell.
        target_cell = random.choice(empty_cells)
        return [{"target_x": target_cell[0], "target_y": target_cell[1]}]

    def execute(
        self,
        entity_id: str,
        simulation_state: AbstractSimulationState,
        params: Dict[str, Any],
        current_tick: int,
    ) -> ActionOutcome:
        """
        Signals the intent to move. The actual move logic is handled by the
        MovementSystem.
        """
        pos_comp = simulation_state.get_component(entity_id, PositionComponent)
        if not pos_comp:
            return ActionOutcome(
                success=False,
                message="Agent has no PositionComponent.",
                base_reward=-0.1,
            )

        pos_comp = cast(PositionComponent, pos_comp)

        target_pos = (params.get("target_x"), params.get("target_y"))
        return ActionOutcome(
            success=True,
            message=f"Agent {entity_id} moves from {pos_comp.position} to {target_pos}.",
            base_reward=1.0,
        )

    def get_feature_vector(
        self,
        entity_id: str,
        simulation_state: AbstractSimulationState,
        params: Dict[str, Any],
    ) -> List[float]:
        """Generates a feature vector for this action (not used in this model)."""
        return [1.0]
