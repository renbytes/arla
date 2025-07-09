# src/simulations/soul_sim/systems/movement_system.py
"""
Handles all entity movement, including standard moves and fleeing.
"""

from typing import Any, Dict, List, Tuple, Type

from agent_core.agents.actions.base_action import ActionOutcome
from agent_core.core.ecs.component import Component
from agent_engine.simulation.system import System

# Import world-specific components from the current simulation package
from ..components import PositionComponent


def _apply_move_logic(current_pos: Tuple[int, int], direction: int, grid_size: Tuple[int, int]) -> Tuple[int, int]:
    """Calculates the new position based on a directional integer."""
    new_pos = list(current_pos)
    grid_height, grid_width = grid_size

    if direction == 0:
        new_pos[0] -= 1  # Up
    elif direction == 1:
        new_pos[0] += 1  # Down
    elif direction == 2:
        new_pos[1] -= 1  # Left
    elif direction == 3:
        new_pos[1] += 1  # Right

    # Clamp the new position to be within the grid boundaries
    new_pos[0] = max(0, min(new_pos[0], grid_height - 1))
    new_pos[1] = max(0, min(new_pos[1], grid_width - 1))
    return (new_pos[0], new_pos[1])


class MovementSystem(System):
    """
    Processes move and flee actions, updates entity positions, and tracks exploration.
    """

    REQUIRED_COMPONENTS: List[Type[Component]] = []  # Event-driven

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.event_bus:
            self.event_bus.subscribe("execute_move_action", self.on_execute_move)
            self.event_bus.subscribe("execute_flee_action", self.on_execute_flee)

    def on_execute_move(self, event_data: Dict[str, Any]):
        """Handles a standard move action."""
        self._handle_movement(event_data, is_flee=False)

    def on_execute_flee(self, event_data: Dict[str, Any]):
        """Handles a flee action, which is mechanically similar to a move."""
        self._handle_movement(event_data, is_flee=True)

    def _handle_movement(self, event_data: Dict[str, Any], is_flee: bool):
        """Shared logic for all movement-based actions."""
        entity_id = event_data["entity_id"]
        action_plan = event_data["action_plan"]

        pos_comp = self.simulation_state.get_component(entity_id, PositionComponent)
        if not isinstance(pos_comp, PositionComponent):
            # Cannot move without a position
            return

        old_pos = pos_comp.position
        direction = action_plan.params.get("direction")
        if not isinstance(direction, int):
            # Invalid action parameters
            return

        env = self.simulation_state.environment
        if not env or not hasattr(env, "height") or not hasattr(env, "width"):
            # Environment not set up correctly
            return

        new_pos = _apply_move_logic(old_pos, direction, (env.height, env.width))

        # Update the component state
        pos_comp.position = new_pos
        pos_comp.history.append(new_pos)

        is_new_tile = new_pos not in pos_comp.visited_positions
        if is_new_tile:
            pos_comp.visited_positions.add(new_pos)

        # Determine outcome
        success = new_pos != old_pos
        base_reward = 0.0
        if success:
            status = "fled_successfully" if is_flee else "moved"
            if is_new_tile:
                # The RewardCalculator will add the subjective exploration bonus
                # This base reward is for the objective act of discovering new territory
                base_reward = self.config.get("learning", {}).get("rewards", {}).get("exploration_base_reward", 0.1)
        else:
            status = "movement_blocked"
            base_reward = -0.02  # Small penalty for bumping into a wall

        details = {
            "status": status,
            "old_pos": old_pos,
            "new_pos": new_pos,
            "explored_new_tile": is_new_tile,
        }
        message = f"Moved from {old_pos} to {new_pos}." if success else "Movement blocked."
        outcome = ActionOutcome(success, message, base_reward, details)

        self._publish_outcome(entity_id, action_plan, outcome, event_data["current_tick"])

    def _publish_outcome(self, entity_id: str, plan: Any, outcome: ActionOutcome, tick: int):
        """Helper to publish the action outcome to the event bus."""
        if self.event_bus:
            self.event_bus.publish(
                "action_outcome_ready",
                {
                    "entity_id": entity_id,
                    "action_outcome": outcome,
                    "original_action_plan": plan,
                    "current_tick": tick,
                },
            )

    async def update(self, current_tick: int) -> None:
        """This system is purely event-driven and does not have per-tick logic."""
        pass
