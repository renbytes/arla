# src/simulations/soul_sim/systems/nest_system.py
"""
Handles the creation of nests, which can provide benefits to agents.
"""

from typing import Any, Dict, List, Type

from agent_core.agents.actions.base_action import ActionOutcome
from agent_core.core.ecs.component import Component
from agent_engine.simulation.system import System

# Import world-specific components from the current simulation package
from ..components import InventoryComponent, NestComponent, PositionComponent


class NestSystem(System):
    """
    Processes nest creation actions, checking for required resources and
    updating the agent's nest locations.
    """

    REQUIRED_COMPONENTS: List[Type[Component]] = []  # Event-driven

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.event_bus:
            self.event_bus.subscribe("execute_nest_action", self.on_execute_nest_action)

    def on_execute_nest_action(self, event_data: Dict[str, Any]):
        """Handles a nest creation action event."""
        entity_id = event_data["entity_id"]
        action_plan = event_data["action_plan_component"]

        # 1. Get required components
        inv_comp = self.simulation_state.get_component(entity_id, InventoryComponent)
        pos_comp = self.simulation_state.get_component(entity_id, PositionComponent)
        nest_comp = self.simulation_state.get_component(entity_id, NestComponent)

        if (
            not isinstance(inv_comp, InventoryComponent)
            or not isinstance(pos_comp, PositionComponent)
            or not isinstance(nest_comp, NestComponent)
        ):
            return

        # 2. Resolve Nest Creation
        # Use direct attribute access on the validated Pydantic model
        nest_cost = self.config.learning.rewards.nest_resource_cost

        if inv_comp.current_resources >= nest_cost:
            inv_comp.current_resources -= nest_cost
            nest_comp.locations.append(pos_comp.position)

            # NOTE: 'nest_creation_bonus' is not in the config schema. Using a hardcoded default.
            base_reward = 5.0
            success = True
            message = f"Successfully built a nest at {pos_comp.position}."
            details = {
                "status": "nest_built",
                "location": pos_comp.position,
                "total_nests": len(nest_comp.locations),
            }
        else:
            base_reward = -0.1
            success = False
            message = "Failed to build nest: not enough resources."
            details = {"status": "failed_insufficient_resources"}

        # 3. Publish Outcome
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
        """This system is purely event-driven."""
        pass
