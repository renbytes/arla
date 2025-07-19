# FILE: simulations/emergence_sim/systems/movement_system.py
from typing import Any, Dict, List, Type
from unittest.mock import MagicMock

from agent_core.agents.actions.base_action import ActionOutcome
from agent_core.core.ecs.component import ActionPlanComponent, Component
from agent_engine.simulation.system import System

from simulations.emergence_sim.components import InventoryComponent, PositionComponent


class MovementSystem(System):
    """Processes move actions, updates positions, and handles all object interactions."""

    REQUIRED_COMPONENTS: List[Type[Component]] = []  # Event-driven

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.event_bus:
            self.event_bus.subscribe("execute_move_action", self.on_execute_move)

    def on_execute_move(self, event_data: Dict[str, Any]):
        entity_id = event_data["entity_id"]
        action_plan = event_data["action_plan_component"]
        pos_comp = self.simulation_state.get_component(entity_id, PositionComponent)
        inv_comp = self.simulation_state.get_component(entity_id, InventoryComponent)

        if not pos_comp or not inv_comp or "new_pos" not in action_plan.params:
            return

        old_pos = pos_comp.position
        new_pos = action_plan.params["new_pos"]

        pos_comp.position = new_pos
        self.simulation_state.environment.update_entity_position(entity_id, old_pos, new_pos)

        base_reward = -0.1
        message = f"Moved from {old_pos} to {new_pos}."

        world_object = self.simulation_state.environment.get_object_at(new_pos)
        if world_object:
            obj_type = world_object.get("obj_type")
            value = world_object.get("value", 0)

            # CORRECTED: Cooperative harvesting must reward ALL participants.
            if obj_type == "cooperative_resource":
                nearby_entities = self.simulation_state.environment.get_entities_in_radius(pos_comp.position, 1)
                if len(nearby_entities) > 1:
                    # Grant resources and a large reward to everyone involved
                    for agent_tuple in nearby_entities:
                        agent_id = agent_tuple[0]
                        coop_inv_comp = self.simulation_state.get_component(agent_id, InventoryComponent)
                        if coop_inv_comp:
                            coop_inv_comp.current_resources += value
                            bonus_outcome = ActionOutcome(
                                True,
                                f"Cooperatived on resource for +{value} resources.",
                                15.0,
                                {},
                            )
                            # Use a dummy ActionPlan for the helper agents
                            dummy_plan = ActionPlanComponent(action_type=MagicMock(action_id="cooperate"))
                            self._publish_outcome(
                                agent_id,
                                dummy_plan,
                                bonus_outcome,
                                event_data["current_tick"],
                            )
                    del self.simulation_state.environment.objects[world_object["id"]]
                    return  # End the turn here as a special event

            elif obj_type == "hazard":
                inv_comp.current_resources = max(0, inv_comp.current_resources - value)
                base_reward = -10.0
                message += f" Lost {value} resources to a hazard."
                del self.simulation_state.environment.objects[world_object["id"]]
            elif obj_type == "resource":
                inv_comp.current_resources += value
                base_reward = 5.0
                message += f" Gained {value} resources."
                del self.simulation_state.environment.objects[world_object["id"]]

        outcome = ActionOutcome(True, message, base_reward, {"new_pos": new_pos})
        self._publish_outcome(entity_id, action_plan, outcome, event_data["current_tick"])

    def _publish_outcome(self, entity_id: str, plan: Any, outcome: ActionOutcome, tick: int):
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
        pass
