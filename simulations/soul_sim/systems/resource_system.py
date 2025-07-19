# src/simulations/soul_sim/systems/resource_system.py
"""
Manages all resource-related actions, including mining, collaboration,
and resource respawning.
"""

from typing import Any, Dict, List, Type, cast

from agent_core.agents.actions.base_action import ActionOutcome
from agent_core.core.ecs.component import Component
from agent_engine.simulation.system import System

# Import world-specific components from the current simulation package
from ..components import InventoryComponent, PositionComponent, ResourceComponent


class ResourceSystem(System):
    """
    Processes resource extraction actions and manages the lifecycle of resources.
    """

    REQUIRED_COMPONENTS: List[Type[Component]] = [ResourceComponent]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.event_bus:
            self.event_bus.subscribe("execute_extract_action", self.on_execute_extract)

    def on_execute_extract(self, event_data: Dict[str, Any]):
        """Handles a resource extraction action event."""
        entity_id = event_data["entity_id"]
        action_plan = event_data["action_plan_component"]
        resource_id = action_plan.params.get("resource_id")

        if not isinstance(resource_id, str):
            return

        # 1. Validate Target and Miner
        resource_comps = self.simulation_state.entities.get(resource_id, {})
        miner_comps = self.simulation_state.entities.get(entity_id, {})

        res_comp = resource_comps.get(ResourceComponent)
        res_pos_comp = resource_comps.get(PositionComponent)
        miner_inv_comp = miner_comps.get(InventoryComponent)
        miner_pos_comp = miner_comps.get(PositionComponent)

        if not all([isinstance(c, Component) for c in [res_comp, res_pos_comp, miner_inv_comp, miner_pos_comp]]):
            return

        # Ensure types for mypy
        res_comp = cast(ResourceComponent, res_comp)
        res_pos_comp = cast(PositionComponent, res_pos_comp)
        miner_inv_comp = cast(InventoryComponent, miner_inv_comp)
        miner_pos_comp = cast(PositionComponent, miner_pos_comp)

        if res_comp.is_depleted or res_pos_comp.position != miner_pos_comp.position:
            outcome = ActionOutcome(
                False,
                "Resource is depleted or not at the correct location.",
                -0.05,
                {"status": "invalid_mine_target"},
            )
            self._publish_outcome(entity_id, action_plan, outcome, event_data["current_tick"])
            return

        # 2. Resolve Mining Action
        res_comp.current_health -= res_comp.mining_rate
        base_reward = res_comp.reward_per_mine_action

        miner_inv_comp.current_resources += base_reward

        was_depleted = res_comp.current_health <= 0
        if was_depleted:
            res_comp.is_depleted = True
            res_comp.current_health = 0
            # Add the final yield bonus upon depletion
            base_reward += res_comp.resource_yield
            miner_inv_comp.current_resources += res_comp.resource_yield

        # 3. Create and Publish Outcome
        status = "resource_depleted" if was_depleted else "mining_progress"
        message = f"Depleted resource {resource_id}!" if was_depleted else f"Mined resource {resource_id}."
        details = {
            "status": status,
            "resource_id": resource_id,
            "resource_type": res_comp.type,
        }
        outcome = ActionOutcome(True, message, base_reward, details)

        self._publish_outcome(entity_id, action_plan, outcome, event_data["current_tick"])

    async def update(self, current_tick: int) -> None:
        """Periodically checks and respawns depleted resources."""
        if current_tick % 10 != 0:  # Check every 10 ticks
            return

        # This gets all entities that have a ResourceComponent
        all_resources = self.simulation_state.get_entities_with_components(self.REQUIRED_COMPONENTS)

        for res_id, comps in all_resources.items():
            res_comp = cast(ResourceComponent, comps.get(ResourceComponent))
            if res_comp.is_depleted:
                res_comp.depleted_timer += 10  # Add the check interval
                if res_comp.depleted_timer >= res_comp.resource_respawn_time:
                    res_comp.is_depleted = False
                    res_comp.current_health = res_comp.initial_health
                    res_comp.depleted_timer = 0
                    print(f"INFO: Resource {res_id} has respawned at tick {current_tick}.")

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
