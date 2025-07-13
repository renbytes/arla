# src/agent_engine/systems/decay_system.py
"""
Applies passive decay to agent vitals and handles inactivation from depletion.
"""

from typing import Dict, List, Type, cast

from agent_core.core.ecs.component import Component, TimeBudgetComponent
from agent_engine.simulation.system import System

from simulations.soul_sim.components import HealthComponent, InventoryComponent


class DecaySystem(System):
    """
    Applies passive decay to agent vitals (health, time) each tick and
    handles entity inactivation when vitals are depleted.
    """

    REQUIRED_COMPONENTS: List[Type[Component]] = [TimeBudgetComponent, HealthComponent]

    async def update(self, current_tick: int) -> None:
        """
        Periodically decays vitals for all active entities and handles inactivation.
        """
        time_decay = self.config.agent.dynamics.decay.time_budget_per_step
        health_decay = self.config.agent.dynamics.decay.health_per_step
        resource_decay = self.config.agent.dynamics.decay.resources_per_step

        target_entities = self.simulation_state.get_entities_with_components(self.REQUIRED_COMPONENTS)

        for entity_id, components in target_entities.items():
            time_comp = cast(TimeBudgetComponent, components.get(TimeBudgetComponent))

            if not time_comp.is_active:
                continue

            time_comp.current_time_budget -= time_decay
            cast(HealthComponent, components.get(HealthComponent)).current_health -= health_decay

            if isinstance(inv_comp := components.get(InventoryComponent), InventoryComponent):
                inv_comp.current_resources = max(0, inv_comp.current_resources - resource_decay)

            if (
                time_comp.current_time_budget <= 0
                or cast(HealthComponent, components.get(HealthComponent)).current_health <= 0
            ):
                self._inactivate_entity(entity_id, components, current_tick)

    def _inactivate_entity(
        self,
        entity_id: str,
        components: Dict[Type[Component], Component],
        current_tick: int,
    ):
        """Helper function to handle the inactivation of an entity."""
        time_comp = cast(TimeBudgetComponent, components.get(TimeBudgetComponent))
        health_comp = cast(HealthComponent, components.get(HealthComponent))

        # --- FIX: Determine the reason BEFORE modifying the state ---
        reason = "health depletion"
        if time_comp.current_time_budget <= 0:
            reason = "time budget depletion"

        # Now, modify the state
        time_comp.current_time_budget = 0
        health_comp.current_health = 0
        time_comp.is_active = False

        print(f"INFO: Entity {entity_id} became inactive due to {reason} at tick {current_tick}.")

        if self.event_bus:
            self.event_bus.publish(
                "entity_inactivated",
                {
                    "entity_id": entity_id,
                    "current_tick": current_tick,
                    "reason": reason,
                },
            )
