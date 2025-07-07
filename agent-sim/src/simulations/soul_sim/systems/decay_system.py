# src/simulations/soul_sim/systems/decay_system.py
"""
Applies passive decay to agent vitals and handles inactivation from depletion.
"""
from typing import Any, Dict, List, Optional, Type, cast

from agent_core.core.ecs.component import Component, TimeBudgetComponent
from agent_engine.simulation.system import System

# Import world-specific components from the current simulation package
from ..components import HealthComponent, InventoryComponent, NestComponent


class DecaySystem(System):
    """
    Applies passive decay to agent vitals (health, time) each tick and
    handles entity inactivation when vitals are depleted.
    """
    # Define the components this system operates on in its update loop
    REQUIRED_COMPONENTS: List[Type[Component]] = [TimeBudgetComponent, HealthComponent]

    def update(self, current_tick: int):
        """
        Periodically decays vitals for all active entities and handles inactivation.
        """
        # Get decay rates from config for efficiency
        decay_config = self.config.get("agent", {}).get("dynamics", {}).get("decay", {})
        time_decay = decay_config.get("time_budget_per_step", 0.1)
        health_decay = decay_config.get("health_per_step", 0.0)
        resource_decay = decay_config.get("resources_per_step", 0.0)
        
        # Get all entities that have the required vitals to decay
        target_entities = self.simulation_state.get_entities_with_components(self.REQUIRED_COMPONENTS)

        for entity_id, components in target_entities.items():
            time_comp = cast(TimeBudgetComponent, components.get(TimeBudgetComponent))
            
            # Only process active agents
            if not time_comp.is_active:
                continue

            # --- Apply Decay ---
            health_comp = cast(HealthComponent, components.get(HealthComponent))
            
            time_comp.current_time_budget -= time_decay
            health_comp.current_health -= health_decay
            
            # Optional resource decay
            if isinstance(inv_comp := components.get(InventoryComponent), InventoryComponent):
                inv_comp.current_resources = max(0, inv_comp.current_resources - resource_decay)

            # --- Check for Inactivation ---
            if time_comp.current_time_budget <= 0 or health_comp.current_health <= 0:
                self._inactivate_entity(entity_id, components, current_tick)

    def _inactivate_entity(self, entity_id: str, components: Dict[Type[Component], Component], current_tick: int):
        """Helper function to handle the inactivation of an entity."""
        time_comp = cast(TimeBudgetComponent, components.get(TimeBudgetComponent))
        health_comp = cast(HealthComponent, components.get(HealthComponent))

        # Set vitals to zero to prevent negative values
        time_comp.current_time_budget = 0
        health_comp.current_health = 0
        time_comp.is_active = False
        
        reason = "time budget depletion" if time_comp.current_time_budget <= 0 else "health depletion"
        print(f"INFO: Entity {entity_id} became inactive due to {reason} at tick {current_tick}.")

        # Notify other systems that this entity is now inactive
        if self.event_bus:
            self.event_bus.publish(
                "entity_inactivated",
                {
                    "entity_id": entity_id,
                    "current_tick": current_tick,
                    "reason": reason
                },
            )