# FILE: simulations/emergence_sim/systems/decay_system.py
"""
Applies passive resource decay to agents to create economic pressure.
"""

from typing import List, Type, cast

from agent_core.core.ecs.component import Component, TimeBudgetComponent
from agent_engine.simulation.system import System

from simulations.emergence_sim.components import InventoryComponent


class DecaySystem(System):
    """
    Applies a passive decay to agent resources each tick to create a need
    for economic interaction.
    """

    REQUIRED_COMPONENTS: List[Type[Component]] = [
        TimeBudgetComponent,
        InventoryComponent,
    ]

    async def update(self, current_tick: int) -> None:
        """Periodically decays resources for all active entities."""

        resource_decay = self.config.agent.dynamics.decay.resources_per_step

        target_entities = self.simulation_state.get_entities_with_components(self.REQUIRED_COMPONENTS)

        for _entity_id, components in target_entities.items():
            time_comp = cast(TimeBudgetComponent, components.get(TimeBudgetComponent))

            if not time_comp.is_active:
                continue

            if isinstance(inv_comp := components.get(InventoryComponent), InventoryComponent):
                inv_comp.current_resources = max(0, inv_comp.current_resources - resource_decay)
