# FILE: simulations/emergence_sim/systems/object_interaction_system.py
from typing import List, Type, cast

from agent_engine.simulation.system import System

from simulations.emergence_sim.components import InventoryComponent, PositionComponent


class ObjectInteractionSystem(System):
    """Handles the consequences of agents interacting with world objects."""

    REQUIRED_COMPONENTS: List[Type] = [PositionComponent, InventoryComponent]

    async def update(self, current_tick: int) -> None:
        if not self.simulation_state.environment:
            return

        all_agents = self.simulation_state.get_entities_with_components(self.REQUIRED_COMPONENTS)

        for agent_id, components in all_agents.items():
            pos_comp = cast(PositionComponent, components.get(PositionComponent))
            inv_comp = cast(InventoryComponent, components.get(InventoryComponent))

            world_object = self.simulation_state.environment.get_object_at(pos_comp.position)

            if world_object:
                obj_type = world_object.get("obj_type")
                value = world_object.get("value", 0)

                if obj_type == "cooperative_resource":
                    # Check for at least one other agent nearby (radius of 1)
                    nearby_entities = self.simulation_state.environment.get_entities_in_radius(pos_comp.position, 1)
                    if len(nearby_entities) > 1:
                        inv_comp.current_resources += value
                        print(f"CO-OP! Agent {agent_id} and others harvested a resource for {value}.")
                        del self.simulation_state.environment.objects[world_object["id"]]
                elif obj_type == "hazard":
                    inv_comp.current_resources = max(0, inv_comp.current_resources - value)
                    print(f"ZAP! Agent {agent_id} lost {value} resources.")
                    del self.simulation_state.environment.objects[world_object["id"]]
                elif obj_type == "resource":
                    inv_comp.current_resources += value
                    print(f"NICE! Agent {agent_id} gained {value} resources.")
                    del self.simulation_state.environment.objects[world_object["id"]]
