# FILE: agent-engine/src/agent_engine/systems/perception_system.py

from typing import Any, List, Type, cast

from agent_core.core.ecs.component import Component, PerceptionComponent
from agent_core.environment.perception_provider_interface import (
    PerceptionProviderInterface,
)

from agent_engine.simulation.simulation_state import SimulationState
from agent_engine.simulation.system import System


class PerceptionSystem(System):
    """
    A world-agnostic system that updates agents' sensory information.

    This system iterates through all entities with a PerceptionComponent and
    uses a dependency-injected PerceptionProvider to populate it with
    world-specific data.
    """

    REQUIRED_COMPONENTS: List[Type[Component]] = [PerceptionComponent]

    def __init__(
        self,
        simulation_state: SimulationState,
        config: Any,
        cognitive_scaffold: Any,
        perception_provider: PerceptionProviderInterface,
    ):
        """
        Initializes the PerceptionSystem.

        Args:
            simulation_state: The main state object of the simulation.
            config: The simulation configuration object.
            cognitive_scaffold: The interface for LLM interactions.
            perception_provider: A concrete implementation of the
                                 PerceptionProviderInterface that contains
                                 the world-specific logic for sensing.
        """
        super().__init__(simulation_state, config, cognitive_scaffold)
        self.perception_provider = perception_provider

    async def update(self, current_tick: int) -> None:
        """
        On each tick, update the perception for all entities that can see.
        """
        entities_that_perceive = self.simulation_state.get_entities_with_components(
            self.REQUIRED_COMPONENTS
        )

        for entity_id, components in entities_that_perceive.items():
            perception_comp = cast(
                PerceptionComponent, components.get(PerceptionComponent)
            )
            if perception_comp:
                # Delegate the actual "seeing" logic to the provider
                self.perception_provider.update_perception(
                    entity_id, components, self.simulation_state, current_tick
                )
