# src/agent_engine/simulation/simulation_state.py
"""
Consolidates all simulation data, acting as the central state container.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

import numpy as np
from agent_core.core.ecs.abstractions import AbstractSimulationState
from agent_core.core.ecs.component import (
    Component,
)
from agent_core.core.ecs.component_factory_interface import ComponentFactoryInterface
from agent_persist.models import AgentSnapshot, ComponentSnapshot, SimulationSnapshot

# Forward references for type hinting to avoid circular imports
if TYPE_CHECKING:
    from agent_core.cognition.scaffolding import CognitiveScaffold
    from agent_core.core.ecs.event_bus import EventBus
    from agent_core.environment.interface import EnvironmentInterface

    from agent_engine.simulation.system import SystemManager


class SimulationState(AbstractSimulationState):
    """
    Consolidates all simulation data into a single, world-agnostic container.
    """

    def __init__(self, config: Any, device: Any) -> None:
        self.config = config
        self.device = device
        self.entities: Dict[str, Dict[Type[Component], Component]] = {}
        self.simulation_id: str = ""
        self.environment: Optional["EnvironmentInterface"] = None
        self._event_bus: Optional["EventBus"] = None
        self.system_manager: Optional["SystemManager"] = None
        self.cognitive_scaffold: Optional["CognitiveScaffold"] = None
        self.main_rng: Optional[np.random.Generator] = None
        self.current_tick: int = 0
        self.db_logger: Optional[Any] = None

    @property
    def event_bus(self) -> Optional["EventBus"]:
        """Provides access to the simulation's event bus."""
        return self._event_bus

    @event_bus.setter
    def event_bus(self, value: Optional["EventBus"]) -> None:
        self._event_bus = value

    @classmethod
    def from_snapshot(
        cls,
        snapshot: SimulationSnapshot,
        config: Any,
        component_factory: ComponentFactoryInterface,
        environment: "EnvironmentInterface",
        event_bus: "EventBus",
        db_logger: Any,
    ) -> "SimulationState":
        """Creates a new SimulationState instance from a snapshot."""
        # Initialize a new state object
        sim_state = cls(config, "cpu")
        sim_state.current_tick = snapshot.current_tick
        sim_state.simulation_id = snapshot.simulation_id
        sim_state.environment = environment
        sim_state.event_bus = event_bus
        sim_state.db_logger = db_logger

        # Restore environment state first
        if snapshot.environment_state and sim_state.environment:
            sim_state.environment.restore_from_dict(snapshot.environment_state)

        # Restore agents and their components using the factory
        for agent_snapshot in snapshot.agents:
            agent_id = agent_snapshot.agent_id
            sim_state.add_entity(agent_id)
            for comp_snapshot in agent_snapshot.components:
                try:
                    # The factory handles the creation of all components
                    component_instance = component_factory.create_component(
                        comp_snapshot.component_type, comp_snapshot.data
                    )
                    sim_state.add_component(agent_id, component_instance)
                except Exception as e:
                    print(f"Could not restore component {comp_snapshot.component_type} for agent {agent_id}: {e}")

        return sim_state

    def to_snapshot(self) -> SimulationSnapshot:
        """
        Converts the live SimulationState object into a serializable SimulationSnapshot model.
        """
        agent_snapshots = []
        for agent_id, components in self.entities.items():
            component_snapshots = []
            for component in components.values():
                comp_snapshot = ComponentSnapshot(
                    component_type=f"{component.__class__.__module__}.{component.__class__.__name__}",
                    data=component.to_dict(),
                )
                component_snapshots.append(comp_snapshot)

            agent_snapshots.append(AgentSnapshot(agent_id=agent_id, components=component_snapshots))

        return SimulationSnapshot(
            simulation_id=self.simulation_id,
            current_tick=self.current_tick,
            agents=agent_snapshots,
            environment_state=(self.environment.to_dict() if self.environment else None),
        )

    def add_entity(self, entity_id: str) -> None:
        if entity_id in self.entities:
            raise ValueError(f"Entity with ID {entity_id} already exists.")
        self.entities[entity_id] = {}

    def add_component(self, entity_id: str, component: Component) -> None:
        if entity_id not in self.entities:
            raise ValueError(f"Cannot add component. Entity with ID {entity_id} does not exist.")
        self.entities[entity_id][type(component)] = component

    def get_component(self, entity_id: str, component_type: Type[Component]) -> Optional[Component]:
        return self.entities.get(entity_id, {}).get(component_type)

    def remove_entity(self, entity_id: str) -> None:
        if entity_id in self.entities:
            del self.entities[entity_id]

    def get_entities_with_components(
        self, component_types: List[Type[Component]]
    ) -> Dict[str, Dict[Type[Component], Component]]:
        matching_entities = {}
        for entity_id, components in self.entities.items():
            if all(comp_type in components for comp_type in component_types):
                matching_entities[entity_id] = components
        return matching_entities
