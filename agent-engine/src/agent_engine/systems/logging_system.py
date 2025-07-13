# FILE: agent-engine/src/agent_engine/systems/logging_system.py

import importlib
from typing import Any, Dict, List, Type

from agent_core.core.ecs.component import Component, TimeBudgetComponent
from agent_core.core.ecs.event_bus import EventBus

from agent_engine.logging.exporter_interface import ExporterInterface
from agent_engine.simulation.system import System


class LoggingSystem(System):
    """
    A system dedicated to logging simulation data via a decoupled emitter.
    """

    def __init__(
        self,
        simulation_state: Any,
        config: Any,
        cognitive_scaffold: Any,
        exporters: List[ExporterInterface],
    ):
        super().__init__(simulation_state, config, cognitive_scaffold)
        self.exporters = exporters

        if not self.simulation_state.event_bus:
            raise ValueError("LoggingSystem requires an EventBus.")
        event_bus: EventBus = self.simulation_state.event_bus

        event_bus.subscribe("action_executed", self.on_action_executed)
        event_bus.subscribe("q_learning_update", self.on_q_learning_update)

        # Use direct attribute access for the 'logging' section,
        # which is a dictionary in our Pydantic model.
        self.component_paths_to_log = self.config.logging.get("components_to_log", [])
        self.component_types_to_log: List[Type[Component]] = self._import_component_types()

    def _import_component_types(self) -> List[Type[Component]]:
        """Dynamically imports component classes from the paths in the config."""
        types = []
        for path in self.component_paths_to_log:
            try:
                module_path, class_name = path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                types.append(getattr(module, class_name))
            except (ImportError, AttributeError, ValueError) as e:
                print(f"WARNING: LoggingSystem could not load component '{path}': {e}")
        return types

    async def update(self, current_tick: int) -> None:
        """Logs the state of all agents each tick to all exporters."""
        target_entities = self.simulation_state.get_entities_with_components([TimeBudgetComponent])

        for agent_id, components in target_entities.items():
            state_data = self._collect_state_data(components)
            if not state_data:
                continue

            for exporter in self.exporters:
                await exporter.log_agent_state(
                    tick=current_tick,
                    agent_id=agent_id,
                    components_data=state_data,
                )

    def _collect_state_data(self, components: Dict[Type[Component], Component]) -> Dict[str, Any]:
        """Gathers data from components specified in the config."""
        data = {}
        for comp_type in self.component_types_to_log:
            if comp_instance := components.get(comp_type):
                data[comp_type.__name__] = comp_instance.to_dict()
        return data

    async def on_action_executed(self, event_data: Dict[str, Any]) -> None:
        for exporter in self.exporters:
            await exporter.log_event(event_data)

    async def on_q_learning_update(self, event_data: Dict[str, Any]) -> None:
        for exporter in self.exporters:
            await exporter.log_learning_curve(
                tick=event_data["current_tick"],
                agent_id=event_data["entity_id"],
                q_loss=event_data["q_loss"],
            )
