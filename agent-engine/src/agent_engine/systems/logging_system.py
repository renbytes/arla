# agent-engine/src/agent_engine/systems/logging_system.py

import importlib
from typing import Any, Dict, List, Type

from agent_core.core.ecs.component import Component, TimeBudgetComponent
from agent_core.core.ecs.event_bus import EventBus

from agent_engine.simulation.system import System

from ..logging.emitter_interface import MetricsEmitterInterface


class LoggingSystem(System):
    """
    Collects and aggregates high-level simulation metrics and passes them
    to a configured emitter. This system is completely decoupled from the
    final logging backend (DB, OpenTelemetry, etc.).
    """

    def __init__(
        self, simulation_state: Any, config: Dict[str, Any], cognitive_scaffold: Any, emitter: MetricsEmitterInterface
    ):
        super().__init__(simulation_state, config, cognitive_scaffold)
        self.emitter = emitter

        # FIX: Assert that event_bus is not None to satisfy mypy
        if not self.simulation_state.event_bus:
            raise ValueError("LoggingSystem requires an EventBus.")
        event_bus: EventBus = self.simulation_state.event_bus

        event_bus.subscribe("action_executed", self.on_action_executed)
        event_bus.subscribe("q_learning_update", self.on_q_learning_update)

    def _import_component_types(self) -> List[Type[Component]]:
        """Dynamically imports component classes from the paths in the config."""
        types = []
        for path in self.config.get("logging", {}).get("components_to_log", []):
            try:
                module_path, class_name = path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                types.append(getattr(module, class_name))
            except (ImportError, AttributeError, ValueError) as e:
                print(f"WARNING: LoggingSystem could not load component '{path}': {e}")
        return types

    async def update(self, current_tick: int) -> None:
        """Gathers stats and gives them to the emitter."""
        agent_states = self.simulation_state.get_entities_with_components([TimeBudgetComponent])
        metrics = self._summarise_tick(agent_states)
        await self.emitter.log_metrics(current_tick, metrics)

    async def on_action_executed(self, event_data: Dict[str, Any]) -> None:
        await self.emitter.log_event(event_data)

    async def on_q_learning_update(self, event_data: Dict[str, Any]) -> None:
        await self.emitter.log_learning_curve(
            tick=event_data["current_tick"],
            agent_id=event_data["entity_id"],
            q_loss=event_data["q_loss"],
        )

    def _summarise_tick(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        # Placeholder for your detailed aggregation logic.
        active_agents = len(
            [e for e, c in entities.items() if (time_comp := c.get(TimeBudgetComponent)) and time_comp.is_active]
        )
        return {"db_metrics": {"active_agents": active_agents}}
