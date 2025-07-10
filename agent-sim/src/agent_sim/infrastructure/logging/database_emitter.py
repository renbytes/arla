# FILE: agent-sim/src/agent_sim/infrastructure/logging/database_emitter.py

import uuid
from typing import Any, Dict

from agent_engine.logging.metrics_exporter_interface import MetricsExporterInterface
from agent_sim.infrastructure.database.async_database_manager import AsyncDatabaseManager


class DatabaseEmitter(MetricsExporterInterface):
    """An emitter that logs all data to a relational database."""

    def __init__(self, db_manager: AsyncDatabaseManager, simulation_id: uuid.UUID):
        self.db_manager = db_manager
        self.simulation_id = simulation_id

    async def log_event(self, event_data: Dict[str, Any]) -> None:
        action_plan = event_data.get("action_plan_component")
        action_outcome = event_data.get("action_outcome")
        if not action_plan or not action_outcome:
            return

        await self.db_manager.log_event(
            simulation_id=self.simulation_id,
            tick=event_data["current_tick"],
            agent_id=event_data["entity_id"],
            action_type=getattr(action_plan.action_type, "name", "Unknown"),
            success=action_outcome.success,
            reward=action_outcome.reward,
            message=action_outcome.message,
            details=action_outcome.details,
        )

    async def export_metrics(self, tick: int, metrics: Dict[str, Any]) -> None:
        """
        Filters the incoming metrics dictionary to only include keys that
        match columns in the 'metrics' database table, then logs them.
        """
        # Filter the unified metrics dictionary for what the DB table can accept
        db_metrics = {key: value for key, value in metrics.items() if key in self.valid_metric_columns}

        if db_metrics:
            await self.db_manager.log_metrics(simulation_id=self.simulation_id, tick=tick, metrics_data=db_metrics)

    async def log_learning_curve(self, tick: int, agent_id: str, q_loss: float) -> None:
        await self.db_manager.log_learning_curve(
            simulation_id=self.simulation_id, tick=tick, agent_id=agent_id, q_loss=q_loss
        )
