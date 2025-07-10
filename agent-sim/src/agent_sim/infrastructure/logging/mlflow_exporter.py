# FILE: agent-sim/src/agent_sim/infrastructure/logging/mlflow_exporter.py

from typing import Any, Dict

import mlflow
import numpy as np
from agent_engine.logging.exporter_interface import ExporterInterface


class MLflowExporter(ExporterInterface):
    """
    An implementation of the ExporterInterface that logs metrics
    to an active MLflow run.
    """

    async def export_metrics(self, tick: int, metrics: Dict[str, float]) -> None:
        """Logs a dictionary of metrics to MLflow."""
        try:
            finite_metrics = {k: v for k, v in metrics.items() if np.isfinite(v)}
            if finite_metrics:
                mlflow.log_metrics(finite_metrics, step=tick)
        except Exception as e:
            print(f"Warning: MLflow logging failed at tick {tick}: {e}")

    async def log_event(self, event_data: Dict[str, Any]) -> None:
        """MLflow is not used for high-frequency event logging. Pass."""
        pass

    async def log_agent_state(self, tick: int, agent_id: str, components_data: Dict[str, Any]) -> None:
        """MLflow is not used for high-frequency agent state logging."""
        pass

    async def log_learning_curve(self, tick: int, agent_id: str, q_loss: float) -> None:
        """Logs Q-loss to MLflow, prefixing with agent_id for clarity."""
        try:
            # MLflow works best with flat metric names
            metric_name = f"q_loss_{agent_id}"
            mlflow.log_metric(key=metric_name, value=q_loss, step=tick)
        except Exception as e:
            print(f"Warning: MLflow Q-loss logging failed at tick {tick}: {e}")
