# FILE: agent-sim/src/agent_sim/infrastructure/logging/mlflow_exporter.py

import os
from typing import Any, Dict

import mlflow
import numpy as np
from agent_engine.logging.exporter_interface import ExporterInterface


class MLflowExporter(ExporterInterface):
    """
    An implementation of the ExporterInterface that logs to MLflow.
    It includes a "circuit breaker" to gracefully handle connection issues
    without crashing or slowing down the simulation.
    """

    def __init__(self, run_id: str, experiment_id: str):
        self.enabled = False
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

        if not tracking_uri:
            print("WARNING: MLflow logging disabled (MLFLOW_TRACKING_URI not set).")
            return

        try:
            mlflow.set_tracking_uri(tracking_uri)
            # This connects to an existing run created by the Celery/CLI orchestrator
            mlflow.start_run(run_id=run_id, experiment_id=experiment_id)
            self.enabled = True
            print(f"✅ MLflowExporter connected to tracking URI: {tracking_uri}")
        except Exception as e:
            print(f"WARNING: MLflow connection failed: {e}. Disabling MLflow logging for this run.")
            self.enabled = False

    async def export_metrics(self, tick: int, metrics: Dict[str, Any]) -> None:
        """Logs a dictionary of metrics to MLflow if enabled."""
        if not self.enabled:
            return
        try:
            # Filter out non-finite values which MLflow cannot log
            finite_metrics = {k: v for k, v in metrics.items() if np.isfinite(v)}
            if finite_metrics:
                mlflow.log_metrics(finite_metrics, step=tick)
        except Exception as e:
            print(f"WARNING: MLflow logging failed at tick {tick}: {e}. Disabling for remainder of run.")
            self.enabled = False  # Trip the circuit breaker

    async def log_learning_curve(self, tick: int, agent_id: str, q_loss: float) -> None:
        """Logs Q-loss to MLflow, prefixing with agent_id for clarity."""
        if not self.enabled:
            return
        try:
            metric_name = f"q_loss_{agent_id}"
            mlflow.log_metric(key=metric_name, value=q_loss, step=tick)
        except Exception as e:
            print(f"WARNING: MLflow Q-loss logging failed at tick {tick}: {e}. Disabling for remainder of run.")
            self.enabled = False  # Trip the circuit breaker

    async def log_event(self, event_data: Dict[str, Any]) -> None:
        """MLflow is not typically used for high-frequency event logging."""
        pass

    async def log_agent_state(self, tick: int, agent_id: str, components_data: Dict[str, Any]) -> None:
        """MLflow is not typically used for high-frequency agent state logging."""
        pass
