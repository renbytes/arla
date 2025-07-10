# FILE: agent-sim/src/agent_sim/infrastructure/logging/mlflow_exporter.py

from typing import Dict

import mlflow
import numpy as np
from agent_engine.logging.metrics_exporter_interface import MetricsExporterInterface


class MLflowExporter(MetricsExporterInterface):
    """
    An implementation of the MetricsExporterInterface that logs metrics
    to an active MLflow run.
    """

    async def export_metrics(self, tick: int, metrics: Dict[str, float]) -> None:
        """
        Logs a dictionary of metrics to MLflow, ensuring all values are finite.

        Args:
            tick: The current simulation tick, used as the step for logging.
            metrics: The dictionary of aggregated metrics to log.
        """
        try:
            # Filter out any non-finite values (NaN, infinity) that would crash MLflow
            finite_metrics = {k: v for k, v in metrics.items() if np.isfinite(v)}
            if finite_metrics:
                mlflow.log_metrics(finite_metrics, step=tick)
        except Exception as e:
            # This provides a safeguard if MLflow calls fail for any reason
            print(f"Warning: MLflow logging failed at tick {tick}: {e}")
