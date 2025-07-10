# FILE: agent-engine/src/agent_engine/logging/metrics_calculator_interface.py

from abc import ABC, abstractmethod
from typing import Dict


class MetricsExporterInterface(ABC):
    """
    A generic interface for exporting aggregated simulation metrics to a
    monitoring backend like MLflow, Prometheus, etc.
    """

    @abstractmethod
    async def export_metrics(self, tick: int, metrics: Dict[str, float]) -> None:
        """
        Exports a dictionary of aggregated metrics for a given simulation tick.

        Args:
            tick: The current simulation tick.
            metrics: A dictionary of aggregated metric names and their values.
        """
        raise NotImplementedError
