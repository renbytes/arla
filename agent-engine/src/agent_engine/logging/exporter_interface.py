# FILE: agent-engine/src/agent_engine/logging/exporter_interface.py

from abc import ABC, abstractmethod
from typing import Any, Dict


class ExporterInterface(ABC):
    """
    A generic interface for exporting all types of simulation data, including
    events, aggregated metrics, and learning curves.
    """

    @abstractmethod
    async def export_metrics(self, tick: int, metrics: Dict[str, Any]) -> None:
        """Exports a dictionary of aggregated metrics for a given tick."""
        raise NotImplementedError

    @abstractmethod
    async def log_event(self, event_data: Dict[str, Any]) -> None:
        """Logs a single, discrete event from the simulation."""
        raise NotImplementedError

    @abstractmethod
    async def log_agent_state(self, tick: int, agent_id: str, components_data: Dict[str, Any]) -> None:
        """Logs a snapshot of an agent's components for a given tick."""
        raise NotImplementedError

    @abstractmethod
    async def log_learning_curve(self, tick: int, agent_id: str, q_loss: float) -> None:
        """Logs a data point for an agent's learning curve."""
        raise NotImplementedError
