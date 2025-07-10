# FILE: agent-engine/src/agent_engine/logging/emitter_interface.py

from abc import ABC, abstractmethod
from typing import Any, Dict


class MetricsEmitterInterface(ABC):
    """A generic interface for exporting simulation metrics and events."""

    @abstractmethod
    async def log_event(self, event_data: Dict[str, Any]) -> None:
        """Logs a single, discrete event."""
        raise NotImplementedError

    @abstractmethod
    async def log_metrics(self, tick: int, metrics: Dict[str, Any]) -> None:
        """Logs a dictionary of aggregated metrics for a given tick."""
        raise NotImplementedError

    @abstractmethod
    async def log_learning_curve(self, tick: int, agent_id: str, q_loss: float) -> None:
        """Logs a data point for an agent's learning curve."""
        raise NotImplementedError
