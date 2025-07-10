# FILE: agent-engine/src/agent_engine/logging/metrics_calculator_interface.py

from abc import ABC, abstractmethod
from typing import Any, Dict


class MetricsCalculatorInterface(ABC):
    """
    An interface for a class that calculates a specific set of metrics
    from the simulation state.
    """

    @abstractmethod
    def calculate_metrics(self, simulation_state: Any) -> Dict[str, Any]:
        """
        Calculates and returns a dictionary of metrics.

        Args:
            simulation_state: The current state of the entire simulation.

        Returns:
            A dictionary where keys are metric names and values are the
            calculated metric values.
        """
        raise NotImplementedError
