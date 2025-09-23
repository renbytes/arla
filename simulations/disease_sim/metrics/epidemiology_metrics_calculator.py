# simulations/disease_sim/metrics/epidemiology_metrics_calculator.py
"""
Defines the metrics calculator for the Disease simulation.
"""

from typing import Any, Dict, cast
from collections import Counter
from agent_engine.logging.metrics_calculator_interface import MetricsCalculatorInterface
from ..components import DiseaseStateComponent, DiseaseStateEnum


class EpidemiologyMetricsCalculator(MetricsCalculatorInterface):
    """
    Calculates simulation-wide metrics for the SEIR model.
    """

    def calculate_metrics(self, simulation_state: Any) -> Dict[str, Any]:
        """
        Calculates the total number of agents in each SEIR state.
        """
        all_agents = simulation_state.get_entities_with_components(
            [DiseaseStateComponent]
        )

        if not all_agents:
            return {
                "susceptible_total": 0,
                "exposed_total": 0,
                "infectious_total": 0,
                "removed_total": 0,
                "active_agents": 0,
            }

        state_counts = Counter(
            cast(DiseaseStateComponent, comps.get(DiseaseStateComponent)).state.value
            for comps in all_agents.values()
        )

        return {
            "susceptible_total": state_counts.get(
                DiseaseStateEnum.SUSCEPTIBLE.value, 0
            ),
            "exposed_total": state_counts.get(DiseaseStateEnum.EXPOSED.value, 0),
            "infectious_total": state_counts.get(DiseaseStateEnum.INFECTIOUS.value, 0),
            "removed_total": state_counts.get(DiseaseStateEnum.REMOVED.value, 0),
            "active_agents": len(all_agents),
        }
