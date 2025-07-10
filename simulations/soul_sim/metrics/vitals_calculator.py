# FILE: simulations/soul_sim/metrics/vitals_calculator.py

from collections import defaultdict
from typing import Any, Dict, List, Type

import numpy as np
from agent_core.core.ecs.component import Component, TimeBudgetComponent
from agent_engine.logging.metrics_calculator_interface import MetricsCalculatorInterface
from simulations.soul_sim.components import HealthComponent, InventoryComponent


class VitalsAndEconomyCalculator(MetricsCalculatorInterface):
    """
    A soul-sim specific calculator for agent vitals and basic economy metrics.
    """

    def calculate_metrics(self, simulation_state: Any) -> Dict[str, Any]:
        """
        Calculates average health, time budget, total resources, and Gini coefficient.
        """
        stats = defaultdict(list)
        active_agents = 0

        entities = simulation_state.get_entities_with_components([TimeBudgetComponent])

        for comps in entities.values():
            time_comp = comps.get(TimeBudgetComponent)
            if not (time_comp and time_comp.is_active):
                continue

            active_agents += 1
            if health := comps.get(HealthComponent):
                stats["health"].append(health.current_health)
            if inventory := comps.get(InventoryComponent):
                stats["resources"].append(inventory.current_resources)
            stats["time_budget"].append(time_comp.current_time_budget)

        gini = self._calculate_gini(np.array(stats["resources"])) if stats["resources"] else 0.0

        return {
            "vitals_active_agents": float(active_agents),
            "vitals_avg_health": np.mean(stats["health"]) if stats["health"] else 0.0,
            "vitals_avg_time_budget": np.mean(stats["time_budget"]) if stats["time_budget"] else 0.0,
            "economy_total_resources": np.sum(stats["resources"]),
            "economy_gini_coefficient": float(gini),
        }

    def _calculate_gini(self, x: np.ndarray) -> float:
        """Calculates the Gini coefficient of a numpy array."""
        if x.size < 2 or np.sum(x) == 0:
            return 0.0
        x_sorted = np.sort(x)
        n = x.size
        cumx = np.cumsum(x_sorted, dtype=float)
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
