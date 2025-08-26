"""
Defines the metrics calculator for the Sugarscape simulation.

This class is responsible for calculating simulation-wide metrics at each
tick, such as population size, average energy, and counts of social
interactions. It implements the MetricsCalculatorInterface.
"""

from collections import defaultdict
from typing import Any, Dict

from agent_core.core.ecs.component import TimeBudgetComponent
from agent_engine.logging.metrics_calculator_interface import (
    MetricsCalculatorInterface,
)

from ..components import EnergyComponent
from ..environment import SugarscapeEnvironment


class SugarscapeMetricsCalculator(MetricsCalculatorInterface):
    """
    Calculates and aggregates key metrics for the Sugarscape simulation.
    """

    def __init__(self):
        # Use defaultdict to simplify counting
        self.action_counts = defaultdict(int)

    def calculate_metrics(self, simulation_state: Any) -> Dict[str, Any]:
        """
        Calculates the current state of all relevant metrics in the simulation.

        Args:
            simulation_state: The current state of the entire simulation.

        Returns:
            A dictionary containing the calculated metrics for the current tick.
        """
        all_agents = simulation_state.get_entities_with_components(
            [EnergyComponent, TimeBudgetComponent]
        )

        if not all_agents:
            return {
                "active_agents": 0,
                "average_agent_energy": 0.0,
                "total_sugar_in_env": 0.0,
            }

        total_energy = 0
        active_agents_count = 0

        for components in all_agents.values():
            time_comp = components.get(TimeBudgetComponent)
            if time_comp and time_comp.is_active:
                active_agents_count += 1
                energy_comp = components.get(EnergyComponent)
                if energy_comp:
                    total_energy += energy_comp.current_energy

        average_energy = (
            total_energy / active_agents_count if active_agents_count > 0 else 0.0
        )

        env = simulation_state.environment
        total_sugar = 0
        if isinstance(env, SugarscapeEnvironment):
            total_sugar = env.sugar_map.sum()

        metrics = {
            "active_agents": active_agents_count,
            "average_agent_energy": average_energy,
            "total_sugar_in_env": float(total_sugar),
        }
        # Add the counts of social actions for this tick
        metrics.update(self.action_counts)
        # Reset counts for the next tick
        self.action_counts.clear()

        return metrics

    def update_with_event(self, event_data: Dict[str, Any]):
        """
        Updates internal counters based on an 'action_executed' event.
        This is called by a tracker system that listens to the event bus.
        """
        action_plan = event_data.get("action_plan")
        if action_plan and hasattr(action_plan.action_type, "action_id"):
            action_id = action_plan.action_type.action_id
            if action_id in ["share", "attack", "reproduce"]:
                self.action_counts[f"{action_id}_count"] += 1
