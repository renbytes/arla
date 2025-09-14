"""
Defines the metrics calculator for the Tragedy of the Commons simulation.
"""

from typing import Any, Dict, cast

from agent_core.core.ecs.component import TimeBudgetComponent
from agent_engine.logging.metrics_calculator_interface import MetricsCalculatorInterface

from ..components import EnergyComponent, ResourceComponent


class CommonsMetricsCalculator(MetricsCalculatorInterface):
    """
    Calculates simulation-wide metrics for the Tragedy of the Commons model.
    """

    def calculate_metrics(self, simulation_state: Any) -> Dict[str, Any]:
        """
        Calculates total resources in the commons and average agent energy.
        """
        # Calculate total resources
        resource_patches = simulation_state.get_entities_with_components(
            [ResourceComponent]
        )
        total_resources = sum(
            cast(ResourceComponent, comps.get(ResourceComponent)).current_resource
            for comps in resource_patches.values()
        )

        # Calculate average agent energy
        agents = simulation_state.get_entities_with_components(
            [EnergyComponent, TimeBudgetComponent]
        )
        total_energy = 0.0  # FIX: Initialize as a float
        active_agents_count = 0
        for comps in agents.values():
            time_comp = cast(TimeBudgetComponent, comps.get(TimeBudgetComponent))
            if time_comp.is_active:
                active_agents_count += 1
                energy_comp = cast(EnergyComponent, comps.get(EnergyComponent))
                if energy_comp:
                    total_energy += energy_comp.current_energy

        average_energy = (
            total_energy / active_agents_count if active_agents_count > 0 else 0.0
        )

        return {
            "active_agents": active_agents_count,
            "total_resources_in_commons": total_resources,
            "average_agent_energy": average_energy,
        }
