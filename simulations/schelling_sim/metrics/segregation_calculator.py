# simulations/schelling_sim/metrics/segregation_calculator.py

from typing import Any, Dict

from agent_engine.logging.metrics_calculator_interface import MetricsCalculatorInterface

from ..components import GroupComponent, SatisfactionComponent


class SegregationCalculator(MetricsCalculatorInterface):
    """
    Calculates simulation-wide metrics for the Schelling model.
    """

    def calculate_metrics(self, simulation_state: Any) -> Dict[str, Any]:
        """
        Calculates the overall satisfaction rate and the segregation index.

        Args:
            simulation_state: The current state of the entire simulation.

        Returns:
            A dictionary containing the calculated metrics.
        """
        all_agents = simulation_state.get_entities_with_components([GroupComponent, SatisfactionComponent])

        if not all_agents:
            return {
                "satisfaction_rate": 1.0,
                "segregation_index": 0.0,
                "active_agents": 0,
            }

        satisfied_count = 0
        group_1_count = 0
        group_2_count = 0

        for components in all_agents.values():
            satisfaction_comp = components.get(SatisfactionComponent)
            group_comp = components.get(GroupComponent)

            if satisfaction_comp and satisfaction_comp.is_satisfied:
                satisfied_count += 1

            if group_comp:
                if group_comp.agent_type == 1:
                    group_1_count += 1
                else:
                    group_2_count += 1

        # Calculate the segregation index
        total_agents = len(all_agents)
        p1 = group_1_count / total_agents if total_agents > 0 else 0
        p2 = group_2_count / total_agents if total_agents > 0 else 0
        segregation_index = 1 - (satisfied_count / total_agents) * (2 * p1 * p2) if total_agents > 0 else 0

        return {
            "satisfaction_rate": satisfied_count / total_agents if total_agents > 0 else 1.0,
            "segregation_index": segregation_index,
            "active_agents": total_agents,
        }
