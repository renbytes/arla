# FILE: agent-engine/src/agent_engine/systems/metrics_system.py

from typing import Any, Dict, List

from agent_engine.logging.metrics_calculator_interface import MetricsCalculatorInterface
from agent_engine.logging.metrics_exporter_interface import MetricsExporterInterface
from agent_engine.simulation.system import System


class MetricsSystem(System):
    """
    A generic system that orchestrates the calculation and exporting of
    metrics. It iterates through a list of injected 'MetricCalculator'
    objects and passes their results to a list of injected 'MetricsExporter'
    objects.
    """

    def __init__(
        self,
        simulation_state: Any,
        config: Dict[str, Any],
        cognitive_scaffold: Any,
        calculators: List[MetricsCalculatorInterface],
        exporters: List[MetricsExporterInterface],
    ):
        super().__init__(simulation_state, config, cognitive_scaffold)
        self.calculators = calculators
        self.exporters = exporters

    async def update(self, current_tick: int) -> None:
        """
        On each tick, run all metric calculators and dispatch the combined
        results to all registered exporters.
        """
        all_metrics: Dict[str, Any] = {}

        # Run each calculator and merge the results into a single dictionary
        for calculator in self.calculators:
            try:
                metrics_to_add = calculator.calculate_metrics(self.simulation_state)
                all_metrics.update(metrics_to_add)
            except Exception as e:
                print(f"Warning: Metric calculator {calculator.__class__.__name__} failed: {e}")

        # Pass the final, combined dictionary to EVERY exporter
        if all_metrics:
            for exporter in self.exporters:
                try:
                    await exporter.export_metrics(current_tick, all_metrics)
                except Exception as e:
                    print(f"Warning: Metrics exporter {exporter.__class__.__name__} failed: {e}")
