# FILE: simulations/emergence_sim/metrics/calculators.py

from collections import Counter
from typing import Any, Dict, List, Type

import numpy as np
from agent_core.core.ecs.abstractions import SimulationState
from agent_core.core.ecs.component import Component
from agent_engine.logging.metrics_calculator_interface import MetricsCalculatorInterface

from simulations.emergence_sim.components import (
    ConceptualSpaceComponent,
    DebtLedgerComponent,
    InventoryComponent,
    SocialCreditComponent,
)


class EmergenceMetricsCalculator(MetricsCalculatorInterface):
    """
    Calculates advanced metrics to track emergent social, economic,
    and cultural phenomena in the simulation.
    """

    # CORRECTED: Renamed this method from 'calculate' to 'calculate_metrics'
    def calculate_metrics(self, sim_state: SimulationState) -> Dict[str, Any]:
        """
        Main calculation method called by the MetricsSystem.
        Gathers data from all agents and computes aggregate metrics
        """
        all_entities = sim_state.entities.values()
        if not all_entities:
            return {}

        metrics = {}
        metrics.update(self._calculate_economic_metrics(all_entities))
        metrics.update(self._calculate_social_metrics(all_entities, sim_state))
        metrics.update(self._calculate_cultural_metrics(all_entities))

        return metrics

    def _get_component_data(
        self, entities: List[Dict[Type[Component], Component]], comp_type: Type[Component]
    ) -> List[Any]:
        """Helper to safely extract a list of components from all entities."""
        return [e.get(comp_type) for e in entities if e.get(comp_type) is not None]

    def _calculate_gini(self, data: np.ndarray) -> float:
        """Calculates the Gini coefficient for a 1D array of data."""
        if data.size == 0 or np.all(data == data[0]):
            return 0.0

        data = np.maximum(data, 0)
        sorted_data = np.sort(data)
        n = len(sorted_data)
        cum_data = np.cumsum(sorted_data, dtype=float)

        sum_of_proportions = cum_data.sum() / cum_data[-1]
        lorenz_area = (sum_of_proportions - (n + 1) / 2.0) / n
        return 2 * lorenz_area

    def _calculate_economic_metrics(self, entities: List[Dict[Type[Component], Component]]) -> Dict[str, float]:
        """Measures the distribution and inequality of material wealth."""
        inventories = self._get_component_data(entities, InventoryComponent)
        resources = np.array([inv.current_resources for inv in inventories])

        if resources.size == 0:
            return {"total_resources": 0.0, "average_resources": 0.0, "gini_coefficient_resources": 0.0}

        return {
            "total_resources": float(resources.sum()),
            "average_resources": float(resources.mean()),
            "gini_coefficient_resources": self._calculate_gini(resources),
        }

    def _calculate_social_metrics(
        self, entities: List[Dict[Type[Component], Component]], sim_state: SimulationState
    ) -> Dict[str, float]:
        """Measures the structure of social capital and relationships."""
        social_credits_comps = self._get_component_data(entities, SocialCreditComponent)
        credits = np.array([sc.score for sc in social_credits_comps])

        debt_ledgers = self._get_component_data(entities, DebtLedgerComponent)
        num_agents = len(sim_state.entities)
        max_possible_edges = num_agents * (num_agents - 1)

        active_edges = set()
        for ledger in debt_ledgers:
            for obligation in ledger.obligations:
                # The provided code for this section had a bug, it has been corrected
                # to access 'debtor' and 'creditor' keys if they exist.
                debtor = obligation.get("debtor_id") or obligation.get("debtor")
                creditor = obligation.get("creditor_id") or obligation.get("creditor")
                if debtor and creditor:
                    active_edges.add((debtor, creditor))

        network_density = len(active_edges) / max_possible_edges if max_possible_edges > 0 else 0.0

        credit_metrics = {
            "average_social_credit": float(credits.mean()) if credits.size > 0 else 0.0,
            "gini_coefficient_social_credit": self._calculate_gini(credits),
            "network_density": network_density,
            "total_active_obligations": float(len(active_edges)),
        }
        return credit_metrics

    def _calculate_cultural_metrics(self, entities: List[Dict[Type[Component], Component]]) -> Dict[str, float]:
        """Measures the emergence and consensus of shared symbols."""
        conceptual_spaces = self._get_component_data(entities, ConceptualSpaceComponent)

        if not conceptual_spaces:
            return {"total_unique_symbols": 0.0, "symbol_dominance": 0.0, "average_symbols_per_agent": 0.0}

        all_symbols = []
        for space in conceptual_spaces:
            all_symbols.extend(space.concepts.keys())

        if not all_symbols:
            return {"total_unique_symbols": 0.0, "symbol_dominance": 0.0, "average_symbols_per_agent": 0.0}

        symbol_counts = Counter(all_symbols)
        total_unique_symbols = len(symbol_counts)

        most_common_symbol_count = symbol_counts.most_common(1)[0][1]
        symbol_dominance = most_common_symbol_count / len(conceptual_spaces)

        return {
            "total_unique_symbols": float(total_unique_symbols),
            "symbol_dominance": symbol_dominance,
            "average_symbols_per_agent": len(all_symbols) / len(conceptual_spaces),
        }
