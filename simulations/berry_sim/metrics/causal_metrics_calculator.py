# simulations/berry_sim/metrics/causal_metrics_calculator.py

from collections import defaultdict
from typing import Any, Dict

from agent_core.agents.actions.base_action import ActionOutcome
from agent_core.core.ecs.component import TimeBudgetComponent
from agent_engine.logging.metrics_calculator_interface import MetricsCalculatorInterface

from simulations.berry_sim.actions import EatBerryAction
from ..components import HealthComponent, PositionComponent
from ..environment import BerryWorldEnvironment


class CausalMetricsCalculator(MetricsCalculatorInterface):
    """Calculates and stores the CUS and CCI for the Berry Toxicity experiment."""

    def __init__(self):
        self.novel_context_decisions = defaultdict(lambda: {"correct": 0, "total": 0})
        self.yellow_berry_eats = defaultdict(lambda: {"correct": 0, "total": 0})
        self.causal_understanding_score = 0.0
        self.correlation_confusion_index = 1.0
        self.average_agent_health = 100.0
        self.active_agents = 0

    def update_with_event(self, event_data: Dict[str, Any], sim_state: Any):
        """Update internal state based on an 'action_executed' event."""
        action_plan = event_data.get("action_plan")
        if not action_plan or not isinstance(action_plan.action_type, EatBerryAction):
            return

        tick = event_data["current_tick"]
        agent_id = event_data["entity_id"]
        outcome: ActionOutcome = event_data["action_outcome"]
        berry_type = action_plan.params.get("berry_type")
        env = sim_state.environment
        pos_comp = sim_state.get_component(agent_id, PositionComponent)
        if not isinstance(env, BerryWorldEnvironment) or not pos_comp:
            return

        # CUS LOGIC
        if 1000 <= tick < 1100 and berry_type == "blue" and outcome.success:
            self.novel_context_decisions[agent_id]["total"] += 1
            is_near_water = env.is_near_feature(
                pos_comp.position, env.water_locations, 2
            )
            if outcome.reward > 0 and not is_near_water:
                self.novel_context_decisions[agent_id]["correct"] += 1
            elif outcome.reward < 0 and is_near_water:
                pass

        # CCI LOGIC
        if berry_type == "yellow" and outcome.success:
            self.yellow_berry_eats[agent_id]["total"] += 1
            if outcome.reward > 0:
                self.yellow_berry_eats[agent_id]["correct"] += 1

    def calculate_metrics(self, simulation_state: Any) -> Dict[str, Any]:
        """Calculate and return the final metrics based on stored state."""
        total_correct_novel = sum(
            d["correct"] for d in self.novel_context_decisions.values()
        )
        total_decisions_novel = sum(
            d["total"] for d in self.novel_context_decisions.values()
        )
        self.causal_understanding_score = (
            total_correct_novel / total_decisions_novel
            if total_decisions_novel > 0
            else 0.0
        )

        total_yellow_eats = sum(d["total"] for d in self.yellow_berry_eats.values())
        total_yellow_correct = sum(
            d["correct"] for d in self.yellow_berry_eats.values()
        )
        if total_yellow_eats > 10:
            correct_ratio = total_yellow_correct / total_yellow_eats
            self.correlation_confusion_index = 1.0 - abs(correct_ratio - 0.5) * 2
        else:
            self.correlation_confusion_index = 1.0

        all_agents = simulation_state.get_entities_with_components(
            [HealthComponent, TimeBudgetComponent]
        )
        total_health = 0
        active_count = 0
        for components in all_agents.values():
            health_comp = components.get(HealthComponent)
            time_comp = components.get(TimeBudgetComponent)
            if health_comp and time_comp and time_comp.is_active:
                total_health += health_comp.current_health
                active_count += 1

        self.average_agent_health = (
            total_health / active_count if active_count > 0 else 0
        )
        self.active_agents = active_count

        return {
            "causal_understanding_score": self.causal_understanding_score,
            "correlation_confusion_index": self.correlation_confusion_index,
            "average_agent_health": self.average_agent_health,
            "active_agents": self.active_agents,
        }
