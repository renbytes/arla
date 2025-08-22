# src/agent_engine/systems/action_system.py
"""
Orchestrates the action execution pipeline.
"""

import uuid
from typing import Any, Dict, List, Type, cast

from agent_core.agents.actions.action_interface import ActionInterface
from agent_core.agents.actions.base_action import ActionOutcome
from agent_core.core.ecs.component import (
    ActionOutcomeComponent,
    ActionPlanComponent,
    CompetenceComponent,
    Component,
    TimeBudgetComponent,
)
from agent_core.policy.reward_calculator_interface import RewardCalculatorInterface

from agent_engine.simulation.simulation_state import SimulationState
from agent_engine.simulation.system import System


class ActionSystem(System):
    """
    Orchestrates the action execution pipeline, from dispatching actions
    to calculating final rewards using an injected reward calculator.
    """

    REQUIRED_COMPONENTS: List[Type[Component]] = []  # Event-driven

    def __init__(
        self,
        simulation_state: SimulationState,
        config: Any,
        cognitive_scaffold: Any,
        reward_calculator: RewardCalculatorInterface,
    ):
        super().__init__(simulation_state, config, cognitive_scaffold)

        # The check here ensures the event_bus is not None upon initialization.
        if not self.event_bus:
            raise ValueError("EventBus cannot be None for ActionSystem.")

        self.reward_calculator = reward_calculator
        self.event_bus.subscribe("action_chosen", self.on_action_chosen)
        self.event_bus.subscribe("action_outcome_ready", self.on_action_outcome_ready)

    def on_action_chosen(self, event_data: Dict[str, Any]) -> None:
        """
        Receives a chosen action and dispatches it to the appropriate
        world-specific system for execution.
        """
        action_plan = cast(ActionPlanComponent, event_data.get("action_plan_component"))
        if not action_plan or not isinstance(action_plan.action_type, ActionInterface):
            return

        # FIX: Add a type guard to assure mypy that self.event_bus is not None here.
        if self.event_bus:
            specific_event_name = f"execute_{action_plan.action_type.action_id}_action"
            self.event_bus.publish(specific_event_name, event_data)

    def on_action_outcome_ready(self, event_data: Dict[str, Any]) -> None:
        """
        Receives a preliminary outcome, calculates the final reward, attaches a
        unique event ID, and publishes the final "action_executed" event.
        """
        entity_id = event_data["entity_id"]
        action_outcome = cast(ActionOutcome, event_data["action_outcome"])
        action_plan = cast(ActionPlanComponent, event_data["original_action_plan"])

        if not action_plan or not isinstance(action_plan.action_type, ActionInterface):
            return

        entity_components = self.simulation_state.entities.get(entity_id, {})
        intent_name = action_plan.intent.name if action_plan.intent else "UNKNOWN"

        final_reward, breakdown = self.reward_calculator.calculate_final_reward(
            base_reward=action_outcome.base_reward,
            action_type=action_plan.action_type,
            action_intent=intent_name,
            outcome_details=action_outcome.details.copy(),
            entity_components=entity_components,
        )
        action_outcome.reward = final_reward
        action_outcome.details["reward_breakdown"] = breakdown
        action_outcome.details["event_id"] = uuid.uuid4().hex

        self._update_entity_components(entity_id, action_outcome, action_plan)

        # FIX: Add a type guard for mypy before publishing.
        if self.event_bus:
            self.event_bus.publish(
                "action_executed",
                {
                    "entity_id": entity_id,
                    "action_plan": action_plan,
                    "action_outcome": action_outcome,
                    "current_tick": event_data["current_tick"],
                },
            )

        print(
            f"""   Entity {entity_id}
              executed {action_plan.action_type.name}.
               Final Reward: {final_reward:.3f}"""
        )

    def _update_entity_components(
        self, entity_id: str, outcome: ActionOutcome, plan: ActionPlanComponent
    ) -> None:
        """Helper to update the agent's components after an action."""
        if isinstance(
            cc := self.simulation_state.get_component(entity_id, CompetenceComponent),
            CompetenceComponent,
        ):
            if isinstance(plan.action_type, ActionInterface):
                cc.action_counts[plan.action_type.action_id] += 1

        if isinstance(
            aoc := self.simulation_state.get_component(
                entity_id, ActionOutcomeComponent
            ),
            ActionOutcomeComponent,
        ):
            aoc.success = outcome.success
            aoc.reward = outcome.reward
            aoc.details = outcome.details

        if isinstance(
            time_comp := self.simulation_state.get_component(
                entity_id, TimeBudgetComponent
            ),
            TimeBudgetComponent,
        ):
            if isinstance(plan.action_type, ActionInterface):
                action_cost = plan.action_type.get_base_cost(self.simulation_state)
                time_comp.current_time_budget -= action_cost

    def update(self, current_tick: int) -> None:
        """This system is purely event-driven."""
        pass
