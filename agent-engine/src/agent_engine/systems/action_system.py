# src/agent_engine/systems/action_system.py

"""
Orchestrates the action execution pipeline.
"""

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
from agent_core.core.ecs.event_bus import EventBus
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
        config: Dict[str, Any],
        cognitive_scaffold: Any,
        reward_calculator: RewardCalculatorInterface,
    ):
        super().__init__(simulation_state, config, cognitive_scaffold)

        event_bus = simulation_state.event_bus
        if not event_bus:
            raise ValueError("EventBus cannot be None for ActionSystem.")
        self.event_bus: EventBus = event_bus

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

        # The engine dispatches the execution request to a world-specific system
        # (e.g., CombatSystem, MovementSystem) which will eventually publish
        # the "action_outcome_ready" event.
        specific_event_name = f"execute_{action_plan.action_type.action_id}_action"
        self.event_bus.publish(specific_event_name, event_data)

    def on_action_outcome_ready(self, event_data: Dict[str, Any]) -> None:
        """
        Receives a preliminary outcome, calculates the final reward using the
        injected calculator, and publishes the final "action_executed" event.
        """
        entity_id = event_data["entity_id"]
        action_outcome = cast(ActionOutcome, event_data["action_outcome"])
        action_plan = cast(ActionPlanComponent, event_data["original_action_plan"])

        # Ensure the action plan and its type are valid before proceeding.
        if not action_plan or not isinstance(action_plan.action_type, ActionInterface):
            return

        entity_components = self.simulation_state.entities.get(entity_id, {})

        # 1. Define a default value
        intent_name = "UNKNOWN"

        # 2. Use an 'if' block to narrow the type
        if action_plan.intent is not None:
            intent_name = action_plan.intent.name

        # Use the injected reward calculator
        # FIX: Pass a copy of the details dictionary to prevent unintended mutation
        # of the object that the mock is inspecting.
        final_reward, breakdown = self.reward_calculator.calculate_final_reward(
            base_reward=action_outcome.base_reward,
            action_type=action_plan.action_type,
            action_intent=intent_name,
            outcome_details=action_outcome.details.copy(),
            entity_components=entity_components,
        )
        action_outcome.reward = final_reward
        action_outcome.details["reward_breakdown"] = breakdown

        # Update entity components with the final outcome
        self._update_entity_components(entity_id, action_outcome, action_plan)

        # Publish the final, complete event for other systems to consume
        self.event_bus.publish(
            "action_executed",
            {
                "entity_id": entity_id,
                "action_plan": action_plan,
                "action_outcome": action_outcome,
                "current_tick": event_data["current_tick"],
            },
        )

        # This line is now safe because of the guard clause at the top
        print(f"   Entity {entity_id} executed {action_plan.action_type.name}. Final Reward: {final_reward:.3f}")

    def _update_entity_components(self, entity_id: str, outcome: ActionOutcome, plan: ActionPlanComponent) -> None:
        """Helper to update the agent's components after an action."""
        # Update competence
        if isinstance(
            cc := self.simulation_state.get_component(entity_id, CompetenceComponent),
            CompetenceComponent,
        ):
            if isinstance(plan.action_type, ActionInterface):
                cc.action_counts[plan.action_type.action_id] += 1

        # Update action outcome
        if isinstance(
            aoc := self.simulation_state.get_component(entity_id, ActionOutcomeComponent),
            ActionOutcomeComponent,
        ):
            aoc.success = outcome.success
            aoc.reward = outcome.reward
            aoc.details = outcome.details

        if isinstance(
            time_comp := self.simulation_state.get_component(entity_id, TimeBudgetComponent),
            TimeBudgetComponent,
        ):
            if isinstance(plan.action_type, ActionInterface):
                action_cost = plan.action_type.get_base_cost(self.simulation_state)
                time_comp.current_time_budget -= action_cost

    async def update(self, current_tick: int) -> None:
        """This system is purely event-driven."""
        pass
