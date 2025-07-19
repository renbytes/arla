# src/simulations/soul_sim/systems/social_interaction_system.py
"""
Handles direct communication between agents.
"""

from typing import Any, Dict, List, Optional, Type

from agent_core.agents.actions.base_action import ActionOutcome, Intent
from agent_core.core.ecs.component import (
    Component,
    TimeBudgetComponent,
)
from agent_engine.simulation.system import System

# Import world-specific components
from ..components import EnvironmentObservationComponent, PositionComponent


class SocialInteractionSystem(System):
    """
    Processes communication actions, enabling information exchange between agents.
    """

    REQUIRED_COMPONENTS: List[Type[Component]] = []  # Event-driven

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.event_bus:
            self.event_bus.subscribe("execute_communicate_action", self.on_execute_communicate)

    def on_execute_communicate(self, event_data: Dict[str, Any]):
        """Handles a communication action event."""
        entity_id = event_data["entity_id"]
        action_plan = event_data["action_plan_component"]
        target_id = action_plan.params.get("target_agent_id")

        if not isinstance(target_id, str):
            return

        # 1. Validate Interaction
        failure_reason = self._validate_interaction(entity_id, target_id)
        if failure_reason:
            outcome = ActionOutcome(
                False,
                failure_reason,
                -0.01,
                {"status": "communication_failed", "reason": failure_reason},
            )
            self._publish_outcome(entity_id, action_plan, outcome, event_data["current_tick"])
            return

        # 2. Handle Information Exchange
        if action_plan.intent == Intent.COOPERATE:
            self._exchange_information(entity_id, target_id)

        # 3. Create and Publish Outcome
        # Use direct attribute access on the validated Pydantic model
        base_reward = self.config.learning.rewards.communicate_reward_base
        if action_plan.intent == Intent.COOPERATE:
            base_reward += self.config.learning.rewards.collaboration_bonus_per_agent

        details = {
            "status": "communicated",
            "target_agent_id": target_id,
            "intent": action_plan.intent.name,
        }
        message = f"Communicated with {target_id}."
        outcome = ActionOutcome(True, message, base_reward, details)

        self._publish_outcome(entity_id, action_plan, outcome, event_data["current_tick"])

    def _validate_interaction(self, entity_id: str, target_id: str) -> Optional[str]:
        """Checks if a communication action is possible."""
        comm_comps = self.simulation_state.entities.get(entity_id, {})
        target_comps = self.simulation_state.entities.get(target_id, {})

        if not self.simulation_state.environment:
            return "Environment not found"

        if not comm_comps or not target_comps:
            return "Communicator or target does not exist."
        comm_pos = comm_comps.get(PositionComponent)
        target_pos = target_comps.get(PositionComponent)
        if not isinstance(comm_pos, PositionComponent) or not isinstance(target_pos, PositionComponent):
            return "Communicator or target is missing a PositionComponent."
        if self.simulation_state.environment.distance(comm_pos.position, target_pos.position) > 2:
            return "Target is too far away."
        target_time = target_comps.get(TimeBudgetComponent)
        if not isinstance(target_time, TimeBudgetComponent) or not target_time.is_active:
            return "Target is inactive."
        return None

    def _exchange_information(self, entity_id: str, target_id: str):
        """Exchanges known entity locations between two agents."""
        comm_obs = self.simulation_state.get_component(entity_id, EnvironmentObservationComponent)
        target_obs = self.simulation_state.get_component(target_id, EnvironmentObservationComponent)

        if isinstance(comm_obs, EnvironmentObservationComponent) and isinstance(
            target_obs, EnvironmentObservationComponent
        ):
            comm_obs.known_entity_locations.update(target_obs.known_entity_locations)
            target_obs.known_entity_locations.update(comm_obs.known_entity_locations)

    def _publish_outcome(self, entity_id: str, plan: Any, outcome: ActionOutcome, tick: int):
        """Helper to publish the action outcome to the event bus."""
        if self.event_bus:
            self.event_bus.publish(
                "action_outcome_ready",
                {
                    "entity_id": entity_id,
                    "action_outcome": outcome,
                    "original_action_plan": plan,
                    "current_tick": tick,
                },
            )

    async def update(self, current_tick: int) -> None:
        """This system is purely event-driven."""
        pass
