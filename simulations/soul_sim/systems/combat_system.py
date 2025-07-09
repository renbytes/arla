# src/simulations/soul_sim/systems/combat_system.py
"""
Handles all combat-related actions and their consequences.
"""

from typing import Any, Dict, List, Optional, Type

import numpy as np
from agent_core.agents.actions.base_action import ActionOutcome
from agent_core.core.ecs.component import Component, TimeBudgetComponent
from agent_engine.simulation.system import System

# Import world-specific components from the current simulation package
from ..components import CombatComponent, HealthComponent, PositionComponent


def _resolve_combat(attacker_power: float, rng: np.random.Generator) -> Dict[str, Any]:
    """Calculates combat damage based on attacker's power and randomness."""
    damage = attacker_power * rng.uniform(0.8, 1.2)
    return {"damage_dealt": damage}


class CombatSystem(System):
    """
    Processes combat actions, calculates damage, and updates entity states.
    """

    REQUIRED_COMPONENTS: List[Type[Component]] = []  # Event-driven

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.event_bus:
            self.event_bus.subscribe("execute_combat_action", self.on_execute_combat)

    def on_execute_combat(self, event_data: Dict[str, Any]):
        """Handles a combat action event."""
        attacker_id = event_data["entity_id"]
        action_plan = event_data["action_plan"]
        target_id = action_plan.params.get("target_agent_id")

        if not isinstance(target_id, str):
            return

        # --- 1. Validate Combatants ---
        attacker_comps = self.simulation_state.entities.get(attacker_id, {})
        defender_comps = self.simulation_state.entities.get(target_id, {})

        failure_reason = self._validate_combatants(attacker_comps, defender_comps)
        if failure_reason:
            outcome = ActionOutcome(
                False,
                failure_reason,
                -0.05,
                {"status": "combat_failed", "reason": failure_reason},
            )
            self._publish_outcome(attacker_id, action_plan, outcome, event_data["current_tick"])
            return

        # --- 2. Resolve Combat ---
        attacker_combat = attacker_comps.get(CombatComponent)
        defender_health = defender_comps.get(HealthComponent)
        if not isinstance(attacker_combat, CombatComponent) or not isinstance(defender_health, HealthComponent):
            return

        rng = self.simulation_state.main_rng
        if not rng:
            return

        result = _resolve_combat(attacker_combat.attack_power, rng)
        damage = result["damage_dealt"]

        # --- 3. Apply Consequences ---
        defender_health.current_health -= damage
        was_defeated = defender_health.current_health <= 0

        if was_defeated:
            defender_health.current_health = 0
            if isinstance(
                time_comp := defender_comps.get(TimeBudgetComponent),
                TimeBudgetComponent,
            ):
                time_comp.is_active = False

                if not self.event_bus:
                    return

                self.event_bus.publish(
                    "entity_inactivated",
                    {
                        "entity_id": target_id,
                        "current_tick": event_data["current_tick"],
                    },
                )

        # --- 4. Create and Publish Outcome ---
        base_reward = self.config.get("learning", {}).get("rewards", {}).get("combat_reward_hit", 0.1)
        if was_defeated:
            base_reward += self.config.get("learning", {}).get("rewards", {}).get("combat_reward_defeat", 10.0)

        details = {
            "status": "defeated_entity" if was_defeated else "hit_target",
            "damage_dealt": damage,
            "target_agent_id": target_id,
        }
        message = f"Defeated {target_id}!" if was_defeated else f"Attacked {target_id} for {damage:.1f} damage."
        outcome = ActionOutcome(True, message, base_reward, details)

        self._publish_outcome(attacker_id, action_plan, outcome, event_data["current_tick"])

    def _validate_combatants(self, attacker_comps: Dict, defender_comps: Dict) -> Optional[str]:
        """Checks if combat is possible. Returns a failure reason string or None."""
        if not attacker_comps or not defender_comps:
            return "Attacker or defender does not exist."

        attacker_pos = attacker_comps.get(PositionComponent)
        defender_pos = defender_comps.get(PositionComponent)
        if not isinstance(attacker_pos, PositionComponent) or not isinstance(defender_pos, PositionComponent):
            return "Attacker or defender is missing a PositionComponent."

        if not self.simulation_state.environment:
            return "Environment not loaded"

        if self.simulation_state.environment.distance(attacker_pos.position, defender_pos.position) > 1:
            return "Target is too far away."

        defender_time = defender_comps.get(TimeBudgetComponent)
        if not isinstance(defender_time, TimeBudgetComponent) or not defender_time.is_active:
            return "Target is inactive."

        return None

    def _publish_outcome(self, entity_id: str, plan: Any, outcome: ActionOutcome, tick: int):
        """Helper to publish the action outcome to the event bus."""
        if not self.event_bus:
            return
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
