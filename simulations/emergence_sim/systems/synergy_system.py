# FILE: simulations/emergence_sim/systems/synergy_system.py
from typing import Any, Dict, List, Type
from unittest.mock import MagicMock

from agent_core.agents.actions.base_action import ActionOutcome
from agent_engine.simulation.system import System

from simulations.emergence_sim.components import (
    InventoryComponent,
    SynergyTrackerComponent,
)


class SynergySystem(System):
    """Grants bonus resources to agents who have previously helped others."""

    REQUIRED_COMPONENTS: List[Type] = [InventoryComponent, SynergyTrackerComponent]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.synergy_duration = 50  # Ticks the synergy bonus lasts
        self.synergy_bonus_percent = 0.10  # 10% bonus

        if self.event_bus:
            self.event_bus.subscribe("execute_give_resource_action", self.on_give_resource)
            # We'll check for the outcome of a move action to see if resources were gained
            self.event_bus.subscribe("action_outcome_ready", self.on_action_outcome)

    def on_give_resource(self, event_data: Dict[str, Any]):
        """When a gift is given, add the giver to the receiver's synergy list."""
        receiver_id = event_data["action_plan_component"].params.get("target_agent_id")
        giver_id = event_data["entity_id"]

        synergy_comp = self.simulation_state.get_component(receiver_id, SynergyTrackerComponent)
        if synergy_comp:
            synergy_comp.synergy_partners[giver_id] = event_data["current_tick"]

    def on_action_outcome(self, event_data: Dict[str, Any]):
        """When an agent gains resources, grant a bonus to its synergy partners."""
        action_plan = event_data["original_action_plan"]
        action_id = getattr(action_plan.action_type, "action_id", "")

        if action_id == "move" and event_data["action_outcome"].reward > 0:
            finder_id = event_data["entity_id"]
            synergy_comp = self.simulation_state.get_component(finder_id, SynergyTrackerComponent)

            if not synergy_comp:
                return

            for giver_id, gift_tick in list(synergy_comp.synergy_partners.items()):
                if event_data["current_tick"] - gift_tick > self.synergy_duration:
                    del synergy_comp.synergy_partners[giver_id]
                    continue

                giver_inv = self.simulation_state.get_component(giver_id, InventoryComponent)
                if giver_inv:
                    bonus = event_data["action_outcome"].reward * self.synergy_bonus_percent
                    giver_inv.current_resources += bonus
                    print(f"SYNERGY! {giver_id} earned a {bonus:.2f} bonus from {finder_id}'s find.")

                    # Publish a reward outcome for the giver
                    # This tells their brain that their prior investment paid off.
                    bonus_outcome = ActionOutcome(
                        True,
                        "Received a synergy bonus!",
                        bonus,
                        {"from_agent": finder_id},
                    )
                    dummy_plan = MagicMock()  # The reward isn't from an immediate action
                    self._publish_outcome(giver_id, dummy_plan, bonus_outcome, event_data["current_tick"])

    def _publish_outcome(self, entity_id: str, plan: Any, outcome: ActionOutcome, tick: int):
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
        pass  # Event-driven
