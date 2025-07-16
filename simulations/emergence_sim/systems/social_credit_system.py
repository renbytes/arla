# simulations/emergence_sim/systems/social_credit_system.py
"""
Manages the decentralized reputation and debt economy based on agent interactions.
This system listens for resource-giving actions and updates agents' social
credit scores and debt ledgers accordingly. It models a simple form of a
"human economy" where gifts create obligations. It also periodically penalizes
agents for long-standing, unpaid debts, creating a social pressure to reciprocate.
"""

from typing import Any, Dict, List, Type, cast

from agent_core.agents.actions.base_action import ActionOutcome
from agent_core.core.ecs.component import Component
from agent_engine.simulation.system import System

from ..components import DebtLedgerComponent, InventoryComponent, SocialCreditComponent


class SocialCreditSystem(System):
    """
    Manages the decentralized reputation economy by tracking debts and social credit.
    """

    # This system acts on entities that can participate in the economy.
    REQUIRED_COMPONENTS: List[Type[Component]] = [
        InventoryComponent,
        DebtLedgerComponent,
        SocialCreditComponent,
    ]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the system and subscribes to economic action events."""
        super().__init__(*args, **kwargs)
        if not self.event_bus:
            raise ValueError("SocialCreditSystem requires an EventBus.")

        # Subscribe to the event that triggers the action's execution
        self.event_bus.subscribe("execute_give_resource_action", self.on_give_resource)

        # Configurable parameters for the economy's dynamics
        self.debt_age_threshold = 100  # Ticks after which a debt is considered "old"
        self.debt_penalty = 0.01  # Penalty to social credit for each old debt check
        self.reciprocity_credit_bonus = 0.1  # Bonus for repaying a debt
        self.generosity_credit_bonus = 0.05  # Bonus for giving a gift

    def on_give_resource(self, event_data: Dict[str, Any]) -> None:
        """
        Handles the 'GiveResourceAction' by transferring resources and updating
        social ledgers. It distinguishes between repaying a debt and giving a new gift.
        """
        giver_id = event_data["entity_id"]
        action_plan = event_data["action_plan_component"]
        receiver_id = action_plan.params.get("target_agent_id")
        amount = action_plan.params.get("amount", 0.0)

        if not isinstance(receiver_id, str) or amount <= 0:
            return

        # --- 1. Get components for both agents ---
        giver_inv = self.simulation_state.get_component(giver_id, InventoryComponent)
        giver_credit = self.simulation_state.get_component(giver_id, SocialCreditComponent)
        giver_ledger = self.simulation_state.get_component(giver_id, DebtLedgerComponent)

        receiver_inv = self.simulation_state.get_component(receiver_id, InventoryComponent)
        receiver_ledger = self.simulation_state.get_component(receiver_id, DebtLedgerComponent)

        if not all(
            [
                isinstance(c, Component)
                for c in [
                    giver_inv,
                    giver_credit,
                    giver_ledger,
                    receiver_inv,
                    receiver_ledger,
                ]
            ]
        ):
            return  # One of the agents is not part of the credit economy

        # Cast for type safety
        giver_inv = cast(InventoryComponent, giver_inv)
        giver_credit = cast(SocialCreditComponent, giver_credit)
        giver_ledger = cast(DebtLedgerComponent, giver_ledger)
        receiver_inv = cast(InventoryComponent, receiver_inv)
        receiver_ledger = cast(DebtLedgerComponent, receiver_ledger)

        # --- 2. Validate Giver's Resources ---
        if giver_inv.current_resources < amount:
            outcome = ActionOutcome(
                success=False,
                message="Not enough resources to give.",
                reward=-0.1,
                result_details={"status": "failed_insufficient_resources"},
            )
            self._publish_outcome(giver_id, action_plan, outcome, event_data["current_tick"])
            return

        # --- 3. Check for Debt Repayment vs. New Gift ---
        # Is the giver repaying a debt they owe to the receiver?
        debt_to_repay = None
        for i, obligation in enumerate(giver_ledger.obligations):
            if obligation.get("creditor") == receiver_id:
                debt_to_repay = i
                break

        # --- 4. Execute Transfer and Update Ledgers ---
        giver_inv.current_resources -= amount
        receiver_inv.current_resources += amount

        if debt_to_repay is not None:
            # This is a REPAYMENT
            giver_ledger.obligations.pop(debt_to_repay)
            giver_credit.score = min(1.0, giver_credit.score + self.reciprocity_credit_bonus)
            message = f"Repaid a debt of {amount} to {receiver_id}."
            details = {"status": "debt_repaid", "amount": amount}
            base_reward = 2.0  # Positive reward for fulfilling an obligation
        else:
            # This is a new GIFT, creating a new debt for the receiver
            giver_credit.score = min(1.0, giver_credit.score + self.generosity_credit_bonus)
            receiver_ledger.obligations.append(
                {
                    "creditor": giver_id,
                    "debtor": receiver_id,
                    "amount": amount,
                    "tick": event_data["current_tick"],
                }
            )
            message = f"Gave a gift of {amount} to {receiver_id}."
            details = {"status": "gift_given", "amount": amount}
            base_reward = 1.0  # Smaller positive reward for simple generosity

        outcome = ActionOutcome(True, message, base_reward, details)
        self._publish_outcome(giver_id, action_plan, outcome, event_data["current_tick"])

    async def update(self, current_tick: int) -> None:
        """
        Periodically checks for old, un-reciprocated debts and applies a small
        social credit penalty to the debtor.
        """
        if current_tick % 20 != 0:  # Run this check every 20 ticks
            return

        all_debtors = self.simulation_state.get_entities_with_components(*self.REQUIRED_COMPONENTS)

        for _, components in all_debtors.items():
            ledger = cast(DebtLedgerComponent, components[DebtLedgerComponent])
            credit = cast(SocialCreditComponent, components[SocialCreditComponent])

            for obligation in ledger.obligations:
                debt_age = current_tick - obligation.get("tick", current_tick)
                if debt_age > self.debt_age_threshold:
                    # Apply a small, recurring penalty for old debts
                    credit.score = max(0.0, credit.score - self.debt_penalty)

    def _publish_outcome(self, entity_id: str, plan: Any, outcome: ActionOutcome, tick: int) -> None:
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
