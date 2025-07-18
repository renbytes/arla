# FILE: tests/systems/test_social_credit_system.py
from unittest.mock import MagicMock

import pytest

from simulations.emergence_sim.components import (
    DebtLedgerComponent,
    InventoryComponent,
    SocialCreditComponent,
)
from simulations.emergence_sim.systems.social_credit_system import SocialCreditSystem


@pytest.fixture
def mock_simulation_state():
    """Provides a mocked SimulationState with an event bus and entity store."""
    state = MagicMock()
    state.event_bus = MagicMock()
    state.entities = {}

    # Helper to simplify getting components in tests
    def get_component(entity_id, comp_type):
        return state.entities.get(entity_id, {}).get(comp_type)

    state.get_component = get_component

    return state


@pytest.fixture
def social_credit_system(mock_simulation_state):
    """Returns a configured instance of the SocialCreditSystem."""
    mock_simulation_state.config = MagicMock()
    system = SocialCreditSystem(
        simulation_state=mock_simulation_state,
        config=mock_simulation_state.config,
        cognitive_scaffold=MagicMock(),
    )
    return system


@pytest.fixture
def two_agents(mock_simulation_state):
    """Creates a 'giver' and 'receiver' agent with all necessary components."""
    agents_data = {
        "giver": {
            InventoryComponent: InventoryComponent(initial_resources=10.0),
            SocialCreditComponent: SocialCreditComponent(initial_credit=0.5),
            DebtLedgerComponent: DebtLedgerComponent(),
        },
        "receiver": {
            InventoryComponent: InventoryComponent(initial_resources=10.0),
            SocialCreditComponent: SocialCreditComponent(initial_credit=0.5),
            DebtLedgerComponent: DebtLedgerComponent(),
        },
    }
    mock_simulation_state.entities = agents_data
    return agents_data


def test_giving_gift_increases_generosity_and_creates_debt(social_credit_system, two_agents):
    """
    Tests that giving a resource (as a gift) increases the giver's social credit
    and creates an obligation for the receiver.
    """
    # 1. ARRANGE
    giver_credit = two_agents["giver"][SocialCreditComponent]
    receiver_ledger = two_agents["receiver"][DebtLedgerComponent]

    initial_giver_score = giver_credit.score
    assert len(receiver_ledger.obligations) == 0

    event_data = {
        "entity_id": "giver",
        "current_tick": 50,
        "action_plan_component": MagicMock(params={"target_agent_id": "receiver", "amount": 1.0}),
    }

    # 2. ACT
    social_credit_system.on_give_resource(event_data)

    # 3. ASSERT
    assert giver_credit.score == pytest.approx(initial_giver_score + social_credit_system.generosity_credit_bonus)
    assert len(receiver_ledger.obligations) == 1

    social_credit_system.event_bus.publish.assert_called_once()
    published_outcome = social_credit_system.event_bus.publish.call_args[0][1]["action_outcome"]
    assert published_outcome.details["status"] == "gift_given"


def test_repaying_debt_increases_reciprocity_and_clears_debt(social_credit_system, two_agents):
    """
    Tests that giving a resource to a creditor repays the debt and provides
    a larger reciprocity bonus.
    """
    # 1. ARRANGE
    giver_ledger = two_agents["giver"][DebtLedgerComponent]
    giver_credit = two_agents["giver"][SocialCreditComponent]
    giver_ledger.obligations.append({"creditor": "receiver", "debtor": "giver", "amount": 1.0, "tick": 10})

    initial_giver_score = giver_credit.score
    assert len(giver_ledger.obligations) == 1

    event_data = {
        "entity_id": "giver",
        "current_tick": 50,
        "action_plan_component": MagicMock(params={"target_agent_id": "receiver", "amount": 1.0}),
    }

    # 2. ACT
    social_credit_system.on_give_resource(event_data)

    # 3. ASSERT
    assert giver_credit.score == pytest.approx(initial_giver_score + social_credit_system.reciprocity_credit_bonus)
    assert len(giver_ledger.obligations) == 0

    published_outcome = social_credit_system.event_bus.publish.call_args[0][1]["action_outcome"]
    assert published_outcome.details["status"] == "debt_repaid"
