# FILE: tests/systems/test_normative_abstraction_system.py
from unittest.mock import MagicMock, patch

import pytest

from simulations.emergence_sim.components import (
    RitualComponent,
    SocialCreditComponent,
)
from simulations.emergence_sim.systems.normative_abstraction_system import (
    NormativeAbstractionSystem,
)


@pytest.fixture
def mock_simulation_state():
    """Provides a mocked SimulationState with essential objects."""
    state = MagicMock()
    state.event_bus = MagicMock()
    state.cognitive_scaffold = MagicMock()
    state.entities = {}

    # Helper to simplify getting components in tests
    def get_entities_with_components(comp_type):
        entities = {}
        for agent_id, comps in state.entities.items():
            if comp_type in comps:
                entities[agent_id] = comps
        return entities

    state.get_entities_with_components = get_entities_with_components

    return state


@pytest.fixture
def normative_system(mock_simulation_state):
    """Returns a configured instance of the NormativeAbstractionSystem."""
    system = NormativeAbstractionSystem(
        simulation_state=mock_simulation_state,
        config=MagicMock(),  # Config is not used by this system
        cognitive_scaffold=mock_simulation_state.cognitive_scaffold,
    )
    # Patch the loaded flag to ensure the system runs even in isolated tests
    with patch(
        "simulations.emergence_sim.systems.normative_abstraction_system.EMERGENCE_COMPONENTS_LOADED",
        True,
    ):
        yield system


@pytest.mark.asyncio
async def test_reciprocity_norm_detected_and_published(normative_system, mock_simulation_state):
    """
    Tests that a reciprocity norm is detected when average social credit is high,
    and an event is published after the LLM names it.
    """
    # 1. ARRANGE
    # Create agents with high social credit scores (avg > 0.7 threshold)
    mock_simulation_state.entities = {
        "agent_1": {SocialCreditComponent: SocialCreditComponent(initial_credit=0.8)},
        "agent_2": {SocialCreditComponent: SocialCreditComponent(initial_credit=0.9)},
        "agent_3": {SocialCreditComponent: SocialCreditComponent(initial_credit=0.85)},
    }

    # Mock the LLM to return an abstract name for the norm
    normative_system.cognitive_scaffold.query.return_value = "Reciprocity"

    # 2. ACT
    # Run the system at a tick divisible by its interval (100)
    await normative_system.update(current_tick=100)

    # 3. ASSERT
    # The system should have queried the LLM to name the pattern
    normative_system.cognitive_scaffold.query.assert_called_once()
    prompt_arg = normative_system.cognitive_scaffold.query.call_args[1]["prompt"]
    assert "high average social credit score" in prompt_arg

    # The system should have published the newly named norm
    normative_system.event_bus.publish.assert_called_once_with(
        "abstract_norm_defined",
        {"label": "Reciprocity", "source_key": "reciprocity"},
    )

    # The system should remember that it has defined this norm
    assert "reciprocity" in normative_system.defined_norms


@pytest.mark.asyncio
async def test_ritual_norm_detected_and_published(normative_system, mock_simulation_state):
    """
    Tests that a ritual norm is detected when a majority of agents adopt a
    specific ritual.
    """
    # 1. ARRANGE
    # Three out of four agents know the "post_combat_loss" ritual (>50% threshold)
    mock_simulation_state.entities = {
        "agent_1": {RitualComponent: RitualComponent()},
        "agent_2": {RitualComponent: RitualComponent()},
        "agent_3": {RitualComponent: RitualComponent()},
        "agent_4": {RitualComponent: RitualComponent()},  # This one doesn't know the ritual
    }
    mock_simulation_state.entities["agent_1"][RitualComponent].codified_rituals["post_combat_loss"] = ["action_a"]
    mock_simulation_state.entities["agent_2"][RitualComponent].codified_rituals["post_combat_loss"] = ["action_a"]
    mock_simulation_state.entities["agent_3"][RitualComponent].codified_rituals["post_combat_loss"] = ["action_a"]

    normative_system.cognitive_scaffold.query.return_value = "Atonement"

    # 2. ACT
    await normative_system.update(current_tick=100)

    # 3. ASSERT
    normative_system.cognitive_scaffold.query.assert_called_once()
    prompt_arg = normative_system.cognitive_scaffold.query.call_args[1]["prompt"]
    assert "A significant portion of the population (75%)" in prompt_arg

    normative_system.event_bus.publish.assert_called_once_with(
        "abstract_norm_defined",
        {"label": "Atonement", "source_key": "post_combat_loss"},
    )

    assert "post_combat_loss" in normative_system.defined_norms


@pytest.mark.asyncio
async def test_no_norm_detected_if_thresholds_not_met(normative_system, mock_simulation_state):
    """
    Tests that no LLM calls or events occur if social patterns are not strong
    enough to be considered norms.
    """
    # 1. ARRANGE
    # Average social credit is low (avg < 0.7)
    mock_simulation_state.entities = {
        "agent_1": {SocialCreditComponent: SocialCreditComponent(initial_credit=0.2)},
        "agent_2": {SocialCreditComponent: SocialCreditComponent(initial_credit=0.3)},
    }

    # 2. ACT
    await normative_system.update(current_tick=100)

    # 3. ASSERT
    normative_system.cognitive_scaffold.query.assert_not_called()
    normative_system.event_bus.publish.assert_not_called()
