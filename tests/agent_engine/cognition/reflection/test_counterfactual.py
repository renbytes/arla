# tests/agent_engine/cognition/reflection/test_counterfactual.py
"""
Unit tests for the formal counterfactual generation function.
"""

from unittest.mock import MagicMock, create_autospec

import pytest
from agent_core.core.ecs.component import MemoryComponent
from agent_engine.cognition.reflection.counterfactual import generate_counterfactual
from agent_engine.cognition.reflection.episode import Episode
from agent_engine.simulation.simulation_state import SimulationState


@pytest.fixture
def mock_simulation_state():
    """Creates a mock SimulationState with a pre-configured agent memory."""
    state = create_autospec(SimulationState, instance=True)
    mem_comp = MemoryComponent()

    # Setup mock causal model and data
    mock_causal_model = MagicMock()
    # Mock the 'whatif' method to return a predictable result
    mock_causal_model.whatif.return_value = MagicMock(value=5.5)
    mem_comp.causal_model = mock_causal_model

    # Add a data point that can be found by its event_id
    mem_comp.causal_data = [{"event_id": "event_123", "action": "action_a", "outcome": 1.0}]

    state.get_component.return_value = mem_comp
    return state


def test_generate_counterfactual_success(mock_simulation_state):
    """
    Tests that a counterfactual is successfully generated when all conditions are met.
    """
    # This episode contains the key event with the matching ID
    episode = Episode(
        start_tick=0,
        end_tick=10,
        theme="test_theme",
        events=[{"reward": 10.0, "event_id": "event_123", "action": {"action_type": "action_a"}}],
    )

    result = generate_counterfactual(
        episode=episode,
        simulation_state=mock_simulation_state,
        agent_id="agent_1",
        alternative_action="action_b",
    )

    assert result is not None
    assert result.counterfactual_action == "action_b"
    assert "would have been approximately 5.50" in result.predicted_outcome
    # Ensure the model's whatif method was called correctly
    mock_simulation_state.get_component.return_value.causal_model.whatif.assert_called_once()


def test_generate_counterfactual_no_causal_model(mock_simulation_state):
    """
    Tests that the function returns None if the agent has no causal model.
    """
    mock_simulation_state.get_component.return_value.causal_model = None
    episode = Episode(start_tick=0, end_tick=1, theme="t", events=[{"event_id": "e1"}])

    result = generate_counterfactual(episode, mock_simulation_state, "agent_1", "action_b")

    assert result is None


def test_generate_counterfactual_event_id_not_found(mock_simulation_state):
    """
    Tests that the function returns None if the key event's ID isn't in the causal data.
    """
    episode = Episode(start_tick=0, end_tick=1, theme="t", events=[{"event_id": "unfindable_event"}])

    result = generate_counterfactual(episode, mock_simulation_state, "agent_1", "action_b")

    assert result is None


def test_generate_counterfactual_empty_episode():
    """Tests that the function handles an empty episode gracefully."""
    result = generate_counterfactual(Episode(0, 0, "t"), MagicMock(), "a", "b")
    assert result is None
