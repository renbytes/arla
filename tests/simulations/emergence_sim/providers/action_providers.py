# FILE: tests/providers/test_action_providers.py

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

# Mock dependencies
from agent_core.core.ecs.component import ActionPlanComponent
from agent_engine.systems.components import QLearningComponent

# The classes we are testing
from simulations.emergence_sim.providers.action_providers import (
    EmergenceDecisionSelector,
)

# Test Fixtures


@pytest.fixture
def mock_config():
    """Provides a mock config with the action feature dimension."""
    config = MagicMock()
    config.learning.q_learning.action_feature_dim = 15
    return config


@pytest.fixture
def mock_state_encoder(mock_config):
    """Mocks the state encoder to provide consistent state vectors."""
    encoder = MagicMock()
    encoder.encode_state.return_value = np.zeros(16)
    encoder.encode_internal_state.return_value = np.zeros(8)
    return encoder


@pytest.fixture
def mock_q_learning_component(mocker):
    """
    Mocks the QLearningComponent, including the neural network.
    The utility_network is mocked to return specific Q-values for testing.
    """
    q_comp = MagicMock(spec=QLearningComponent)
    q_comp.current_epsilon = 0.0  # Default to exploitation mode

    # Mock the neural network call. `side_effect` lets us return
    # different values for different calls.
    q_comp.utility_network.side_effect = [
        torch.tensor([10.0]),  # Q-value for action 1
        torch.tensor([50.0]),  # Q-value for action 2 (the best)
        torch.tensor([-5.0]),  # Q-value for action 3
    ]
    return q_comp


@pytest.fixture
def mock_simulation_state(mock_config, mock_q_learning_component):
    """Mocks the main simulation state object."""
    sim_state = MagicMock()
    sim_state.config = mock_config
    sim_state.get_component.return_value = mock_q_learning_component
    sim_state.entities = {"agent_1": {}}
    return sim_state


@pytest.fixture
def mock_action_plans(mock_config):
    """
    Creates a list of mock ActionPlanComponents. Each has a mock action_type
    with a get_feature_vector method that returns a correctly shaped vector.
    """
    plans = []
    for i in range(3):
        mock_action_type = MagicMock()
        # This is the method we want to ensure gets called correctly.
        mock_action_type.get_feature_vector.return_value = np.ones(mock_config.learning.q_learning.action_feature_dim)
        plan = MagicMock(spec=ActionPlanComponent)
        plan.action_type = mock_action_type
        plan.params = {}
        # Give each plan a unique name for easy identification in tests
        plan.name = f"ActionPlan_{i + 1}"
        plans.append(plan)
    return plans


# Unit Tests for EmergenceDecisionSelector


def test_decision_selector_exploit_selects_best_action(mock_state_encoder, mock_simulation_state, mock_action_plans):
    """
    Tests the core "exploit" logic: given a choice, the selector should
    pick the action with the highest Q-value from the network.
    This test verifies that get_feature_vector is called for each action.
    """
    # 1. ARRANGE: Create the selector with its dependencies
    selector = EmergenceDecisionSelector(state_encoder=mock_state_encoder)
    q_network = mock_simulation_state.get_component.return_value.utility_network

    # 2. ACT: Call the select method
    best_plan = selector.select(
        simulation_state=mock_simulation_state,
        entity_id="agent_1",
        possible_actions=mock_action_plans,
    )

    # 3. ASSERT:
    # Check that the feature vector was requested for all three actions
    for plan in mock_action_plans:
        plan.action_type.get_feature_vector.assert_called_once()

    # Check that the Q-network was queried for all three actions
    assert q_network.call_count == 3

    # Check that the selector returned the action corresponding to the highest
    # Q-value (50.0), which is the second action plan.
    assert best_plan is not None
    assert best_plan.name == "ActionPlan_2"


def test_decision_selector_explore_selects_randomly(
    mocker, mock_state_encoder, mock_simulation_state, mock_action_plans
):
    """
    Tests the "explore" logic: if epsilon is high, the selector should
    choose a random action and not even consult the Q-network.
    """
    # 1. ARRANGE:
    # Override the Q-component's epsilon to force exploration
    mock_q_comp = mock_simulation_state.get_component.return_value
    mock_q_comp.current_epsilon = 1.0  # 100% chance to explore
    q_network = mock_q_comp.utility_network

    # Mock `random.choice` to see what it was called with
    mock_random_choice = mocker.patch("random.choice", side_effect=lambda x: x[0])

    selector = EmergenceDecisionSelector(state_encoder=mock_state_encoder)

    # 2. ACT: Call the select method
    chosen_plan = selector.select(
        simulation_state=mock_simulation_state,
        entity_id="agent_1",
        possible_actions=mock_action_plans,
    )

    # 3. ASSERT:
    # Check that the Q-network was NOT called
    q_network.assert_not_called()

    # Check that random.choice was called with the list of possible actions
    mock_random_choice.assert_called_once_with(mock_action_plans)

    # Check that it returned the result of random.choice
    assert chosen_plan is not None
    assert chosen_plan.name == "ActionPlan_1"  # The first plan, as per our mock


def test_decision_selector_handles_no_actions(mock_state_encoder, mock_simulation_state):
    """
    Tests the edge case where there are no possible actions.
    The selector should return None gracefully.
    """
    # 1. ARRANGE
    selector = EmergenceDecisionSelector(state_encoder=mock_state_encoder)
    q_network = mock_simulation_state.get_component.return_value.utility_network

    # 2. ACT
    result = selector.select(
        simulation_state=mock_simulation_state,
        entity_id="agent_1",
        possible_actions=[],  # Empty list
    )

    # 3. ASSERT
    assert result is None
    q_network.assert_not_called()
