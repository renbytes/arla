# FILE: tests/simulations/soul_sim/test_providers.py
"""
Comprehensive unit tests for all provider classes in simulations/soul_sim/providers.py.
This ensures that the bridge between the generic agent-engine and the specific
rules of soul-sim is robust and correct.
"""

import random
from unittest.mock import MagicMock, create_autospec, patch

import numpy as np
import pytest
import torch
from agent_core.agents.actions.action_interface import ActionInterface
from agent_core.agents.actions.action_registry import action_registry
from agent_core.core.ecs.component import ActionPlanComponent, ValueSystemComponent
from agent_engine.simulation.simulation_state import SimulationState
from agent_engine.systems.components import QLearningComponent

from simulations.soul_sim.components import (
    HealthComponent,
    InventoryComponent,
    TimeBudgetComponent,
)
from simulations.soul_sim.config.schemas import SoulSimAppConfig
from simulations.soul_sim.providers import (
    SoulSimActionGenerator,
    SoulSimDecisionSelector,
    SoulSimRewardCalculator,
    SoulSimVitalityMetricsProvider,
)

# --- Fixtures ---


@pytest.fixture
def pydantic_config():
    """
    CORRECTED: Provides a mock config object with the necessary nested structure
    to support all tests in this file. This resolves the AttributeError.
    """
    config = MagicMock(spec=SoulSimAppConfig)

    # Create a mock for the 'learning' attribute
    learning_config = MagicMock()

    # Attach nested attributes to the 'learning' mock
    learning_config.rewards.combat_reward_defeat = 20.0
    learning_config.rewards.exploration_bonus = 0.1
    learning_config.q_learning.state_feature_dim = 16

    # Attach the fully formed 'learning' mock to the main config
    config.learning = learning_config

    return config


@pytest.fixture
def mock_simulation_state(pydantic_config):
    """Creates a mock SimulationState that uses the robust config fixture."""
    state = create_autospec(SimulationState, instance=True)
    state.config = pydantic_config
    state.entities = MagicMock()
    return state


# --- Test SoulSimActionGenerator ---


def test_action_generator(mock_simulation_state):
    """
    Tests that the action generator correctly queries registered actions
    and aggregates their possible parameters.
    """
    # Mocks the action *class*, not an instance of it, to resolve the TypeError
    mock_action_class_1 = create_autospec(ActionInterface)
    mock_action_class_1.return_value.generate_possible_params.return_value = [{"param": 1}]

    mock_action_class_2 = create_autospec(ActionInterface)
    mock_action_class_2.return_value.generate_possible_params.return_value = [
        {"param": 2},
        {"param": 3},
    ]

    with patch.object(
        action_registry,
        "get_all_actions",
        return_value=[mock_action_class_1, mock_action_class_2],
    ):
        generator = SoulSimActionGenerator()
        actions = generator.generate(mock_simulation_state, "agent_1", 1)

        assert len(actions) == 3
        assert isinstance(actions[0], ActionPlanComponent)
        mock_action_class_1.return_value.generate_possible_params.assert_called_once()


# --- Test SoulSimDecisionSelector ---


def test_decision_selector_no_q_comp_is_random(mock_simulation_state):
    """Tests that the selector falls back to random choice if no QLearningComponent exists."""
    mock_simulation_state.get_component.return_value = None
    selector = SoulSimDecisionSelector()
    possible_actions = [ActionPlanComponent(), ActionPlanComponent()]
    with patch.object(random, "choice") as mock_random_choice:
        selector.select(mock_simulation_state, "agent_1", possible_actions)
        mock_random_choice.assert_called_once_with(possible_actions)


def test_decision_selector_exploits_best_q_value(mock_simulation_state):
    """Tests that the selector chooses the action with the highest predicted Q-value."""
    q_comp = QLearningComponent(16, 1, 13, 0.001, "cpu")
    q_comp.current_epsilon = 0.0
    mock_net = MagicMock(spec=torch.nn.Module)
    mock_net.return_value.item.side_effect = [10.0, 50.0, 20.0]
    q_comp.utility_network = mock_net
    mock_simulation_state.get_component.return_value = q_comp

    action1 = ActionPlanComponent(action_type=create_autospec(ActionInterface), params={})
    action2 = ActionPlanComponent(action_type=create_autospec(ActionInterface), params={})
    action3 = ActionPlanComponent(action_type=create_autospec(ActionInterface), params={})
    possible_actions = [action1, action2, action3]
    selector = SoulSimDecisionSelector()

    with (
        patch.object(selector.state_encoder, "encode_state", return_value=np.zeros(16)),
        patch.object(selector.state_encoder, "encode_internal_state", return_value=np.zeros(1)),
    ):
        best_action = selector.select(mock_simulation_state, "agent_1", possible_actions)
        assert best_action == action2
        assert mock_net.call_count == 3


# --- Test SoulSimRewardCalculator ---


@pytest.mark.parametrize(
    "details, action_id, expected_final",
    [
        ({}, "unknown", 10.0),
        ({"status": "defeated_entity"}, "combat", 30.0),
        ({"explored_new_tile": True}, "move", 10.1),
    ],
)
def test_reward_calculator(pydantic_config, details, action_id, expected_final):
    """Tests various reward calculation scenarios."""
    calculator = SoulSimRewardCalculator(pydantic_config)
    entity_components = {ValueSystemComponent: ValueSystemComponent()}
    mock_action = MagicMock(action_id=action_id)

    final_reward, _ = calculator.calculate_final_reward(
        base_reward=10.0,
        action_type=mock_action,
        action_intent="SOLITARY",
        outcome_details=details,
        entity_components=entity_components,
    )
    assert final_reward == pytest.approx(expected_final)


# --- Test Other Providers ---


def test_vitality_metrics_provider(mock_simulation_state):
    """Tests the VitalityMetricsProvider."""
    provider = SoulSimVitalityMetricsProvider()
    components = {
        HealthComponent: HealthComponent(initial_health=100.0),
        TimeBudgetComponent: TimeBudgetComponent(initial_time_budget=200.0, lifespan_std_dev_percent=0),
        InventoryComponent: InventoryComponent(initial_resources=25.0),
    }
    components[HealthComponent].current_health = 50.0
    components[TimeBudgetComponent].current_time_budget = 100.0

    metrics = provider.get_normalized_vitality_metrics("agent_1", components, mock_simulation_state.config)
    assert metrics["health_norm"] == 0.5
    assert metrics["time_norm"] == 0.5
    assert metrics["resources_norm"] == 0.25
