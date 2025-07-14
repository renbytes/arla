# tests/simulations/soul_sim/test_providers.py
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
from agent_core.core.ecs.component import (
    ActionPlanComponent,
    TimeBudgetComponent,
    ValueSystemComponent,
)
from agent_engine.simulation.simulation_state import SimulationState
from agent_engine.systems.components import QLearningComponent
from omegaconf import OmegaConf

from simulations.soul_sim.components import (
    FailedStatesComponent,
    HealthComponent,
    InventoryComponent,
    PositionComponent,
)
from simulations.soul_sim.config.schemas import SoulSimAppConfig
from simulations.soul_sim.providers import (
    SoulSimActionGenerator,
    SoulSimControllabilityProvider,
    SoulSimDecisionSelector,
    SoulSimNarrativeContextProvider,
    SoulSimRewardCalculator,
    SoulSimStateEncoder,
    SoulSimStateNodeEncoder,
    SoulSimVitalityMetricsProvider,
)

# --- Fixtures ---


@pytest.fixture(scope="module")
def pydantic_config():
    """Loads a stable, test-specific config into a validated Pydantic model."""
    conf = OmegaConf.load("tests/simulations/soul_sim/test_config.yml")
    return SoulSimAppConfig(**OmegaConf.to_container(conf, resolve=True))


@pytest.fixture
def mock_simulation_state(pydantic_config):
    """Creates a mock SimulationState with a validated config."""
    state = create_autospec(SimulationState, instance=True)
    state.config = pydantic_config
    state.entities = {}
    return state


# --- Test SoulSimActionGenerator ---


def test_action_generator(mock_simulation_state):
    """
    Tests that the action generator correctly queries registered actions
    and aggregates their possible parameters.
    """
    # Mock two action classes
    mock_action_class_1 = create_autospec(ActionInterface)
    mock_action_instance_1 = mock_action_class_1.return_value
    mock_action_instance_1.generate_possible_params.return_value = [{"param": 1}]

    mock_action_class_2 = create_autospec(ActionInterface)
    mock_action_instance_2 = mock_action_class_2.return_value
    mock_action_instance_2.generate_possible_params.return_value = [{"param": 2}, {"param": 3}]

    # Patch the action registry to return our mocks
    with patch.object(action_registry, "get_all_actions", return_value=[mock_action_class_1, mock_action_class_2]):
        generator = SoulSimActionGenerator()
        actions = generator.generate(mock_simulation_state, "agent_1", 1)

        assert len(actions) == 3
        assert isinstance(actions[0], ActionPlanComponent)
        assert actions[0].params == {"param": 1}
        assert actions[2].params == {"param": 3}
        mock_action_instance_1.generate_possible_params.assert_called_once()
        mock_action_instance_2.generate_possible_params.assert_called_once()


# --- Test SoulSimDecisionSelector ---


def test_decision_selector_no_q_comp_is_random(mock_simulation_state):
    """Tests that the selector falls back to random choice if no QLearningComponent exists."""
    mock_simulation_state.get_component.return_value = None  # No Q-comp
    selector = SoulSimDecisionSelector()
    possible_actions = [ActionPlanComponent(), ActionPlanComponent()]

    with patch.object(random, "choice") as mock_random_choice:
        selector.select(mock_simulation_state, "agent_1", possible_actions)
        mock_random_choice.assert_called_once_with(possible_actions)


def test_decision_selector_epsilon_greedy(mock_simulation_state):
    """Tests that the selector explores with probability epsilon."""
    q_comp = QLearningComponent(16, 1, 13, 0.001, "cpu")
    q_comp.current_epsilon = 0.99  # High epsilon to ensure exploration
    mock_simulation_state.get_component.return_value = q_comp
    selector = SoulSimDecisionSelector()
    possible_actions = [ActionPlanComponent(), ActionPlanComponent()]

    with patch.object(random, "random", return_value=0.5), patch.object(random, "choice") as mock_random_choice:
        selector.select(mock_simulation_state, "agent_1", possible_actions)
        mock_random_choice.assert_called_once_with(possible_actions)


def test_decision_selector_exploits_best_q_value(mock_simulation_state):
    """Tests that the selector chooses the action with the highest predicted Q-value."""
    # Setup mocks
    q_comp = QLearningComponent(16, 1, 13, 0.001, "cpu")
    q_comp.current_epsilon = 0.0  # No exploration

    # Mock the utility network to return predictable values
    mock_net = MagicMock(spec=torch.nn.Module)
    mock_net.return_value.item.side_effect = [10.0, 50.0, 20.0]  # Q-values for 3 actions
    q_comp.utility_network = mock_net

    mock_simulation_state.get_component.return_value = q_comp

    # Create mock actions
    action1 = ActionPlanComponent(action_type=create_autospec(ActionInterface), params={})
    action2 = ActionPlanComponent(action_type=create_autospec(ActionInterface), params={})  # This one should be chosen
    action3 = ActionPlanComponent(action_type=create_autospec(ActionInterface), params={})
    possible_actions = [action1, action2, action3]

    selector = SoulSimDecisionSelector()
    # Mock the internal state encoder calls
    with patch.object(selector.state_encoder, "encode_state", return_value=np.zeros(16)):
        mock_simulation_state.get_internal_state_features_for_entity.return_value = np.zeros(1)

        best_action = selector.select(mock_simulation_state, "agent_1", possible_actions)

        assert best_action == action2
        assert mock_net.call_count == 3


# --- Test SoulSimRewardCalculator ---


@pytest.mark.parametrize(
    "details, action_id, intent, expected_bonus, expected_multiplier, expected_final",
    [
        ({}, "unknown", "SOLITARY", 0.0, 1.0, 10.0),
        ({"status": "defeated_entity"}, "combat", "COMPETE", 20.0, 1.0, 30.0),
        ({"explored_new_tile": True}, "move", "SOLITARY", 0.1, 1.0, 10.1),
        ({}, "combat", "COMPETE", 0.0, 1.5, 15.0),
        ({}, "communicate", "COOPERATE", 0.0, 2.0, 20.0),
    ],
)
def test_reward_calculator(
    pydantic_config, details, action_id, intent, expected_bonus, expected_multiplier, expected_final
):
    """Tests various reward calculation scenarios."""
    calculator = SoulSimRewardCalculator(pydantic_config)

    # Setup mock components
    value_comp = ValueSystemComponent()
    value_comp.combat_victory_multiplier = 1.5
    value_comp.collaboration_multiplier = 2.0
    entity_components = {ValueSystemComponent: value_comp}

    mock_action = MagicMock(action_id=action_id)

    final_reward, breakdown = calculator.calculate_final_reward(
        base_reward=10.0,
        action_type=mock_action,
        action_intent=intent,
        outcome_details=details,
        entity_components=entity_components,
    )

    assert final_reward == pytest.approx(expected_final)
    assert breakdown.get("bonus", 0.0) == expected_bonus
    assert breakdown.get("value_multiplier", 1.0) == expected_multiplier


# --- Test SoulSimStateEncoder ---


def test_state_encoder_pads_correctly(mock_simulation_state, pydantic_config):
    """Tests that the state encoder pads the feature vector to the correct size."""
    # Make the expected size larger than what the function produces
    pydantic_config.learning.q_learning.state_feature_dim = 20
    encoder = SoulSimStateEncoder()

    # Mock components to return valid data
    mock_simulation_state.get_component.side_effect = [
        HealthComponent(100.0),
        TimeBudgetComponent(1000.0, 0),
        InventoryComponent(50.0),
    ]

    vector = encoder.encode_state(mock_simulation_state, "agent_1", pydantic_config)

    assert vector.shape == (20,)
    assert vector[-1] == 0.0  # Last element should be padded zero


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


def test_controllability_provider(mock_simulation_state):
    """Tests the ControllabilityProvider."""
    provider = SoulSimControllabilityProvider()
    components = {
        HealthComponent: HealthComponent(100.0),
        InventoryComponent: InventoryComponent(25.0),
        FailedStatesComponent: FailedStatesComponent(),
    }
    components[HealthComponent].current_health = 80.0
    components[FailedStatesComponent].tracker = {(1, 1): 1, (2, 2): 1}  # 2 failed states

    score = provider.get_controllability_score("agent_1", components)

    # Expected: (health_norm * 0.5) + (resource_norm * 0.5) - (failure_penalty)
    # (0.8 * 0.5) + (0.5 * 0.5) - (2 * 0.1) = 0.4 + 0.25 - 0.2 = 0.45
    assert score == pytest.approx(0.45)


def test_state_node_encoder(mock_simulation_state):
    """Tests the StateNodeEncoder."""
    provider = SoulSimStateNodeEncoder()
    components = {
        HealthComponent: HealthComponent(100.0),
        PositionComponent: PositionComponent(position=(3, 4), environment=MagicMock()),
    }
    components[HealthComponent].current_health = 50.0  # Medium health

    node = provider.encode_state_for_causal_graph("agent_1", components)

    assert node == ("STATE", "medium_health", "at_3_4")


def test_narrative_context_provider(mock_simulation_state):
    """Tests the NarrativeContextProvider."""
    provider = SoulSimNarrativeContextProvider()
    components = {
        HealthComponent: HealthComponent(100.0),
        InventoryComponent: InventoryComponent(20.0),
        PositionComponent: PositionComponent(position=(5, 6), environment=MagicMock()),
    }
    components[HealthComponent].current_health = 90.0

    context = provider.get_narrative_context("agent_1", components)

    assert "at position (5, 6)" in context["narrative"]
    assert "health is at 90" in context["narrative"]
    assert context["resource_level"] == 20.0
