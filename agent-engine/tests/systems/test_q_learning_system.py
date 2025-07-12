# agent-engine/tests/systems/test_q_learning_system.py

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from agent_core.agents.actions.action_interface import ActionInterface
from agent_core.agents.actions.base_action import ActionOutcome
from agent_core.core.ecs.component import (
    ActionPlanComponent,
    AffectComponent,
    EmotionComponent,
    GoalComponent,
    IdentityComponent,
    TimeBudgetComponent,
)
from agent_engine.systems.components import QLearningComponent

# Subject under test
from agent_engine.systems.q_learning_system import QLearningSystem

# --- Fixtures ---


@pytest.fixture
def mock_simulation_state():
    """Mocks the SimulationState and its contained components for testing."""
    state = MagicMock()

    # Mock the QLearningComponent and its internal PyTorch objects
    q_comp = MagicMock(spec=QLearningComponent)
    # The network returns a tensor with a single value
    q_comp.utility_network = MagicMock(return_value=torch.tensor([[0.5]]))
    q_comp.optimizer = MagicMock()
    # The loss function returns a mock tensor that we can call .backward() on
    mock_loss_tensor = MagicMock()
    q_comp.loss_fn = MagicMock(return_value=mock_loss_tensor)

    # Mock other required components
    time_budget_comp = TimeBudgetComponent(initial_time_budget=100.0, lifespan_std_dev_percent=0.0)
    time_budget_comp.is_active = True

    # Set up the entities dictionary
    state.entities = {
        "agent1": {
            TimeBudgetComponent: time_budget_comp,
            QLearningComponent: q_comp,
            IdentityComponent: MagicMock(),
            AffectComponent: MagicMock(affective_buffer_maxlen=10),
            GoalComponent: MagicMock(embedding_dim=4),
            EmotionComponent: MagicMock(),
        }
    }

    # Configure the mock's methods
    state.get_component.side_effect = lambda eid, ctype: state.entities.get(eid, {}).get(ctype)
    state.get_entities_with_components.return_value = {"agent1": state.entities["agent1"]}
    state.get_internal_state_features_for_entity.return_value = np.zeros(10)
    state.device = "cpu"
    state.config = {"learning": {"q_learning": {"gamma": 0.99}}}

    return state


@pytest.fixture
def mock_state_encoder():
    """Mocks the StateEncoderInterface to return a predictable vector."""
    encoder = MagicMock()
    encoder.encode_state.return_value = np.ones(20)
    return encoder


@pytest.fixture
def mock_event_bus():
    """Mocks the EventBus."""
    return MagicMock()


@pytest.fixture
@patch("agent_engine.systems.q_learning_system.action_registry")
def q_learning_system(mock_registry, mock_simulation_state, mock_state_encoder, mock_event_bus):
    """Provides an initialized QLearningSystem with all dependencies mocked."""
    mock_simulation_state.event_bus = mock_event_bus

    # Mock the action registry to return a mock action
    mock_action_class = MagicMock()
    mock_action_instance = MagicMock(spec=ActionInterface)
    mock_action_instance.generate_possible_params.return_value = [{}]
    mock_action_instance.get_feature_vector.return_value = [0.0] * 5
    mock_action_class.return_value = mock_action_instance
    mock_registry.get_all_actions.return_value = [mock_action_class]

    system = QLearningSystem(
        simulation_state=mock_simulation_state,
        config=mock_simulation_state.config,
        cognitive_scaffold=MagicMock(),
        state_encoder=mock_state_encoder,
    )
    return system


# --- Test Cases ---


class TestQLearningSystem:
    """Groups tests for the QLearningSystem."""

    @pytest.mark.asyncio
    async def test_update_and_execute_sequence_happy_path(
        self, q_learning_system, mock_simulation_state, mock_event_bus
    ):
        """
        Tests the ideal operational sequence: update() is called, then on_action_executed().
        This verifies that the learning step works correctly when state is properly cached.
        """
        # --- Arrange ---
        q_comp = mock_simulation_state.get_component("agent1", QLearningComponent)
        mock_action_type = MagicMock(spec=ActionInterface)
        mock_action_type.get_feature_vector.return_value = [0.0] * 5
        event_data = {
            "entity_id": "agent1",
            "action_plan": ActionPlanComponent(action_type=mock_action_type, params={}),
            "action_outcome": ActionOutcome(success=True, message="", base_reward=10.0),
            "current_tick": 5,
        }

        # --- Act ---
        # 1. Run the passive update to cache the state
        await q_learning_system.update(current_tick=5)

        # 2. Run the event handler which triggers learning
        q_learning_system.on_action_executed(event_data)

        # --- Assert ---
        # State was cached
        assert "agent1" in q_learning_system.previous_states

        # Learning step was performed
        q_comp.optimizer.zero_grad.assert_called_once()
        q_comp.loss_fn.return_value.backward.assert_called_once()
        q_comp.optimizer.step.assert_called_once()

        # A learning event was published for monitoring
        mock_event_bus.publish.assert_called_with(
            "q_learning_update",
            {
                "entity_id": "agent1",
                "q_loss": q_comp.loss_fn.return_value.item(),
                "current_tick": 5,
            },
        )

    def test_on_action_executed_skips_if_no_previous_state(self, q_learning_system, mock_simulation_state, capsys):
        """
        Tests the original bug scenario: on_action_executed() is called without a prior
        update(), so there is no cached state. The system should skip learning gracefully.
        """
        # --- Arrange ---
        q_comp = mock_simulation_state.get_component("agent1", QLearningComponent)
        event_data = {
            "entity_id": "agent1",
            "action_plan": ActionPlanComponent(action_type=MagicMock(spec=ActionInterface)),
            "action_outcome": MagicMock(),
            "current_tick": 1,
        }
        # Ensure the state cache is empty
        q_learning_system.previous_states = {}

        # --- Act ---
        q_learning_system.on_action_executed(event_data)

        # --- Assert ---
        # The learning step should NOT have been performed
        q_comp.optimizer.step.assert_not_called()

        # A warning should be printed to the console
        captured = capsys.readouterr()
        assert "Warning: No previous state for agent1. Skipping Q-update." in captured.out

    @pytest.mark.asyncio
    async def test_update_skips_inactive_agents(self, q_learning_system, mock_simulation_state, mock_state_encoder):
        """
        Tests that the update() method does not waste time encoding states for
        agents that are marked as inactive.
        """
        # --- Arrange ---
        # Mark the agent as inactive
        time_comp = mock_simulation_state.get_component("agent1", TimeBudgetComponent)
        time_comp.is_active = False

        # --- Act ---
        await q_learning_system.update(current_tick=1)

        # --- Assert ---
        # The state encoder should not have been called
        mock_state_encoder.encode_state.assert_not_called()
        # The state cache should remain empty
        assert "agent1" not in q_learning_system.previous_states


class TestQLearningComponent:
    """Groups tests for the QLearningComponent data container."""

    def test_component_validation_success(self):
        """Tests that a healthy component passes validation."""
        # Arrange
        q_comp = QLearningComponent(
            state_feature_dim=10,
            internal_state_dim=5,
            action_feature_dim=3,
            q_learning_alpha=0.01,
            device="cpu",
        )
        q_comp.current_epsilon = 0.5

        # Act
        is_valid, errors = q_comp.validate("agent1")

        # Assert
        assert is_valid is True
        assert not errors

    def test_component_validation_nan_weights(self):
        """Tests that validation catches NaN values in the network parameters."""
        # Arrange
        q_comp = QLearningComponent(10, 5, 3, 0.01, "cpu")
        # Manually introduce a NaN into the weights
        with torch.no_grad():
            q_comp.utility_network.fc1.weight[0, 0] = float("nan")

        # Act
        is_valid, errors = q_comp.validate("agent1")

        # Assert
        assert is_valid is False
        assert len(errors) == 1
        assert "contains NaN values" in errors[0]
        assert "fc1.weight" in errors[0]

    @pytest.mark.parametrize("bad_epsilon", [-0.1, 1.1, float("nan")])
    def test_component_validation_bad_epsilon(self, bad_epsilon):
        """Tests that validation catches out-of-bounds or NaN epsilon values."""
        # Arrange
        q_comp = QLearningComponent(10, 5, 3, 0.01, "cpu")
        q_comp.current_epsilon = bad_epsilon

        # Act
        is_valid, errors = q_comp.validate("agent1")

        # Assert
        assert is_valid is False
        assert len(errors) == 1
        assert "epsilon" in errors[0]
