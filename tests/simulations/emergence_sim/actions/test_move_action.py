from unittest.mock import MagicMock, patch

import pytest
from agent_core.agents.actions.action_registry import action_registry
from agent_core.agents.actions.base_action import Intent

from simulations.emergence_sim.actions.move_action import MoveAction
from simulations.emergence_sim.components import PositionComponent

# Fixtures


@pytest.fixture
def mock_simulation_state():
    """Provides a mock SimulationState with a basic config and component access."""
    state = MagicMock()
    state.config.learning.q_learning.action_feature_dim = 15
    state._components = {}

    def get_component_side_effect(entity_id, component_type):
        return state._components.get(entity_id, {}).get(component_type)

    state.get_component.side_effect = get_component_side_effect
    return state


# MoveAction Tests


class TestMoveAction:
    def test_generate_possible_params_gets_valid_moves(self, mock_simulation_state):
        """
        Tests that params are generated only for valid, passable neighbor tiles.
        """
        # ARRANGE
        action = MoveAction()
        mock_env = MagicMock()
        # Mock the environment to return only two valid moves out of four possibilities
        mock_env.is_valid_position.side_effect = lambda pos: pos in [(0, 1), (1, 0)]

        pos_comp = PositionComponent(position=(0, 0), environment=mock_env)
        mock_simulation_state._components["agent_1"] = {PositionComponent: pos_comp}

        # ACT
        params = action.generate_possible_params("agent_1", mock_simulation_state, 0)

        # ASSERT
        assert len(params) == 2
        # Check that the generated params match the valid positions
        generated_positions = {p["new_pos"] for p in params}
        assert (1, 0) in generated_positions
        assert (0, 1) in generated_positions

    def test_execute_returns_confirmation(self):
        """Tests that execute returns the correct status dictionary."""
        action = MoveAction()
        result = action.execute("agent_1", MagicMock(), {}, 0)
        assert result == {"status": "move_attempted"}

    def test_get_feature_vector(self, mock_simulation_state):
        """Tests that a feature vector of the correct dimension is created."""
        action = MoveAction()
        params = {"intent": Intent.SOLITARY, "direction": 2}

        with patch.object(action_registry, "_actions", {"move": None}):
            vector = action.get_feature_vector("agent_1", mock_simulation_state, params)

        assert len(vector) == mock_simulation_state.config.learning.q_learning.action_feature_dim
        assert vector[0] == 1.0  # One-hot encoding for the move action
