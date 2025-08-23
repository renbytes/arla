# FILE: tests/simulations/berry_sim/test_actions.py
"""
Unit tests for the actions defined in the berry_sim simulation.

Ensures that actions correctly generate their possible parameters based on
the simulation state and produce the expected feature vectors.
"""

import pytest
from unittest.mock import MagicMock
from simulations.berry_sim.actions import MoveAction, EatBerryAction
from simulations.berry_sim.components import PositionComponent
from simulations.berry_sim.environment import BerryWorldEnvironment


@pytest.fixture
def mock_sim_state_actions():
    """Provides a mock SimulationState object for testing actions."""
    state = MagicMock()
    state.environment = BerryWorldEnvironment(width=10, height=10)
    return state


class TestMoveAction:
    """Tests for the MoveAction."""

    def test_generate_possible_params(self, mock_sim_state_actions):
        """Verify that valid move parameters are generated."""
        action = MoveAction()
        agent_id = "agent_1"
        pos_comp = PositionComponent(x=5, y=5)
        mock_sim_state_actions.get_component.return_value = pos_comp

        params = action.generate_possible_params(
            agent_id, mock_sim_state_actions, tick=1
        )

        assert len(params) == 4
        assert {"target_pos": (5, 6), "direction": "N"} in params
        assert {"target_pos": (5, 4), "direction": "S"} in params
        assert {"target_pos": (6, 5), "direction": "E"} in params
        assert {"target_pos": (4, 5), "direction": "W"} in params

    def test_generate_possible_params_at_corner(self, mock_sim_state_actions):
        """Verify that moves are correctly constrained at the world edge."""
        action = MoveAction()
        agent_id = "agent_1"
        pos_comp = PositionComponent(x=0, y=0)
        mock_sim_state_actions.get_component.return_value = pos_comp

        params = action.generate_possible_params(
            agent_id, mock_sim_state_actions, tick=1
        )

        assert len(params) == 2
        assert {"target_pos": (0, 1), "direction": "N"} in params
        assert {"target_pos": (1, 0), "direction": "E"} in params

    def test_get_feature_vector(self):
        """Test that the feature vector has the correct format and size."""
        action = MoveAction()
        vector = action.get_feature_vector("agent_1", MagicMock(), {})
        assert len(vector) == 4
        assert vector == [1.0, 0.0, 0.0, 0.0]


class TestEatBerryAction:
    """Tests for the EatBerryAction."""

    def test_generate_possible_params_on_berry(self, mock_sim_state_actions):
        """Verify that an eat action is generated when the agent is on a berry."""
        action = EatBerryAction()
        agent_id = "agent_1"
        pos_comp = PositionComponent(x=3, y=3)
        mock_sim_state_actions.get_component.return_value = pos_comp
        mock_sim_state_actions.environment.berry_locations[(3, 3)] = "red"

        params = action.generate_possible_params(
            agent_id, mock_sim_state_actions, tick=1
        )

        assert len(params) == 1
        assert params[0] == {"berry_type": "red"}

    def test_generate_possible_params_no_berry(self, mock_sim_state_actions):
        """Verify that no eat action is generated when there is no berry."""
        action = EatBerryAction()
        agent_id = "agent_1"
        pos_comp = PositionComponent(x=4, y=4)
        mock_sim_state_actions.get_component.return_value = pos_comp

        params = action.generate_possible_params(
            agent_id, mock_sim_state_actions, tick=1
        )
        assert len(params) == 0

    def test_get_feature_vector(self):
        """Test that the feature vector is correctly one-hot encoded."""
        action = EatBerryAction()

        vector_red = action.get_feature_vector(
            "agent_1", MagicMock(), {"berry_type": "red"}
        )
        assert vector_red == [0.0, 1.0, 0.0, 0.0]

        vector_blue = action.get_feature_vector(
            "agent_1", MagicMock(), {"berry_type": "blue"}
        )
        assert vector_blue == [0.0, 0.0, 1.0, 0.0]

        vector_yellow = action.get_feature_vector(
            "agent_1", MagicMock(), {"berry_type": "yellow"}
        )
        assert vector_yellow == [0.0, 0.0, 0.0, 1.0]
