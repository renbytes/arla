# tests/simulations/berry_sim/test_actions.py

import unittest
from unittest.mock import MagicMock

from simulations.berry_sim.actions import EatBerryAction, MoveAction
from simulations.berry_sim.components import PositionComponent
from simulations.berry_sim.environment import BerryWorldEnvironment


class TestMoveAction(unittest.TestCase):
    """Tests for the MoveAction class."""

    def test_action_properties(self):
        """Verify the action_id and name properties are correct."""
        action = MoveAction()
        self.assertEqual(action.action_id, "move")
        self.assertEqual(action.name, "Move")

    def test_get_base_cost(self):
        """Test that the base cost is returned correctly."""
        action = MoveAction()
        self.assertEqual(action.get_base_cost(MagicMock()), 1.0)

    def test_generate_possible_params_valid(self):
        """Test parameter generation when valid moves are available."""
        action = MoveAction()
        mock_sim_state = MagicMock()
        mock_env = MagicMock(spec=BerryWorldEnvironment)
        mock_pos = PositionComponent(x=5, y=5)

        # Configure mocks
        mock_sim_state.get_component.return_value = mock_pos
        mock_sim_state.environment = mock_env
        mock_env.is_valid_position.return_value = True
        mock_env.is_occupied.return_value = False

        params = action.generate_possible_params("agent_1", mock_sim_state, 1)

        # Expect four valid moves (N, S, E, W)
        self.assertEqual(len(params), 4)
        self.assertIn({"target_pos": (5, 6), "direction": "N"}, params)

    def test_generate_possible_params_no_pos(self):
        """Test that no params are generated if the agent has no position."""
        action = MoveAction()
        mock_sim_state = MagicMock()
        mock_sim_state.get_component.return_value = None  # No PositionComponent

        params = action.generate_possible_params("agent_1", mock_sim_state, 1)
        self.assertEqual(len(params), 0)

    def test_get_feature_vector(self):
        """Test that the feature vector has the correct format and size."""
        action = MoveAction()
        vector = action.get_feature_vector("agent_1", MagicMock(), {})
        # Cector size is now 5 due to the addition of the orange berry.
        # Schema: [is_move, is_eat_red, is_eat_blue, is_eat_yellow, is_eat_orange]
        self.assertEqual(len(vector), 5)
        self.assertEqual(vector, [1.0, 0.0, 0.0, 0.0, 0.0])


class TestEatBerryAction(unittest.TestCase):
    """Tests for the EatBerryAction class."""

    def test_action_properties(self):
        """Verify the action_id and name properties are correct."""
        action = EatBerryAction()
        self.assertEqual(action.action_id, "eat_berry")
        self.assertEqual(action.name, "Eat Berry")

    def test_generate_possible_params_on_berry(self):
        """Test param generation when the agent is on a berry."""
        action = EatBerryAction()
        mock_sim_state = MagicMock()
        mock_env = MagicMock(spec=BerryWorldEnvironment)
        mock_pos = PositionComponent(x=3, y=3)

        mock_sim_state.get_component.return_value = mock_pos
        mock_sim_state.environment = mock_env
        mock_env.berry_locations = {(3, 3): "red"}

        params = action.generate_possible_params("agent_1", mock_sim_state, 1)
        self.assertEqual(len(params), 1)
        self.assertEqual(params[0], {"berry_type": "red"})

    def test_generate_possible_params_no_berry(self):
        """Test that no params are generated if the agent is not on a berry."""
        action = EatBerryAction()
        mock_sim_state = MagicMock()
        mock_env = MagicMock(spec=BerryWorldEnvironment)
        mock_pos = PositionComponent(x=3, y=3)

        mock_sim_state.get_component.return_value = mock_pos
        mock_sim_state.environment = mock_env
        mock_env.berry_locations = {}  # No berries

        params = action.generate_possible_params("agent_1", mock_sim_state, 1)
        self.assertEqual(len(params), 0)

    def test_get_feature_vector(self):
        """Test that the feature vector is correctly one-hot encoded."""
        action = EatBerryAction()

        # Test for each berry type
        vector_red = action.get_feature_vector(
            "agent_1", MagicMock(), {"berry_type": "red"}
        )
        vector_blue = action.get_feature_vector(
            "agent_1", MagicMock(), {"berry_type": "blue"}
        )
        vector_yellow = action.get_feature_vector(
            "agent_1", MagicMock(), {"berry_type": "yellow"}
        )
        vector_orange = action.get_feature_vector(
            "agent_1", MagicMock(), {"berry_type": "orange"}
        )

        self.assertEqual(vector_red, [0.0, 1.0, 0.0, 0.0, 0.0])
        self.assertEqual(vector_blue, [0.0, 0.0, 1.0, 0.0, 0.0])
        self.assertEqual(vector_yellow, [0.0, 0.0, 0.0, 1.0, 0.0])
        self.assertEqual(vector_orange, [0.0, 0.0, 0.0, 0.0, 1.0])
