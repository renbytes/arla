"""
Unit tests for the actions in the Sugarscape simulation.

These tests verify that each action class correctly generates possible
parameters based on a given simulation state, returns the correct feature
vector for machine learning, and produces the expected ActionOutcome.
"""

import unittest
from unittest.mock import MagicMock

from simulations.sugarscape_sim.actions import (
    AttackAction,
    HarvestAction,
    MoveAction,
    ReproduceAction,
    ShareAction,
    StayAction,
)
from simulations.sugarscape_sim.components import (
    EnergyComponent,
    PositionComponent,
)
from simulations.sugarscape_sim.environment import SugarscapeEnvironment


class TestSugarscapeActions(unittest.TestCase):
    """A test suite for all actions in the Sugarscape simulation."""

    def setUp(self):
        """Set up common mock objects for all tests."""
        self.mock_sim_state = MagicMock()
        self.mock_env = MagicMock(spec=SugarscapeEnvironment)
        self.mock_sim_state.environment = self.mock_env

    def test_move_action(self):
        """Test the MoveAction."""
        action = MoveAction()
        pos_comp = PositionComponent(x=5, y=5)
        self.mock_sim_state.get_component.return_value = pos_comp
        self.mock_env.is_valid_position.return_value = True
        self.mock_env.get_entities_at_position.return_value = set()

        # Test parameter generation
        params = action.generate_possible_params("agent_1", self.mock_sim_state, 0)
        self.assertEqual(len(params), 4)
        self.assertIn({"target_pos": (5, 4)}, params)

        # Test feature vector
        vector = action.get_feature_vector("agent_1", self.mock_sim_state, {})
        self.assertEqual(vector, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_harvest_action(self):
        """Test the HarvestAction."""
        action = HarvestAction()
        pos_comp = PositionComponent(x=5, y=5)
        self.mock_sim_state.get_component.return_value = pos_comp
        self.mock_env.get_sugar_at.return_value = 3

        # Test parameter generation
        params = action.generate_possible_params("agent_1", self.mock_sim_state, 0)
        self.assertEqual(len(params), 1)

        # Test feature vector
        vector = action.get_feature_vector("agent_1", self.mock_sim_state, {})
        self.assertEqual(vector, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    def test_share_action(self):
        """Test the ShareAction."""
        action = ShareAction()
        pos_comp = PositionComponent(x=5, y=5)
        energy_comp = EnergyComponent(current_energy=100, initial_energy=100)
        self.mock_sim_state.get_component.side_effect = lambda _, comp_type: {
            PositionComponent: pos_comp,
            EnergyComponent: energy_comp,
        }.get(comp_type)
        self.mock_env.get_neighbors.return_value = [(5, 6)]
        self.mock_env.get_entities_at_position.return_value = {"agent_2"}

        # Test parameter generation
        params = action.generate_possible_params("agent_1", self.mock_sim_state, 0)
        self.assertEqual(len(params), 1)
        self.assertEqual(params[0]["target_id"], "agent_2")
        self.assertEqual(params[0]["amount"], 25)

        # Test feature vector
        vector = action.get_feature_vector("agent_1", self.mock_sim_state, {})
        self.assertEqual(vector, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    def test_attack_action(self):
        """Test the AttackAction."""
        action = AttackAction()
        pos_comp = PositionComponent(x=5, y=5)
        self.mock_sim_state.get_component.return_value = pos_comp
        self.mock_env.get_neighbors.return_value = [(5, 6)]
        self.mock_env.get_entities_at_position.return_value = {"agent_2"}

        # Test parameter generation
        params = action.generate_possible_params("agent_1", self.mock_sim_state, 0)
        self.assertEqual(len(params), 1)
        self.assertEqual(params[0]["target_id"], "agent_2")

        # Test feature vector
        vector = action.get_feature_vector("agent_1", self.mock_sim_state, {})
        self.assertEqual(vector, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    def test_reproduce_action(self):
        """Test the ReproduceAction."""
        action = ReproduceAction()
        energy_comp = EnergyComponent(current_energy=200, initial_energy=200)
        self.mock_sim_state.get_component.return_value = energy_comp

        # Test parameter generation (possible)
        params_possible = action.generate_possible_params(
            "agent_1", self.mock_sim_state, 0
        )
        self.assertEqual(len(params_possible), 1)

        # Test parameter generation (not enough energy)
        energy_comp.current_energy = 100
        params_impossible = action.generate_possible_params(
            "agent_1", self.mock_sim_state, 0
        )
        self.assertEqual(len(params_impossible), 0)

        # Test feature vector
        vector = action.get_feature_vector("agent_1", self.mock_sim_state, {})
        self.assertEqual(vector, [0.0, 0.0, 0.0, 0.0, 1.0, 0.0])

    def test_stay_action(self):
        """Test the StayAction."""
        action = StayAction()

        # Test parameter generation
        params = action.generate_possible_params("agent_1", self.mock_sim_state, 0)
        self.assertEqual(len(params), 1)

        # Test feature vector
        vector = action.get_feature_vector("agent_1", self.mock_sim_state, {})
        self.assertEqual(vector, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
