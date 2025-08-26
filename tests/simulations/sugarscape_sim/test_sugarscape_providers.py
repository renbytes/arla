# FILE: tests/simulations/sugarscape_sim/test_sugarscape_providers.py
"""
Unit tests for the Provider classes in the Sugarscape simulation.

These tests verify that the classes responsible for bridging the simulation-specific
logic with the core agent-engine are functioning correctly. This includes
perception, action generation, decision selection, and state encoding.
"""

import unittest
from unittest.mock import MagicMock

import numpy as np
from agent_core.core.ecs.component import PerceptionComponent

from simulations.sugarscape_sim.components import (
    EnergyComponent,
    MetabolismComponent,
    PositionComponent,
)
from simulations.sugarscape_sim.environment import SugarscapeEnvironment
from simulations.sugarscape_sim.providers import (
    HeuristicDecisionSelector,
    SugarscapePerceptionProvider,
    SugarscapeStateEncoder,
)


class TestSugarscapePerceptionProvider(unittest.TestCase):
    """Tests for the SugarscapePerceptionProvider."""

    def test_update_perception(self):
        """Verify that the provider correctly identifies visible entities."""
        provider = SugarscapePerceptionProvider()
        mock_sim_state = MagicMock()
        mock_env = MagicMock(spec=SugarscapeEnvironment)
        mock_pos = PositionComponent(x=10, y=10)
        mock_metabolism = MetabolismComponent(metabolic_rate=1, vision_range=5)
        mock_perc = PerceptionComponent(vision_range=5)

        # Setup environment state
        mock_env.agent_positions = {"agent_2": (11, 11)}
        mock_env.sugar_map = np.zeros((20, 20))
        mock_env.sugar_map[12, 12] = 4
        mock_env.sugar_map[18, 18] = 2  # This one is out of range

        mock_env.width = 20
        mock_env.height = 20

        def get_sugar_at(pos):
            return mock_env.sugar_map[pos[1], pos[0]]

        mock_env.get_sugar_at.side_effect = get_sugar_at
        mock_env.distance.side_effect = lambda p1, p2: abs(p1[0] - p2[0]) + abs(
            p1[1] - p2[1]
        )
        mock_sim_state.environment = mock_env

        # FIX: The test must pass the agent's components to the provider.
        components = {
            PositionComponent: mock_pos,
            MetabolismComponent: mock_metabolism,
            PerceptionComponent: mock_perc,
        }

        provider.update_perception("agent_1", components, mock_sim_state, 0)

        visible = mock_perc.visible_entities
        self.assertEqual(len(visible), 2)
        self.assertIn("agent_2", visible)
        self.assertIn("sugar_12_12", visible)
        self.assertEqual(visible["agent_2"]["type"], "agent")
        self.assertEqual(visible["sugar_12_12"]["amount"], 4)


class TestHeuristicDecisionSelector(unittest.TestCase):
    """Tests for the HeuristicDecisionSelector."""

    def test_selection_priority(self):
        """Verify the selector prioritizes harvesting, then moving to sugar."""
        selector = HeuristicDecisionSelector()
        mock_sim_state = MagicMock()
        mock_env = MagicMock(spec=SugarscapeEnvironment)
        mock_sim_state.environment = mock_env

        # Mock actions
        harvest_action = MagicMock()
        harvest_action.action_type.action_id = "harvest"
        move_to_sugar_action = MagicMock()
        move_to_sugar_action.action_type.action_id = "move"
        move_to_sugar_action.params = {"target_pos": (11, 11)}
        random_move_action = MagicMock()
        random_move_action.action_type.action_id = "move"
        random_move_action.params = {"target_pos": (9, 9)}

        # 1. Test harvest priority
        actions = [random_move_action, harvest_action]
        selected = selector.select(mock_sim_state, "agent_1", actions)
        self.assertEqual(selected, harvest_action)

        # 2. Test move-to-sugar priority
        mock_pos = PositionComponent(x=10, y=10)
        mock_perc = PerceptionComponent(vision_range=5)
        mock_perc.visible_entities = {
            "sugar_11_11": {"type": "sugar", "position": (11, 11), "amount": 4}
        }

        def get_component(entity_id, comp_type):
            if comp_type == PositionComponent:
                return mock_pos
            if comp_type == PerceptionComponent:
                return mock_perc
            return None

        mock_sim_state.get_component.side_effect = get_component
        mock_env.distance.side_effect = lambda p1, p2: abs(p1[0] - p2[0]) + abs(
            p1[1] - p2[1]
        )

        actions = [random_move_action, move_to_sugar_action]
        selected = selector.select(mock_sim_state, "agent_1", actions)
        self.assertEqual(selected, move_to_sugar_action)


class TestSugarscapeStateEncoder(unittest.TestCase):
    """Tests for the SugarscapeStateEncoder."""

    def test_encode_state_vector(self):
        """Verify the state vector has the correct format and values."""
        encoder = SugarscapeStateEncoder()
        mock_sim_state = MagicMock()
        mock_env = MagicMock(spec=SugarscapeEnvironment)
        mock_config = MagicMock()

        # Mock components and env properties
        mock_pos = PositionComponent(x=25, y=25)
        mock_energy = EnergyComponent(current_energy=50, initial_energy=100)
        mock_metabolism = MetabolismComponent(metabolic_rate=1, vision_range=10)
        mock_perc = PerceptionComponent(vision_range=10)
        mock_perc.visible_entities = {
            "sugar_26_26": {
                "type": "sugar",
                "position": (26, 26),
                "distance": 2,
                "amount": 4,
            }
        }
        mock_env.width = 50
        mock_env.height = 50
        mock_env.max_sugar_per_cell = 4
        mock_sim_state.environment = mock_env

        def get_component(entity_id, comp_type):
            if comp_type == PositionComponent:
                return mock_pos
            if comp_type == EnergyComponent:
                return mock_energy
            if comp_type == MetabolismComponent:
                return mock_metabolism
            if comp_type == PerceptionComponent:
                return mock_perc
            return None

        mock_sim_state.get_component.side_effect = get_component

        vector = encoder.encode_state(mock_sim_state, "agent_1", mock_config)

        # FIX: The vector size is 9 (3 for agent state + 2*3 for perception).
        self.assertEqual(len(vector), 9)

        # Agent state assertions
        self.assertAlmostEqual(vector[0], 0.5)  # x
        self.assertAlmostEqual(vector[1], 0.5)  # y
        self.assertAlmostEqual(vector[2], 0.5)  # energy

        # Nearest sugar patch assertions
        self.assertAlmostEqual(vector[3], 0.2)  # distance
        self.assertAlmostEqual(vector[4], 0.25)  # angle
        self.assertAlmostEqual(vector[5], 1.0)  # amount

        # Second nearest sugar patch (defaults)
        self.assertAlmostEqual(vector[6], 1.0)
        self.assertAlmostEqual(vector[7], 0.0)
        self.assertAlmostEqual(vector[8], 0.0)
