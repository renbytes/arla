# tests/simulations/berry_sim/test_providers.py

import unittest
from unittest.mock import MagicMock

import numpy as np
from agent_core.core.ecs.component import PerceptionComponent
from simulations.berry_sim.components import HealthComponent, PositionComponent
from simulations.berry_sim.environment import BerryWorldEnvironment
from simulations.berry_sim.providers import (
    BerryPerceptionProvider,
    BerryStateEncoder,
)


class TestBerryPerceptionProvider(unittest.TestCase):
    """Tests for the BerryPerceptionProvider."""

    def test_update_perception(self):
        """Verify that the provider correctly identifies visible berries."""
        provider = BerryPerceptionProvider()
        mock_sim_state = MagicMock()
        mock_env = MagicMock(spec=BerryWorldEnvironment)
        mock_pos = PositionComponent(x=10, y=10)
        mock_perc = PerceptionComponent(vision_range=5)

        # Setup environment state
        mock_env.berry_locations = {
            (11, 11): "red",  # In range
            (1, 1): "blue",  # Out of range
            (12, 12): "yellow",  # In range
        }
        mock_env.crystal_locations = {
            (10, 14),  # In range
            (20, 20),  # Out of range
        }
        mock_env.distance.side_effect = lambda p1, p2: abs(p1[0] - p2[0]) + abs(
            p1[1] - p2[1]
        )
        mock_sim_state.environment = mock_env

        components = {PositionComponent: mock_pos, PerceptionComponent: mock_perc}

        provider.update_perception("agent_1", components, mock_sim_state, 1)

        # Assertions
        visible = mock_perc.visible_entities
        self.assertEqual(len(visible), 3)
        self.assertIn("berry_11_11", visible)
        self.assertIn("berry_12_12", visible)
        self.assertIn("crystal_10_14", visible)
        self.assertEqual(visible["berry_11_11"]["berry_type"], "red")
        self.assertEqual(visible["crystal_10_14"]["type"], "crystal")


class TestBerryStateEncoder(unittest.TestCase):
    """Tests for the BerryStateEncoder."""

    def test_encode_state(self):
        """Verify the state vector has the correct size and content."""
        # The BerryStateEncoder constructor takes no arguments.
        # The mock provider argument has been removed.
        encoder = BerryStateEncoder()
        mock_sim_state = MagicMock()
        mock_config = MagicMock()

        # Mock components
        mock_pos = PositionComponent(x=25, y=25)
        mock_health = HealthComponent(current_health=80.0, initial_health=100.0)
        mock_perc = PerceptionComponent(vision_range=10)
        mock_perc.visible_entities = {
            "berry_26_26": {
                "type": "berry",
                "berry_type": "red",
                "position": (26, 26),
                "distance": 2,
            }
        }

        # Mock config values
        mock_config.environment.params.width = 50
        mock_config.environment.params.height = 50
        mock_config.agent.vision_range = 10

        # Mock get_component to return the correct component based on type
        def get_component_side_effect(entity_id, component_type):
            if component_type == PositionComponent:
                return mock_pos
            if component_type == HealthComponent:
                return mock_health
            if component_type == PerceptionComponent:
                return mock_perc
            return None

        mock_sim_state.get_component.side_effect = get_component_side_effect

        vector = encoder.encode_state(mock_sim_state, "agent_1", mock_config)

        # Expected size: 4 (agent) + 5*2 (perception) = 14
        self.assertEqual(len(vector), 14)
        self.assertIsInstance(vector, np.ndarray)

        # Check agent state part: [x, y, health, is_boosted]
        self.assertAlmostEqual(vector[0], 0.5)  # 25/50
        self.assertAlmostEqual(vector[1], 0.5)  # 25/50
        self.assertAlmostEqual(vector[2], 0.8)  # 80/100
        self.assertAlmostEqual(vector[3], 0.0)  # Not boosted

        # Check perception part for red berry: [dist, angle]
        self.assertAlmostEqual(vector[4], 0.2)  # dist: 2/10
        self.assertAlmostEqual(vector[5], 0.25)  # angle: atan2(1,1)/pi = 0.25

        # Check perception part for other berries (should be default)
        # Blue berry: [dist, angle]
        self.assertAlmostEqual(vector[6], 1.0)
        self.assertAlmostEqual(vector[7], 0.0)


if __name__ == "__main__":
    unittest.main()
