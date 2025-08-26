# FILE: tests/simulations/berry_sim/test_berry_providers.py

import unittest
from unittest.mock import MagicMock

from agent_core.core.ecs.component import PerceptionComponent
from agent_engine.simulation.simulation_state import SimulationState

from simulations.berry_sim.components import (
    HealthComponent,
    MetabolicBoostComponent,
    PositionComponent,
)
from simulations.berry_sim.environment import BerryWorldEnvironment
from simulations.berry_sim.providers import (
    BerryPerceptionProvider,
    BerryStateEncoder,
)


class TestBerryPerceptionProvider(unittest.TestCase):
    """Tests for the BerryPerceptionProvider."""

    def test_update_perception(self):
        """Verify that the provider correctly identifies visible entities."""
        provider = BerryPerceptionProvider()
        mock_sim_state = MagicMock(spec=SimulationState)
        mock_env = MagicMock(spec=BerryWorldEnvironment)
        mock_pos = PositionComponent(x=10, y=10)
        mock_perc = PerceptionComponent(vision_range=7)

        # Setup environment state
        mock_env.berry_locations = {
            (11, 11): "red",  # distance 2, visible
            (18, 18): "blue",  # distance 16, not visible
        }
        mock_env.crystal_locations = {(9, 9)}  # distance 2, visible

        mock_env.distance.side_effect = lambda p1, p2: abs(p1[0] - p2[0]) + abs(
            p1[1] - p2[1]
        )
        mock_sim_state.environment = mock_env

        components = {PositionComponent: mock_pos, PerceptionComponent: mock_perc}

        provider.update_perception("agent_1", components, mock_sim_state, 0)

        visible = mock_perc.visible_entities
        # FIX: The test was asserting 3, but only 2 items are in range.
        self.assertEqual(len(visible), 2)
        self.assertIn("berry_11_11", visible)
        self.assertIn("crystal_9_9", visible)
        self.assertNotIn("berry_18_18", visible)


class TestBerryStateEncoder(unittest.TestCase):
    """Tests for the BerryStateEncoder."""

    def test_encode_state(self):
        """Verify the state vector has the correct format and values."""
        encoder = BerryStateEncoder()
        mock_sim_state = MagicMock(spec=SimulationState)
        mock_env = MagicMock(spec=BerryWorldEnvironment)
        mock_config = MagicMock()
        mock_config.agent.vision_range = 10
        mock_config.environment.params.width = 50
        mock_config.environment.params.height = 50

        # Mock components
        mock_pos = PositionComponent(x=25, y=25)
        mock_health = HealthComponent(current_health=50, initial_health=100)
        mock_perc = PerceptionComponent(vision_range=10)
        mock_boost = MetabolicBoostComponent(active=True)

        mock_perc.visible_entities = {
            "berry_26_26": {
                "type": "berry",
                "berry_type": "red",
                "position": (26, 26),
                "distance": 2,
            }
        }

        mock_sim_state.environment = mock_env
        mock_sim_state.get_component.side_effect = lambda _, comp_type: {
            PositionComponent: mock_pos,
            HealthComponent: mock_health,
            PerceptionComponent: mock_perc,
            MetabolicBoostComponent: mock_boost,
        }.get(comp_type)

        vector = encoder.encode_state(mock_sim_state, "agent_1", mock_config)

        self.assertEqual(len(vector), 14)
        # agent_x, agent_y, health, is_boosted
        self.assertAlmostEqual(vector[0], 0.5)  # x
        self.assertAlmostEqual(vector[1], 0.5)  # y
        # FIX: The test was asserting 0.0, but 50/100 is 0.5.
        self.assertAlmostEqual(vector[2], 0.5)  # health
        self.assertAlmostEqual(vector[3], 1.0)  # boosted
