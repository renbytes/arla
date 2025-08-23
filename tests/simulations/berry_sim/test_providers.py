# FILE: tests/simulations/berry_sim/test_providers.py
"""
Unit tests for the provider implementations in the berry_sim simulation.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from agent_core.core.ecs.component import PerceptionComponent
from simulations.berry_sim.providers import (
    BerryPerceptionProvider,
    BerryStateEncoder,
)
from simulations.berry_sim.components import PositionComponent, HealthComponent
from simulations.berry_sim.environment import BerryWorldEnvironment


@pytest.fixture
def mock_sim_state_providers():
    """Provides a mock SimulationState for provider tests."""
    state = MagicMock()
    state.environment = BerryWorldEnvironment(width=50, height=50)
    state.config = MagicMock()
    return state


class TestBerryPerceptionProvider:
    """Tests for the BerryPerceptionProvider."""

    def test_update_perception(self, mock_sim_state_providers):
        """Verify that the provider correctly finds berries within vision range."""
        provider = BerryPerceptionProvider()
        agent_id = "agent_1"

        pos_comp = PositionComponent(x=10, y=10)
        perc_comp = PerceptionComponent(vision_range=5)
        components = {PositionComponent: pos_comp, PerceptionComponent: perc_comp}

        mock_sim_state_providers.environment.berry_locations = {
            (11, 11): "red",  # Manhattan distance = 2 (in range)
            (15, 15): "blue",  # Manhattan distance = 10 (out of range)
        }

        provider.update_perception(
            agent_id, components, mock_sim_state_providers, _current_tick=1
        )

        assert len(perc_comp.visible_entities) == 1
        assert "berry_11_11" in perc_comp.visible_entities
        assert perc_comp.visible_entities["berry_11_11"]["berry_type"] == "red"


class TestBerryStateEncoder:
    """Tests for the BerryStateEncoder."""

    def test_encode_state(self, mock_sim_state_providers):
        """Verify the state vector has the correct size and content."""
        encoder = BerryStateEncoder(mock_sim_state_providers)
        agent_id = "agent_1"

        pos_comp = PositionComponent(x=25, y=10)  # Normalized: 0.5, 0.2
        health_comp = HealthComponent(
            current_health=80, initial_health=100
        )  # Normalized: 0.8
        perc_comp = PerceptionComponent(vision_range=10)

        def get_component_side_effect(eid, comp_type):
            if comp_type == PositionComponent:
                return pos_comp
            if comp_type == HealthComponent:
                return health_comp
            if comp_type == PerceptionComponent:
                return perc_comp
            return None

        mock_sim_state_providers.get_component = get_component_side_effect

        perc_comp.visible_entities["berry_27_12"] = {
            "type": "berry",
            "berry_type": "red",
            "position": (27, 12),
            "distance": 4,
        }

        mock_sim_state_providers.config.environment.get.return_value = {
            "width": 50,
            "height": 50,
        }

        vector = encoder.encode_state(
            mock_sim_state_providers, agent_id, mock_sim_state_providers.config
        )

        # The vector size is 9 (3 agent state + 3*2 perception features).
        assert vector.shape == (9,)
        # Check agent state values
        assert np.isclose(vector[0], 0.5)  # Agent X
        assert np.isclose(vector[1], 0.2)  # Agent Y
        assert np.isclose(vector[2], 0.8)  # Health
        # Check perception values for red berry
        assert np.isclose(vector[3], 0.4)  # Red berry distance
