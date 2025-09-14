"""
Unit tests for the provider classes in the Tragedy of the Commons simulation.

This file tests the logic of action generators, decision selectors, and state
encoders to ensure they produce the correct outputs based on a given
simulation state.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock
from simulations.tragedy_of_the_commons.providers import (
    HeuristicDecisionSelector,
    CommonsStateEncoder,
)
from simulations.tragedy_of_the_commons.environment import CommonsEnvironment
from simulations.tragedy_of_the_commons.components import (
    EnergyComponent,
    PositionComponent,
)
from agent_core.core.ecs.component import ActionPlanComponent


@pytest.fixture
def mock_sim_state_provider():
    """Provides a mock SimulationState for provider tests."""
    state = MagicMock()
    # Make the mock environment identifiable as a CommonsEnvironment
    state.environment = MagicMock(spec=CommonsEnvironment)
    state.config = MagicMock()
    state.config.environment.max_resource_per_patch = 20.0
    return state


class TestHeuristicDecisionSelector:
    """Tests for the HeuristicDecisionSelector."""

    @pytest.fixture
    def selector(self) -> HeuristicDecisionSelector:
        """Provides a HeuristicDecisionSelector instance."""
        # Pass the mock config, similar to how it's done in the run.py
        return HeuristicDecisionSelector(simulation_state=None, config=None)

    def test_prefers_graze_action(self, selector: HeuristicDecisionSelector):
        """Test that the selector chooses to graze if possible."""
        graze_action = ActionPlanComponent(action_type=MagicMock(action_id="graze"))
        move_action = ActionPlanComponent(action_type=MagicMock(action_id="move"))
        possible_actions = [move_action, graze_action]

        selected = selector.select(MagicMock(), "agent_1", possible_actions)
        assert selected is not None
        assert selected.action_type.action_id == "graze"

    def test_prefers_move_to_best_patch_if_no_graze(
        self, selector: HeuristicDecisionSelector, mock_sim_state_provider
    ):
        """Test that the selector moves towards the richest patch if it cannot graze."""

        # Setup environment mock to return different resource levels
        def get_resource_at_side_effect(pos):
            if pos == (1, 1):
                return 10.0  # Best patch
            if pos == (1, 2):
                return 5.0
            return 0.0

        mock_sim_state_provider.environment.get_resource_at.side_effect = (
            get_resource_at_side_effect
        )

        # Create move actions to different patches
        move_to_best = ActionPlanComponent(
            action_type=MagicMock(action_id="move"), params={"target_pos": (1, 1)}
        )
        move_to_good = ActionPlanComponent(
            action_type=MagicMock(action_id="move"), params={"target_pos": (1, 2)}
        )
        wait_action = ActionPlanComponent(
            action_type=MagicMock(action_id="wait"), params={}
        )
        possible_actions = [move_to_good, wait_action, move_to_best]

        selected = selector.select(mock_sim_state_provider, "agent_1", possible_actions)
        assert selected is not None
        assert selected.action_type.action_id == "move"
        assert selected.params["target_pos"] == (1, 1)

    def test_prefers_wait_if_no_good_move(
        self, selector: HeuristicDecisionSelector, mock_sim_state_provider
    ):
        """Test that the selector waits if it cannot graze or find a better patch."""
        # Mock the environment to return 0 for all resource checks
        mock_sim_state_provider.environment.get_resource_at.return_value = 0.0

        move_action = ActionPlanComponent(
            action_type=MagicMock(action_id="move"), params={"target_pos": (1, 1)}
        )
        wait_action = ActionPlanComponent(action_type=MagicMock(action_id="wait"))
        possible_actions = [move_action, wait_action]

        selected = selector.select(mock_sim_state_provider, "agent_1", possible_actions)
        assert selected is not None
        assert selected.action_type.action_id == "wait"


class TestCommonsStateEncoder:
    """Tests for the CommonsStateEncoder."""

    @pytest.fixture
    def encoder(self) -> CommonsStateEncoder:
        """Provides a CommonsStateEncoder instance."""
        return CommonsStateEncoder()

    def test_encode_state(self, encoder: CommonsStateEncoder, mock_sim_state_provider):
        """Test the state encoding logic."""
        # Setup mocks
        pos_comp = PositionComponent(x=5, y=5)
        energy_comp = EnergyComponent(current_energy=80.0, initial_energy=100.0)

        # Use a more robust side_effect that handles different component types
        def get_component_side_effect(entity_id, comp_type):
            if comp_type is PositionComponent:
                return pos_comp
            if comp_type is EnergyComponent:
                return energy_comp
            return None

        mock_sim_state_provider.get_component.side_effect = get_component_side_effect

        # Use a dictionary for a more robust side_effect
        def get_resource_at_side_effect(pos):
            resource_map = {
                (5, 5): 10.0,  # Center
                (5, 4): 12.0,  # North
                (5, 6): 8.0,  # South
                (6, 5): 15.0,  # East
                (4, 5): 5.0,  # West
            }
            return resource_map.get(pos, 0.0)

        mock_sim_state_provider.environment.get_resource_at.side_effect = (
            get_resource_at_side_effect
        )

        # Execute
        features = encoder.encode_state(mock_sim_state_provider, "agent_1", None)

        # Verify
        assert isinstance(features, np.ndarray)
        assert features.shape == (6,)
        assert features.dtype == np.float32

        # Expected values: [norm_energy, center, N, S, E, W]
        # norm_energy = 80/100 = 0.8
        # resources are normalized by max_resource (20.0)
        expected_features = np.array(
            [0.8, 10 / 20, 12 / 20, 8 / 20, 15 / 20, 5 / 20], dtype=np.float32
        )
        np.testing.assert_allclose(features, expected_features, rtol=1e-6)
