# arla/tests/simulations/emergence_sim/actions/test_economic_actions.py
from unittest.mock import MagicMock, patch

import pytest
from agent_core.agents.actions.action_registry import action_registry
from agent_core.agents.actions.base_action import Intent
from agent_core.core.ecs.component import TimeBudgetComponent

from simulations.emergence_sim.actions.economic_actions import (
    GiveResourceAction,
    RequestResourceAction,
)
from simulations.emergence_sim.components import InventoryComponent, PositionComponent

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


# GiveResourceAction Tests


class TestGiveResourceAction:
    def test_generate_params_with_resources_and_neighbors(self, mock_simulation_state):
        action = GiveResourceAction()
        mock_env = MagicMock()
        mock_env.get_entities_in_radius.return_value = [("neighbor_1", (1, 1))]
        pos_comp = PositionComponent(position=(0, 0), environment=mock_env)
        inv_comp = InventoryComponent(initial_resources=5.0)
        mock_simulation_state._components["agent_1"] = {
            PositionComponent: pos_comp,
            InventoryComponent: inv_comp,
        }
        mock_simulation_state._components["neighbor_1"] = {TimeBudgetComponent: TimeBudgetComponent(100, 0)}
        params = action.generate_possible_params("agent_1", mock_simulation_state, 0)
        assert len(params) == 1
        assert params[0]["target_agent_id"] == "neighbor_1"

    def test_generate_params_with_no_resources(self, mock_simulation_state):
        action = GiveResourceAction()
        mock_env = MagicMock()
        mock_env.get_entities_in_radius.return_value = [("neighbor_1", (1, 1))]
        pos_comp = PositionComponent(position=(0, 0), environment=mock_env)
        inv_comp = InventoryComponent(initial_resources=0.5)
        mock_simulation_state._components["agent_1"] = {
            PositionComponent: pos_comp,
            InventoryComponent: inv_comp,
        }
        params = action.generate_possible_params("agent_1", mock_simulation_state, 0)
        assert len(params) == 0

    def test_execute_returns_confirmation(self):
        action = GiveResourceAction()
        result = action.execute("agent_1", MagicMock(), {"amount": 1.0}, 0)
        assert result == {"status": "resource_given", "amount": 1.0}

    def test_get_feature_vector(self, mock_simulation_state):
        """Tests that a feature vector of the correct dimension is created."""
        action = GiveResourceAction()
        params = {"intent": Intent.COOPERATE}

        # CORRECTED: Patch the underlying '_actions' dictionary instead of the property.
        with patch.object(
            action_registry,
            "_actions",
            {"give_resource": None, "request_resource": None},
        ):
            vector = action.get_feature_vector("agent_1", mock_simulation_state, params)

        assert len(vector) == mock_simulation_state.config.learning.q_learning.action_feature_dim
        assert vector[0] == 1.0


# RequestResourceAction Tests


class TestRequestResourceAction:
    def test_generate_params_with_low_resources(self, mock_simulation_state):
        """
        Tests that params are generated when an agent has low resources.
        """
        # ARRANGE
        action = RequestResourceAction()
        mock_env = MagicMock()
        mock_env.get_entities_in_radius.return_value = [("neighbor_1", (1, 1))]

        pos_comp = PositionComponent(position=(0, 0), environment=mock_env)
        inv_comp = InventoryComponent(initial_resources=2.0)  # Low resources
        mock_simulation_state._components["agent_1"] = {
            PositionComponent: pos_comp,
            InventoryComponent: inv_comp,
        }
        mock_simulation_state._components["neighbor_1"] = {TimeBudgetComponent: TimeBudgetComponent(100, 0)}

        # ACT
        params = action.generate_possible_params("agent_1", mock_simulation_state, 0)

        # ASSERT
        assert len(params) == 1
        assert params[0]["target_agent_id"] == "neighbor_1"

    def test_generate_params_with_high_resources(self, mock_simulation_state):
        """
        Tests that no params are generated if the agent has sufficient resources.
        """
        # ARRANGE
        action = RequestResourceAction()
        mock_env = MagicMock()
        mock_env.get_entities_in_radius.return_value = [("neighbor_1", (1, 1))]

        pos_comp = PositionComponent(position=(0, 0), environment=mock_env)
        inv_comp = InventoryComponent(initial_resources=10.0)  # Too many resources
        mock_simulation_state._components["agent_1"] = {
            PositionComponent: pos_comp,
            InventoryComponent: inv_comp,
        }
        mock_simulation_state._components["neighbor_1"] = {TimeBudgetComponent: TimeBudgetComponent(100, 0)}

        # ACT
        params = action.generate_possible_params("agent_1", mock_simulation_state, 0)

        # ASSERT
        assert len(params) == 0

    def test_execute_returns_confirmation(self):
        """Tests that execute returns the correct dictionary."""
        action = RequestResourceAction()
        result = action.execute("agent_1", MagicMock(), {"amount": 1.0}, 0)
        assert result == {"status": "resource_requested", "amount": 1.0}
