# tests/simulations/schelling_sim/test_actions.py

from unittest.mock import Mock

import pytest
from agent_core.agents.actions.base_action import ActionOutcome

from simulations.schelling_sim.actions import MoveToEmptyCellAction
from simulations.schelling_sim.components import (
    PositionComponent,
    SatisfactionComponent,
)
from simulations.schelling_sim.environment import SchellingGridEnvironment


@pytest.fixture
def move_action():
    """Provides a fresh instance of MoveToEmptyCellAction for each test."""
    return MoveToEmptyCellAction()


def test_action_id_and_name(move_action):
    """Tests that the action's identifiers are correct."""
    assert move_action.action_id == "move_to_empty_cell"
    assert move_action.name == "Move to Empty Cell"


def test_get_base_cost(move_action):
    """Tests the base cost of the action."""
    assert move_action.get_base_cost(None) == 1.0


def test_generate_possible_params_when_unsatisfied(move_action):
    """Tests that a move is generated when an agent is unsatisfied."""
    mock_sim_state = Mock()
    mock_satisfaction_comp = SatisfactionComponent(satisfaction_threshold=0.5)
    mock_satisfaction_comp.is_satisfied = False

    mock_env = Mock(spec=SchellingGridEnvironment)
    mock_env.get_empty_cells.return_value = [(10, 10), (12, 15)]
    mock_sim_state.environment = mock_env
    mock_sim_state.get_component.return_value = mock_satisfaction_comp

    params = move_action.generate_possible_params("agent1", mock_sim_state, 1)

    assert len(params) == 1
    assert "target_x" in params[0]
    assert "target_y" in params[0]
    assert (params[0]["target_x"], params[0]["target_y"]) in [(10, 10), (12, 15)]


def test_generate_possible_params_when_satisfied(move_action):
    """Tests that no move is generated when an agent is already satisfied."""
    mock_sim_state = Mock()
    mock_satisfaction_comp = SatisfactionComponent(satisfaction_threshold=0.5)
    mock_satisfaction_comp.is_satisfied = True
    mock_sim_state.get_component.return_value = mock_satisfaction_comp

    params = move_action.generate_possible_params("agent1", mock_sim_state, 1)

    assert params == []


def test_generate_possible_params_no_empty_cells(move_action):
    """Tests that no move is generated when there are no empty cells."""
    mock_sim_state = Mock()
    mock_satisfaction_comp = SatisfactionComponent(satisfaction_threshold=0.5)
    mock_satisfaction_comp.is_satisfied = False

    mock_env = Mock(spec=SchellingGridEnvironment)
    mock_env.get_empty_cells.return_value = []
    mock_sim_state.environment = mock_env
    mock_sim_state.get_component.return_value = mock_satisfaction_comp

    params = move_action.generate_possible_params("agent1", mock_sim_state, 1)
    assert params == []


def test_execute_success(move_action):
    """
    Tests that execute returns a successful ActionOutcome. The actual state
    change is handled by a System, not the Action itself.
    """
    mock_sim_state = Mock()
    mock_pos_comp = PositionComponent(x=1, y=1)
    mock_sim_state.get_component.return_value = mock_pos_comp
    params = {"target_x": 5, "target_y": 5, "direction": "east"}

    result = move_action.execute("agent1", mock_sim_state, params, 1)

    assert isinstance(result, ActionOutcome)
    assert result.success is True
    assert "moves from (1, 1) to (5, 5)" in result.message
    # The component's position should NOT be updated by the action itself.
    assert mock_pos_comp.position == (1, 1)


def test_execute_failure_missing_component(move_action):
    """Tests that execute fails gracefully if the agent is missing a PositionComponent."""
    mock_sim_state = Mock()
    mock_sim_state.get_component.return_value = None  # Simulate missing component
    params = {"target_x": 5, "target_y": 5}

    result = move_action.execute("agent1", mock_sim_state, params, 1)
    assert result.success is False
    assert "no PositionComponent" in result.message


def test_get_feature_vector(move_action):
    """Tests the feature vector generation for the action."""
    assert move_action.get_feature_vector("agent1", None, {}) == [1.0]
