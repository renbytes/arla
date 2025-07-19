# FILE: tests/simulations/emergence_sim/actions/test_communication_actions.py
import random
from unittest.mock import MagicMock, patch

import pytest
from agent_core.agents.actions.action_registry import action_registry
from agent_core.agents.actions.base_action import Intent
from agent_core.core.ecs.component import MemoryComponent

from simulations.emergence_sim.actions.communication_actions import (
    GuessObjectAction,
    ProposeSymbolAction,
    ShareNarrativeAction,
)
from simulations.emergence_sim.components import PositionComponent

# Fixtures


@pytest.fixture
def mock_simulation_state():
    """Provides a mock SimulationState with a basic config and component access."""
    state = MagicMock()
    state.config.learning.q_learning.action_feature_dim = 15
    state.event_bus = MagicMock()

    state._components = {}

    def get_component_side_effect(entity_id, component_type):
        return state._components.get(entity_id, {}).get(component_type)

    state.get_component.side_effect = get_component_side_effect

    return state


# ProposeSymbolAction Tests


class TestProposeSymbolAction:
    def test_generate_possible_params(self, mock_simulation_state):
        action = ProposeSymbolAction()
        mock_env = MagicMock()
        pos_comp = PositionComponent(position=(0, 0), environment=mock_env)
        mock_simulation_state._components["agent_1"] = {PositionComponent: pos_comp}

        with patch.object(random, "randint", return_value=123):
            params = action.generate_possible_params("agent_1", mock_simulation_state, 0)

        assert len(params) == 2
        assert params[0]["target_object_id"] == "object_1"
        assert params[0]["symbol"] == "token_123"

    def test_execute(self):
        action = ProposeSymbolAction()
        result = action.execute("agent_1", MagicMock(), {"symbol": "token_123"}, 0)
        assert result == {"status": "symbol_proposed", "symbol": "token_123"}

    def test_get_feature_vector(self, mock_simulation_state):
        """Tests that a feature vector of the correct dimension is created."""
        action = ProposeSymbolAction()
        params = {"intent": Intent.COOPERATE}

        mock_actions = {
            "propose_symbol": None,
            "guess_object": None,
            "share_narrative": None,
        }

        with patch.object(action_registry, "_actions", mock_actions):
            vector = action.get_feature_vector("agent_1", mock_simulation_state, params)

        assert len(vector) == mock_simulation_state.config.learning.q_learning.action_feature_dim

        # CORRECTED: Find the index dynamically based on the sorted list of action IDs,
        # which is what the real code does. This prevents assertion errors.
        sorted_ids = sorted(mock_actions.keys())
        action_index = sorted_ids.index("propose_symbol")

        assert vector[action_index] == 1.0
        assert sum(vector[: len(sorted_ids)]) == 1.0  # Ensure only one action is hot


# (The rest of the test file remains the same)
class TestGuessObjectAction:
    def test_generate_possible_params(self, mock_simulation_state):
        action = GuessObjectAction()
        mock_env = MagicMock()
        pos_comp = PositionComponent(position=(0, 0), environment=mock_env)
        mock_simulation_state._components["agent_1"] = {PositionComponent: pos_comp}
        params = action.generate_possible_params("agent_1", mock_simulation_state, 0)
        assert len(params) == 2
        assert params[0]["guessed_object_id"] == "object_1"

    def test_execute(self):
        action = GuessObjectAction()
        result = action.execute("agent_1", MagicMock(), {"guessed_object_id": "object_1"}, 0)
        assert result == {"status": "object_guessed", "guess": "object_1"}


class TestShareNarrativeAction:
    def test_generate_params_with_narrative(self, mock_simulation_state):
        action = ShareNarrativeAction()
        mem_comp = MemoryComponent()
        mem_comp.last_llm_reflection_summary = "I learned something new."
        mock_simulation_state._components["agent_1"] = {MemoryComponent: mem_comp}
        params = action.generate_possible_params("agent_1", mock_simulation_state, 0)
        assert len(params) == 1
        assert params[0]["intent"] == Intent.COOPERATE

    def test_execute_publishes_event(self, mock_simulation_state):
        action = ShareNarrativeAction()
        mem_comp = MemoryComponent()
        mem_comp.last_llm_reflection_summary = "A new story"
        mock_simulation_state._components["agent_1"] = {MemoryComponent: mem_comp}
        result = action.execute("agent_1", mock_simulation_state, {}, 100)
        assert result == {"status": "narrative_shared"}
        mock_simulation_state.event_bus.publish.assert_called_once_with(
            "narrative_shared",
            {"speaker_id": "agent_1", "narrative": "A new story", "tick": 100},
        )
