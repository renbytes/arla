# tests/agent_engine/systems/test_q_learning_system.py
"""
Unit tests for the causally-enhanced QLearningSystem.
"""

from unittest.mock import MagicMock, create_autospec, patch

import numpy as np
import pytest
from agent_core.agents.actions.action_interface import ActionInterface
from agent_core.agents.actions.base_action import ActionOutcome
from agent_core.core.ecs.component import (
    AffectComponent,
    EmotionComponent,
    GoalComponent,
    IdentityComponent,
    TimeBudgetComponent,
)
from agent_engine.simulation.simulation_state import SimulationState
from agent_engine.systems.causal_graph_system import CausalGraphSystem
from agent_engine.systems.components import QLearningComponent
from agent_engine.systems.q_learning_system import QLearningSystem


@pytest.fixture
def system_setup():
    """A comprehensive fixture to set up the QLearningSystem and its dependencies."""
    mock_state = create_autospec(SimulationState, instance=True)
    mock_bus = MagicMock()
    mock_encoder = MagicMock()
    mock_causal_system = create_autospec(CausalGraphSystem, instance=True)

    # Mock the causal system's estimate to return a predictable value
    mock_causal_system.estimate_causal_effect.return_value = 2.0

    system = QLearningSystem(
        simulation_state=mock_state,
        config=MagicMock(),
        cognitive_scaffold=MagicMock(),
        state_encoder=mock_encoder,
        causal_graph_system=mock_causal_system,
    )
    system.event_bus = mock_bus

    agent_id = "agent_1"

    # --- FIX: Sophisticated Mocking for get_component ---
    # This dictionary will hold the mock components for our agent.
    agent_components = {
        QLearningComponent: QLearningComponent(16, 1, 13, 0.001, "cpu"),
        IdentityComponent: MagicMock(),
        AffectComponent: MagicMock(),
        GoalComponent: MagicMock(),
        EmotionComponent: MagicMock(),
    }

    # Define a side_effect function that returns the correct component based on type
    def get_component_side_effect(entity_id_arg, component_type_arg):
        if entity_id_arg == agent_id:
            return agent_components.get(component_type_arg)
        return None

    mock_state.get_component.side_effect = get_component_side_effect
    # --- End Fix ---

    system.previous_states[agent_id] = np.ones(16)
    mock_encoder.encode_state.return_value = np.ones(16)
    mock_state.get_internal_state_features_for_entity.return_value = np.ones(1)

    return system, mock_state, mock_bus, mock_encoder, mock_causal_system, agent_id


@patch("agent_engine.systems.q_learning_system.QLearningSystem._perform_learning_step")
def test_on_action_executed_uses_causal_reward(mock_learning_step, system_setup):
    """
    Tests that the event handler correctly blends the observed reward with the
    causal estimate before calling the learning step.
    """
    system, _, _, _, mock_causal_system, agent_id = system_setup

    # Use spec=ActionInterface to make the mock pass the isinstance check
    mock_action_type = create_autospec(ActionInterface, instance=True)
    mock_action_type.action_id = "move"
    mock_action_type.get_feature_vector.return_value = [0.0] * 13

    event_data = {
        "entity_id": agent_id,
        "action_plan": MagicMock(action_type=mock_action_type),
        "action_outcome": ActionOutcome(True, "m", 10.0, {}),  # Observed reward is 10.0
        "current_tick": 1,
    }

    # ACT
    system.on_action_executed(event_data)

    # ASSERT
    mock_causal_system.estimate_causal_effect.assert_called_once_with(agent_id=agent_id, treatment_value="move")
    mock_learning_step.assert_called_once()
    blended_reward = mock_learning_step.call_args[0][6]
    assert blended_reward == pytest.approx(6.0)  # 0.5 * 10.0 (observed) + 0.5 * 2.0 (causal)


def test_on_action_executed_skips_if_no_previous_state(system_setup):
    """
    Tests that the system gracefully skips the learning step if no previous
    state is cached for the agent.
    """
    system, _, _, _, _, agent_id = system_setup
    del system.previous_states[agent_id]

    # FIX: Provide a complete event_data dictionary to avoid the KeyError
    event_data = {
        "entity_id": agent_id,
        "action_plan": MagicMock(action_type=create_autospec(ActionInterface)),
        "action_outcome": ActionOutcome(True, "m", 1.0, {}),
        "current_tick": 1,
    }

    with patch.object(system, "_perform_learning_step") as mock_learning_step:
        system.on_action_executed(event_data)
        mock_learning_step.assert_not_called()


@pytest.mark.asyncio
async def test_update_skips_inactive_agents(system_setup):
    """
    Tests that the update method does not cache states for inactive agents.
    """
    system, mock_state, _, mock_encoder, _, _ = system_setup

    inactive_agent_id = "inactive_agent"
    inactive_time_comp = TimeBudgetComponent(100, 0)
    inactive_time_comp.is_active = False

    # Simulate the system getting the inactive agent
    mock_state.get_entities_with_components.return_value = {
        inactive_agent_id: {TimeBudgetComponent: inactive_time_comp}
    }

    await system.update(current_tick=1)

    # The encoder should not be called for the inactive agent
    mock_encoder.encode_state.assert_not_called()
