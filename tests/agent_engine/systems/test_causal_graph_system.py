# FILE: tests/agent_engine/systems/test_causal_graph_system.py
"""
Unit tests for the dowhy-based CausalGraphSystem.
"""

from unittest.mock import MagicMock, create_autospec, patch

import pytest
from agent_core.agents.actions.action_interface import ActionInterface
from agent_core.agents.actions.base_action import ActionOutcome
from agent_core.core.ecs.component import MemoryComponent, ValidationComponent
from agent_engine.simulation.simulation_state import SimulationState
from agent_engine.systems.causal_graph_system import CausalGraphSystem


@pytest.fixture
def system_setup():
    """Fixture to set up the system and its dependencies with realistic mocks."""
    mock_state = create_autospec(SimulationState, instance=True)
    mock_bus = MagicMock()
    mock_encoder = MagicMock()
    mock_config = MagicMock()
    mock_config.learning.memory.get.return_value = 50  # Set a default rebuild interval

    system = CausalGraphSystem(
        simulation_state=mock_state,
        config=mock_config,
        cognitive_scaffold=MagicMock(),
        state_node_encoder=mock_encoder,
    )
    system.event_bus = mock_bus

    agent_id = "agent_1"
    mem_comp = MemoryComponent()
    val_comp = ValidationComponent()
    # Add both required components for the system
    mock_state.entities = {
        agent_id: {MemoryComponent: mem_comp, ValidationComponent: val_comp}
    }
    # Configure get_component to return the correct component based on type
    mock_state.get_component.side_effect = lambda eid, comp_type: mock_state.entities[
        eid
    ].get(comp_type)

    return system, mock_state, mock_bus, mock_encoder, agent_id, mock_config


@pytest.mark.asyncio
async def test_on_action_executed_logs_data(system_setup):
    """
    Tests that the event handler correctly logs a record to the agent's causal_data.
    """
    system, mock_state, _, mock_encoder, agent_id, _ = system_setup

    # 1. Simulate the update call to cache the pre-action state
    mock_state.get_entities_with_components.return_value = {
        agent_id: mock_state.entities[agent_id]
    }
    mock_encoder.encode_state_for_causal_graph.return_value = (
        "STATE",
        "health_ok",
        "loc_A",
    )
    await system.update(current_tick=10)

    # 2. Fire the event
    mock_action_type = create_autospec(ActionInterface)
    mock_action_type.action_id = "move"
    event_data = {
        "entity_id": agent_id,
        "action_plan": MagicMock(action_type=mock_action_type),
        "action_outcome": ActionOutcome(True, "m", 1.0),
    }
    system.on_action_executed(event_data)

    # 3. Assert the data was recorded correctly
    mem_comp = mock_state.get_component(agent_id, MemoryComponent)
    assert len(mem_comp.causal_data) == 1
    record = mem_comp.causal_data[0]
    assert record["action"] == "move"
    assert record["state_health"] == "ok"


@pytest.mark.asyncio
@patch("agent_engine.systems.causal_graph_system.CausalModel")
@patch("agent_engine.systems.causal_graph_system.CausalModelValidator")
async def test_update_builds_model_periodically(
    mock_validator, mock_causal_model, system_setup
):
    """
    Tests that the update method periodically calls the model building logic.
    """
    system, mock_state, _, _, agent_id, _ = system_setup
    mem_comp = mock_state.get_component(agent_id, MemoryComponent)

    # Provide enough data to trigger a build
    mem_comp.causal_data = [
        {"state_health": "ok", "action": "move", "outcome": 1} for _ in range(25)
    ]
    mock_state.get_entities_with_components.return_value = {
        agent_id: mock_state.entities[agent_id]
    }
    # Mock the validator to return a dummy confidence score
    mock_validator.return_value.check_robustness.return_value = {"test": 0.8}

    # Run update at the rebuild interval (50)
    await system.update(current_tick=50)

    # Assert that the model was built and the validator was used
    mock_causal_model.assert_called_once()
    mock_validator.assert_called_once()
    assert mem_comp.causal_model is not None
    val_comp = mock_state.get_component(agent_id, ValidationComponent)
    assert val_comp.causal_model_confidence == 0.8


def test_estimate_causal_effect_success(system_setup):
    """
    Tests that the system can successfully estimate a causal effect if a model exists.
    """
    system, mock_state, _, _, agent_id, _ = system_setup
    mem_comp = mock_state.get_component(agent_id, MemoryComponent)

    # Mock the causal model
    mock_model = MagicMock()
    mock_model.estimate_effect.return_value = MagicMock(value=2.5)
    mem_comp.causal_model = mock_model

    # The system requires the category map to exist to perform estimation.
    # We must mock this state for the test to pass.
    system._category_maps[agent_id] = {"action": {"move": 0}}

    effect = system.estimate_causal_effect(agent_id, "move")

    assert effect == 2.5
    mock_model.identify_effect.assert_called_once()
    mock_model.estimate_effect.assert_called_once()


def test_estimate_causal_effect_no_model(system_setup):
    """
    Tests that the system returns None if no causal model is available.
    """
    system, mock_state, _, _, agent_id, _ = system_setup
    mock_state.get_component(agent_id, MemoryComponent).causal_model = None

    effect = system.estimate_causal_effect(agent_id, "move")

    assert effect is None
