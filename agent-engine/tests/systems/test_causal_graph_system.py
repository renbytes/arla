# tests/systems/test_causal_graph_system.py

from unittest.mock import MagicMock

import pytest
from agent_core.agents.actions.base_action import ActionOutcome
from agent_core.core.ecs.component import (
    EmotionComponent,
    GoalComponent,
    MemoryComponent,
)

# Subject under test
from agent_engine.systems.causal_graph_system import CausalGraphSystem

# --- Fixtures ---


@pytest.fixture
def mock_simulation_state():
    """Mocks the SimulationState and its contained components."""
    state = MagicMock()

    # Mock components
    mem_comp = MemoryComponent()
    # Set a previous state node to simulate the agent having been in a state before the action
    mem_comp.previous_state_node = "STATE", "health_ok", "at_base"

    emotion_comp = EmotionComponent(valence=0.5, arousal=0.8)
    goal_comp = GoalComponent(embedding_dim=4)

    state.entities = {
        "agent1": {
            MemoryComponent: mem_comp,
            EmotionComponent: emotion_comp,
            GoalComponent: goal_comp,
        }
    }
    state.get_entities_with_components.return_value = {"agent1": state.entities["agent1"]}
    return state


@pytest.fixture
def mock_state_node_encoder():
    """Mocks the StateNodeEncoderInterface to return a predictable state tuple."""
    encoder = MagicMock()
    encoder.encode_state_for_causal_graph.return_value = (
        "STATE",
        "health_good",
        "at_base",
    )
    return encoder


@pytest.fixture
def mock_event_bus():
    """Mocks the EventBus."""
    return MagicMock()


@pytest.fixture
def causal_graph_system(mock_simulation_state, mock_state_node_encoder, mock_event_bus):
    """Provides an initialized CausalGraphSystem with all dependencies mocked."""
    mock_simulation_state.event_bus = mock_event_bus

    system = CausalGraphSystem(
        simulation_state=mock_simulation_state,
        config={"learning": {"causal_decay_rate": 0.9}},
        cognitive_scaffold=MagicMock(),
        state_node_encoder=mock_state_node_encoder,
    )
    return system


# --- Test Cases ---


def test_on_action_executed_creates_causal_link(causal_graph_system, mock_simulation_state, mock_state_node_encoder):
    """
    Tests that a new causal link is added to the agent's memory after an action.
    """
    # Arrange
    action_outcome = ActionOutcome(success=True, message="", base_reward=10.0, details={"status": "SUCCESS"})
    mock_action_plan = MagicMock()
    mock_action_plan.action_type.name = "EXTRACT"
    mock_action_plan.intent.name = "SOLITARY"

    event_data = {
        "entity_id": "agent1",
        "action_outcome": action_outcome,
        "action_plan": mock_action_plan,
        "current_tick": 100,
    }

    mem_comp = mock_simulation_state.entities["agent1"][MemoryComponent]
    previous_node = mem_comp.previous_state_node
    expected_outcome_node = ("ACTION_OUTCOME", "EXTRACT", "SOLITARY", "SUCCESS")
    expected_current_state_node = ("STATE", "health_good", "at_base")

    # Act
    causal_graph_system.on_action_executed(event_data)

    # Assert
    # 1. Verify the state encoder was called to get the current state
    mock_state_node_encoder.encode_state_for_causal_graph.assert_called_once()

    # 2. Check that the link from the previous state to the outcome was created
    assert expected_outcome_node in mem_comp.causal_graph[previous_node]
    assert mem_comp.causal_graph[previous_node][expected_outcome_node] > 0

    # 3. Check that the link from the outcome to the new current state was created
    assert expected_current_state_node in mem_comp.causal_graph[expected_outcome_node]
    assert mem_comp.causal_graph[expected_outcome_node][expected_current_state_node] > 0

    # 4. Check that the 'previous_state_node' was updated for the next cycle
    assert mem_comp.previous_state_node == expected_current_state_node


@pytest.mark.asyncio
async def test_update_decays_link_strength(causal_graph_system, mock_simulation_state):
    """
    Tests that the passive update correctly decays the strength of existing causal links.
    """
    # Arrange
    mem_comp = mock_simulation_state.entities["agent1"][MemoryComponent]
    cause_node = ("STATE", "A")
    effect_node = ("OUTCOME", "B")
    initial_weight = 1.0
    mem_comp.causal_graph = {cause_node: {effect_node: initial_weight}}

    # Act
    # Call update with a tick that triggers the decay logic
    await causal_graph_system.update(current_tick=9)

    # Assert
    decayed_weight = mem_comp.causal_graph[cause_node][effect_node]
    assert decayed_weight < initial_weight
    assert decayed_weight == pytest.approx(initial_weight * 0.9)


@pytest.mark.asyncio
async def test_update_prunes_weak_links(causal_graph_system, mock_simulation_state):
    """
    Tests that very weak links are removed entirely after decay.
    """
    # Arrange
    mem_comp = mock_simulation_state.entities["agent1"][MemoryComponent]
    cause_node = ("STATE", "A")
    effect_node = ("OUTCOME", "B")
    weak_weight = 0.005  # Below the pruning threshold of 0.01
    mem_comp.causal_graph = {cause_node: {effect_node: weak_weight}}

    # Act
    await causal_graph_system.update(current_tick=19)

    # Assert
    # The entire entry for the cause node should be removed if it has no more effects
    assert cause_node not in mem_comp.causal_graph


def test_on_action_executed_missing_components(causal_graph_system, mock_simulation_state, mock_state_node_encoder):
    """
    Tests that the system handles an entity with missing components gracefully.
    """
    # Arrange
    event_data = {
        "entity_id": "agent_missing_comps",
        "action_outcome": MagicMock(),
        "action_plan": MagicMock(),
        "current_tick": 100,
    }
    mock_simulation_state.entities["agent_missing_comps"] = {}  # No components

    # Act
    causal_graph_system.on_action_executed(event_data)

    # Assert
    # The state encoder should not have been called
    mock_state_node_encoder.encode_state_for_causal_graph.assert_not_called()
