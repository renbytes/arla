# FILE: tests/systems/test_narrative_consensus_system.py

from unittest.mock import MagicMock

import pytest
from agent_core.core.ecs.component import (
    AffectComponent,
    BeliefSystemComponent,
    MemoryComponent,
)
from agent_core.core.schemas import Belief

from simulations.emergence_sim.components import PositionComponent
from simulations.emergence_sim.systems.narrative_consensus_system import (
    NarrativeConsensusSystem,
)


@pytest.fixture
def mock_simulation_state():
    """Provides a mocked SimulationState with essential objects."""
    state = MagicMock()
    state.event_bus = MagicMock()
    state.environment = MagicMock()
    state.config = MagicMock()
    state.cognitive_scaffold = MagicMock()
    state.entities = {}
    return state


@pytest.fixture
def system_and_agents(mock_simulation_state):
    """
    Sets up the NarrativeConsensusSystem and two agents (speaker, listener)
    with all the necessary components for testing.
    """
    # Initialize the system with mocked dependencies
    system = NarrativeConsensusSystem(
        simulation_state=mock_simulation_state,
        config=mock_simulation_state.config,
        cognitive_scaffold=mock_simulation_state.cognitive_scaffold,
    )

    # Create a speaker and a listener agent
    agents = {"speaker_id": {}, "listener_id": {}}
    for agent_id in agents:
        agents[agent_id] = {
            MemoryComponent: MemoryComponent(),
            AffectComponent: AffectComponent(affective_buffer_maxlen=10),
            BeliefSystemComponent: BeliefSystemComponent(),
            PositionComponent: PositionComponent(position=(1, 1), environment=mock_simulation_state.environment),
        }

    mock_simulation_state.entities = agents
    return system, agents


@pytest.mark.asyncio
async def test_listener_aligns_with_narrative(system_and_agents):
    """
    Tests that a listener adopts a new Belief when their experience ALIGNS
    with a shared narrative.
    """
    # 1. ARRANGE
    system, agents = system_and_agents
    listener_beliefs = agents["listener_id"][BeliefSystemComponent].belief_base

    # Mock the LLM to return "ALIGN"
    system.cognitive_scaffold.query = MagicMock(return_value="ALIGN")

    assert len(listener_beliefs) == 0

    # 2. ACT
    # Directly call the async method that contains the core logic
    await system._process_listener_response(
        listener_id="listener_id",
        speaker_id="speaker_id",
        shared_narrative="A shared story of success.",
        tick=100,
    )

    # 3. ASSERT
    # The listener should have adopted the narrative as a new belief
    assert len(listener_beliefs) == 1
    new_belief = list(listener_beliefs.values())[0]
    assert isinstance(new_belief, Belief)
    assert new_belief.statement == "A shared story of success."
    assert new_belief.confidence == 0.7


@pytest.mark.asyncio
async def test_listener_conflicts_with_narrative(system_and_agents):
    """
    Tests that a listener's cognitive dissonance increases when their
    experience CONFLICTS with a shared narrative.
    """
    # 1. ARRANGE
    system, agents = system_and_agents
    listener_affect = agents["listener_id"][AffectComponent]
    listener_affect.cognitive_dissonance = 0.1  # Set an initial value

    # Mock the LLM to return "CONFLICT"
    system.cognitive_scaffold.query = MagicMock(return_value="CONFLICT")

    # 2. ACT
    await system._process_listener_response(
        listener_id="listener_id",
        speaker_id="speaker_id",
        shared_narrative="A conflicting story.",
        tick=100,
    )

    # 3. ASSERT
    # The listener's cognitive dissonance should have increased from the initial value
    assert listener_affect.cognitive_dissonance > 0.1
    # The specific value is 0.1 + 0.5, clipped at 1.0
    assert listener_affect.cognitive_dissonance == 0.6
