# FILE: tests/systems/test_ritualization_system.py

from unittest.mock import MagicMock

import pytest
from agent_core.core.ecs.component import (
    ActionPlanComponent,
    AffectComponent,
    MemoryComponent,
)

from simulations.emergence_sim.components import RitualComponent
from simulations.emergence_sim.systems.ritualization_system import RitualizationSystem


@pytest.fixture
def mock_simulation_state():
    """Provides a mocked SimulationState with an entity store."""
    state = MagicMock()
    state.entities = {}

    state.config = MagicMock()
    state.config.learning.memory.reflection_interval = 20

    # Helper to simplify getting components in tests
    def get_entities_with_components(comp_types):
        entities = {}
        for agent_id, comps in state.entities.items():
            if all(ct in comps for ct in comp_types):
                entities[agent_id] = comps
        return entities

    state.get_entities_with_components = get_entities_with_components

    return state


@pytest.fixture
def ritual_system(mock_simulation_state):
    """Returns a configured instance of the RitualizationSystem."""
    system = RitualizationSystem(
        simulation_state=mock_simulation_state,
        config=mock_simulation_state.config,
        cognitive_scaffold=MagicMock(),
    )
    system.ritual_codification_threshold = 3
    system.ritual_sequence_length = 2
    return system


@pytest.fixture
def agent_with_memory(mock_simulation_state):
    """Creates an agent with all components required by the RitualizationSystem."""
    agent_data = {
        "agent_1": {
            MemoryComponent: MemoryComponent(),
            RitualComponent: RitualComponent(),
            # CORRECTED: Added the two missing required components
            AffectComponent: AffectComponent(affective_buffer_maxlen=10),
            ActionPlanComponent: ActionPlanComponent(),
        }
    }
    mock_simulation_state.entities = agent_data
    return agent_data["agent_1"]


def create_event(action_id: str, reward: float) -> dict:
    """Helper to create a mock event dictionary for an agent's memory."""
    return {
        "action_plan": MagicMock(action_type=MagicMock(action_id=action_id)),
        "reward": reward,
    }


@pytest.mark.asyncio
async def test_ritual_is_codified_from_repeated_successful_sequence(ritual_system, agent_with_memory):
    """
    Tests that a sequence of successful actions following a negative trigger,
    when repeated enough times, is codified as a ritual.
    """
    # 1. ARRANGE
    mem_comp = agent_with_memory[MemoryComponent]
    ritual_comp = agent_with_memory[RitualComponent]

    ritual_sequence = ("move", "extract")
    for _ in range(3):  # This meets the threshold of 3
        mem_comp.episodic_memory.append(create_event("combat", -6.0))  # Trigger
        mem_comp.episodic_memory.append(create_event("move", 1.0))  # Successful action 1
        mem_comp.episodic_memory.append(create_event("extract", 1.0))  # Successful action 2

    # 2. ACT
    await ritual_system.update(current_tick=20)

    # 3. ASSERT
    assert "post_combat_loss" in ritual_comp.codified_rituals
    assert ritual_comp.codified_rituals["post_combat_loss"] == list(ritual_sequence)
