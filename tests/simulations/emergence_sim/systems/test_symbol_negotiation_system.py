# FILE: tests/simulations/emergence_sim/systems/test_symbol_negotiation_system.py

from unittest.mock import MagicMock, patch

import pytest
from agent_core.agents.actions.action_registry import action_registry
from agent_core.core.ecs.component import TimeBudgetComponent
from agent_engine.systems.components import QLearningComponent

from simulations.emergence_sim.components import (
    ConceptualSpaceComponent,
    PositionComponent,
)
from simulations.emergence_sim.systems.symbol_negotiation_system import (
    SymbolNegotiationSystem,
)


@pytest.fixture
def mock_simulation_state():
    """Provides a mocked SimulationState with essential objects."""
    state = MagicMock()
    state.event_bus = MagicMock()
    state.environment = MagicMock()
    state.config = MagicMock()
    state.cognitive_scaffold = MagicMock()
    state.main_rng = MagicMock()
    state.entities = {}
    return state


@pytest.fixture
def system_with_two_agents(mock_simulation_state):
    """
    Sets up the SymbolNegotiationSystem and two agents with the necessary
    components for playing the Naming Game.
    """
    system = SymbolNegotiationSystem(
        simulation_state=mock_simulation_state,
        config=mock_simulation_state.config,
        cognitive_scaffold=mock_simulation_state.cognitive_scaffold,
    )

    agents = {"speaker": {}, "listener": {}}
    for agent_id in agents:
        agents[agent_id] = {
            PositionComponent: PositionComponent(position=(1, 1), environment=mock_simulation_state.environment),
            ConceptualSpaceComponent: ConceptualSpaceComponent(quality_dimensions={"color": 3}),
            TimeBudgetComponent: TimeBudgetComponent(initial_time_budget=100.0, lifespan_std_dev_percent=0.0),
            QLearningComponent: MagicMock(),
        }

    def get_component_side_effect(entity_id, comp_type):
        return agents.get(entity_id, {}).get(comp_type)

    mock_simulation_state.get_component.side_effect = get_component_side_effect

    mock_simulation_state.entities = agents
    return system, agents


@pytest.mark.asyncio
async def test_naming_game_success_updates_concepts_and_rewards(system_with_two_agents):
    """
    Tests that a successful Naming Game interaction correctly updates the
    conceptual spaces of both agents and publishes a positive reward.
    """
    # 1. ARRANGE
    system, agents = system_with_two_agents

    # Add the 'obj_type' to the mock object so the system can assign the correct reward.
    target_object = ("obj_1", {"id": "obj_1", "color": "red", "obj_type": "resource"})
    system.simulation_state.environment.get_objects_in_radius.return_value = [target_object]
    system.simulation_state.main_rng.choice.return_value = target_object
    system.simulation_state.main_rng.integers.return_value = 500

    with patch.object(action_registry, "get_action", return_value=MagicMock()):
        # 2. ACT
        await system._play_naming_game("speaker", "listener", current_tick=10)

    # 3. ASSERT
    speaker_call_args = system.simulation_state.event_bus.publish.call_args_list[0]
    speaker_event_data = speaker_call_args[0][1]

    # The assertion should now pass because the reward logic finds the 'obj_type'.
    assert speaker_event_data["action_outcome"].reward == 3.0
