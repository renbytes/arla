from unittest.mock import MagicMock

import pytest

from simulations.emergence_sim.components import InventoryComponent, PositionComponent
from simulations.emergence_sim.systems.object_interaction_system import (
    ObjectInteractionSystem,
)

# Fixtures


@pytest.fixture
def mock_simulation_state():
    """Provides a mocked SimulationState with an environment and entity store."""
    state = MagicMock()
    state.environment = MagicMock()
    state.entities = {}

    def get_entities_with_components(comp_types):
        return state.entities

    state.get_entities_with_components = get_entities_with_components

    return state


@pytest.fixture
def interaction_system(mock_simulation_state):
    """Returns an instance of the ObjectInteractionSystem."""
    return ObjectInteractionSystem(
        simulation_state=mock_simulation_state,
        config=MagicMock(),
        cognitive_scaffold=MagicMock(),
    )


# ObjectInteractionSystem Tests


@pytest.mark.asyncio
async def test_interaction_with_resource_increases_inventory(interaction_system, mock_simulation_state):
    """
    Tests that an agent's resources increase when it moves onto a 'resource' object.
    """
    # 1. ARRANGE
    agent_id = "agent_1"
    agent_pos = (3, 3)

    # Set up the agent with components
    pos_comp = PositionComponent(position=agent_pos, environment=mock_simulation_state.environment)
    inv_comp = InventoryComponent(initial_resources=10.0)
    mock_simulation_state.entities[agent_id] = {
        PositionComponent: pos_comp,
        InventoryComponent: inv_comp,
    }

    # Mock the environment to return a resource object at the agent's position
    resource_object = {"id": "obj_1", "obj_type": "resource", "value": 15}
    mock_simulation_state.environment.get_object_at.return_value = resource_object

    # 2. ACT
    await interaction_system.update(current_tick=1)

    # 3. ASSERT
    # The agent's inventory should have increased by the object's value
    assert inv_comp.current_resources == 25.0  # 10 + 15

    # The object should be removed from the world
    mock_simulation_state.environment.objects.__delitem__.assert_called_once_with("obj_1")


@pytest.mark.asyncio
async def test_interaction_with_hazard_decreases_inventory(interaction_system, mock_simulation_state):
    """
    Tests that an agent's resources decrease when it moves onto a 'hazard' object.
    """
    # 1. ARRANGE
    agent_id = "agent_1"
    agent_pos = (4, 4)

    pos_comp = PositionComponent(position=agent_pos, environment=mock_simulation_state.environment)
    inv_comp = InventoryComponent(initial_resources=20.0)
    mock_simulation_state.entities[agent_id] = {
        PositionComponent: pos_comp,
        InventoryComponent: inv_comp,
    }

    hazard_object = {"id": "obj_2", "obj_type": "hazard", "value": 10}
    mock_simulation_state.environment.get_object_at.return_value = hazard_object

    # 2. ACT
    await interaction_system.update(current_tick=1)

    # 3. ASSERT
    assert inv_comp.current_resources == 10.0

    # Assert that the system called the delete operation on the objects dictionary
    mock_simulation_state.environment.objects.__delitem__.assert_called_once_with("obj_2")
