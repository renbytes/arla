# agent-engine/tests/simulation/test_engine.py

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agent_core.core.ecs.component import ActionPlanComponent, TimeBudgetComponent

# Subject under test
from agent_engine.simulation.engine import SimulationManager
from agent_engine.simulation.simulation_state import SimulationState
from agent_persist.models import SimulationSnapshot
from omegaconf import OmegaConf

# Fixtures


@pytest.fixture
def mock_config():
    """Provides a mock OmegaConf config object for testing."""
    conf_dict = {
        "simulation": {
            "steps": 3,
            "random_seed": 123,
            "log_directory": "/tmp/test_logs",
        },
        # The 'simulation_package' key must be at the top level of the config
        "simulation_package": "simulations.test_sim",
        "scenario_path": "/tmp/scenario.json",
        "enable_debug_logging": False,
    }
    return OmegaConf.create(conf_dict)


@pytest.fixture
def mock_dependencies():
    """Mocks all the major interfaces injected into the SimulationManager."""
    return {
        "environment": MagicMock(),
        "scenario_loader": MagicMock(),
        "action_generator": MagicMock(),
        "decision_selector": MagicMock(),
        "component_factory": MagicMock(),
        "db_logger": AsyncMock(),  # Use AsyncMock for awaitable methods
    }


@pytest.fixture
def sim_manager_with_mocks(mock_config, mock_dependencies):
    """
    Provides an initialized SimulationManager with all internal and external
    dependencies fully mocked for controlled testing.
    """
    with (
        patch(
            "agent_engine.simulation.engine.SystemManager", autospec=True
        ) as mock_system_manager,
        patch(
            "agent_engine.simulation.engine.SimulationState", autospec=True
        ) as mock_sim_state,
        patch(
            "agent_engine.simulation.engine.CognitiveScaffold", autospec=True
        ) as mock_scaffold,
        patch(
            "agent_engine.simulation.engine.EventBus", autospec=True
        ) as mock_event_bus,
        patch(
            "agent_engine.simulation.engine.FileStateStore", autospec=True
        ) as mock_file_store,
    ):
        # Configure Mock Instances
        mock_sim_state_instance = mock_sim_state.return_value
        mock_system_manager_instance = mock_system_manager.return_value

        # FIX: Add the '_systems' attribute to the mock SystemManager to prevent AttributeError
        mock_system_manager_instance._systems = []

        # Configure SimulationState to represent one active entity
        active_entity_components = {TimeBudgetComponent: TimeBudgetComponent(100)}
        mock_sim_state_instance.entities = {"agent_01": active_entity_components}
        mock_sim_state_instance.get_component.return_value = active_entity_components[
            TimeBudgetComponent
        ]

        # Mock the new to_snapshot() method on the SimulationState mock instance
        mock_sim_state_instance.to_snapshot.return_value = MagicMock(
            spec=SimulationSnapshot
        )

        # Configure SystemManager to have an awaitable `update_all` method
        mock_system_manager_instance.update_all = AsyncMock()

        # Configure external dependencies provided to the fixture
        mock_dependencies[
            "decision_selector"
        ].select.return_value = ActionPlanComponent()
        mock_dependencies["environment"].to_dict.return_value = {"world_data": "empty"}

        # Generate a valid UUID for the run_id to prevent the ValueError
        test_run_id = str(uuid.uuid4())

        # Instantiate the Manager
        manager = SimulationManager(
            config=mock_config, run_id=test_run_id, **mock_dependencies
        )

        # Attach Mocks to the Manager for Easy Access in Tests
        manager.mock_system_manager = mock_system_manager_instance
        manager.mock_sim_state = mock_sim_state_instance
        manager.mock_scaffold = mock_scaffold.return_value
        manager.mock_event_bus = mock_event_bus.return_value
        manager.mock_file_store = mock_file_store.return_value

        yield manager


# Test Cases


def test_initialization_and_scenario_loading(sim_manager_with_mocks, mock_dependencies):
    """
    Tests that the SimulationManager initializes its components correctly.
    Note: The manager itself does not call load(); that is the responsibility
    of the simulation's run script.
    """
    assert sim_manager_with_mocks is not None
    assert sim_manager_with_mocks.system_manager is not None
    assert sim_manager_with_mocks.simulation_state is not None

    # Verify that the scenario loader's load() method is NOT called during initialization.
    mock_dependencies["scenario_loader"].load.assert_not_called()


@pytest.mark.asyncio
async def test_full_lifecycle_and_correct_call_order(
    sim_manager_with_mocks, mock_dependencies
):
    """
    This is the key integration test. It verifies:
    1. The main loop runs for the configured number of steps.
    2. The critical order of operations is correct: systems update BEFORE entity turns are processed.
    3. State is saved at the end of the simulation.
    """
    # Arrange
    manager = sim_manager_with_mocks
    call_order_tracker = []

    manager.system_manager.update_all.side_effect = (
        lambda current_tick: call_order_tracker.append(
            f"update_all_tick_{current_tick}"
        )
    )
    # Use a side effect on a method that is called during the entity turn processing
    manager._process_entity_turn = MagicMock(
        side_effect=lambda entity_id, current_tick: call_order_tracker.append(
            f"process_turn_tick_{current_tick}"
        )
    )

    # Mock _get_active_entities to control the loop
    manager._get_active_entities = MagicMock(return_value=["agent_01"])

    # Act
    await manager.run()

    # Assert
    # 1. Verify the loop ran for 3 steps
    assert manager.system_manager.update_all.call_count == 3
    assert manager._process_entity_turn.call_count == 3

    # 2. **CRITICAL**: Verify the call order for each step
    expected_order = [
        "update_all_tick_0",
        "process_turn_tick_0",
        "update_all_tick_1",
        "process_turn_tick_1",
        "update_all_tick_2",
        "process_turn_tick_2",
    ]
    assert call_order_tracker == expected_order, (
        "The order of operations is incorrect! Systems must be updated before entity turns are processed."
    )

    # 3. Verify the final state was saved
    manager.mock_sim_state.to_snapshot.assert_called_once()
    manager.mock_file_store.save.assert_called_once_with(
        manager.mock_sim_state.to_snapshot.return_value
    )


@pytest.mark.asyncio
async def test_run_loop_stops_when_no_active_entities(
    sim_manager_with_mocks, mock_dependencies
):
    """
    Tests the edge case where the simulation ends prematurely because all
    agents have become inactive.
    """
    # Arrange
    manager = sim_manager_with_mocks
    manager._get_active_entities = MagicMock(return_value=[])

    # Act
    await manager.run()

    # Assert
    # The system update should not be called if there are no active entities
    manager.system_manager.update_all.assert_not_called()

    # The final state should still be saved.
    manager.mock_sim_state.to_snapshot.assert_called_once()
    manager.mock_file_store.save.assert_called_once()


def test_load_state_replaces_simulation_state(sim_manager_with_mocks):
    """
    Tests that loading from a checkpoint correctly replaces the existing
    simulation state with a new one created from a snapshot.
    """
    # Arrange
    manager = sim_manager_with_mocks
    original_sim_state = manager.simulation_state

    mock_snapshot = MagicMock(spec=SimulationSnapshot)
    manager.mock_file_store.load.return_value = mock_snapshot

    with patch(
        "agent_engine.simulation.engine.SimulationState.from_snapshot"
    ) as mock_from_snapshot:
        new_mock_state = MagicMock(spec=SimulationState)
        new_mock_state.current_tick = 100
        mock_from_snapshot.return_value = new_mock_state

        # Act
        manager.load_state("/fake/path/to/snapshot.json")

        # Assert
        manager.mock_file_store.load.assert_called_once()
        mock_from_snapshot.assert_called_once()
        assert manager.simulation_state is new_mock_state
        assert manager.simulation_state is not original_sim_state
        assert manager.simulation_state.current_tick == 100
