# agent-engine/tests/simulation/test_engine.py

from unittest.mock import MagicMock, patch, AsyncMock, call
import pytest
from omegaconf import OmegaConf

# Subject under test
from agent_engine.simulation.engine import SimulationManager
from agent_core.core.ecs.component import TimeBudgetComponent, ActionPlanComponent
from agent_engine.simulation.simulation_state import SimulationState
from agent_persist.models import SimulationSnapshot

# --- Fixtures ---

@pytest.fixture
def mock_config():
    """Provides a mock OmegaConf config object for testing."""
    conf_dict = {
        "simulation": {
            "steps": 3,
            "random_seed": 123,
            "log_directory": "/tmp/test_logs"
        },
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
        "db_logger": MagicMock(),
    }


@pytest.fixture
def sim_manager_with_mocks(mock_config, mock_dependencies):
    """
    Provides an initialized SimulationManager with all internal and external
    dependencies fully mocked for controlled testing.
    """
    # Patch all classes that SimulationManager instantiates internally
    with patch("agent_engine.simulation.engine.SystemManager") as MockSystemManager, \
         patch("agent_engine.simulation.engine.SimulationState") as MockSimState, \
         patch("agent_engine.simulation.engine.CognitiveScaffold") as MockScaffold, \
         patch("agent_engine.simulation.engine.EventBus") as MockEventBus, \
         patch("agent_engine.simulation.engine.FileStateStore") as MockFileStore, \
         patch("agent_engine.simulation.engine.create_snapshot_from_state") as mock_create_snapshot:

        # --- Configure Mocks ---

        # Mock SimulationState to represent one active entity
        mock_sim_state_instance = MockSimState.return_value
        active_entity_components = {TimeBudgetComponent: TimeBudgetComponent(100)}
        mock_sim_state_instance.entities = {"agent_01": active_entity_components}
        # Make get_component return the right component
        mock_sim_state_instance.get_component.return_value = active_entity_components[TimeBudgetComponent]

        # Mock SystemManager to have an awaitable `update_all` method
        mock_system_manager_instance = MockSystemManager.return_value
        mock_system_manager_instance.update_all = AsyncMock()

        # Mock DecisionSelector to return a valid action plan
        mock_dependencies["decision_selector"].select.return_value = ActionPlanComponent()

        # Mock the environment's to_dict method, which is called during save_state
        mock_dependencies["environment"].to_dict.return_value = {"world_data": "empty"}

        # --- Instantiate the Manager ---
        manager = SimulationManager(
            config=mock_config,
            **mock_dependencies
        )

        # Attach mocks to the manager instance for easy access in tests
        manager.system_manager = mock_system_manager_instance
        manager.simulation_state = mock_sim_state_instance
        manager.event_bus = MockEventBus.return_value
        manager.mock_file_store = MockFileStore.return_value
        manager.mock_create_snapshot = mock_create_snapshot

        yield manager

# --- Test Cases ---

def test_initialization_and_scenario_loading(sim_manager_with_mocks, mock_dependencies):
    """
    Tests that the SimulationManager initializes its components correctly.
    Note: The manager itself does not call load(); that is the responsibility
    of the simulation's run script.
    """
    # Assertions are implicitly handled by the fixture setup.
    # If the fixture builds without error, initialization is successful.
    assert sim_manager_with_mocks is not None
    assert sim_manager_with_mocks.system_manager is not None
    assert sim_manager_with_mocks.simulation_state is not None

    # Verify that the scenario loader's load() method is NOT called during initialization.
    mock_dependencies["scenario_loader"].load.assert_not_called()


@pytest.mark.asyncio
async def test_full_lifecycle_and_correct_call_order(sim_manager_with_mocks, mock_dependencies):
    """
    This is the key integration test. It verifies:
    1. The main loop runs for the configured number of steps.
    2. The critical order of operations is correct: systems update BEFORE entity turns are processed.
    3. State is saved at the end of the simulation.
    """
    # --- Arrange ---
    manager = sim_manager_with_mocks
    call_order_tracker = []

    # Set up side effects to track the call order of the two main operations in the loop
    manager.system_manager.update_all.side_effect = lambda current_tick: call_order_tracker.append(f"update_all_tick_{current_tick}")
    # We track the turn via the action generator, which is the first step in a turn.
    mock_dependencies["action_generator"].generate.side_effect = lambda simulation_state, entity_id, current_tick: call_order_tracker.append(f"process_turn_tick_{current_tick}")

    # --- Act ---
    await manager.run()

    # --- Assert ---
    # 1. Verify the loop ran for 3 steps
    assert manager.system_manager.update_all.call_count == 3
    assert mock_dependencies["action_generator"].generate.call_count == 3

    # 2. **CRITICAL**: Verify the call order for each step
    # This assertion directly tests for and prevents the "Skipping Q-update" bug.
    expected_order = [
        'update_all_tick_0', 'process_turn_tick_0',
        'update_all_tick_1', 'process_turn_tick_1',
        'update_all_tick_2', 'process_turn_tick_2',
    ]
    assert call_order_tracker == expected_order, \
        "The order of operations is incorrect! Systems must be updated before entity turns are processed."

    # 3. Verify the final state was saved
    manager.mock_create_snapshot.assert_called_once_with(manager.simulation_state)
    manager.mock_file_store.save.assert_called_once_with(manager.mock_create_snapshot.return_value)


@pytest.mark.asyncio
async def test_run_loop_stops_when_no_active_entities(sim_manager_with_mocks, mock_dependencies):
    """
    Tests the edge case where the simulation ends prematurely because all
    agents have become inactive.
    """
    # --- Arrange ---
    manager = sim_manager_with_mocks

    # Configure the mock state to have no active entities
    inactive_entity_components = {TimeBudgetComponent: TimeBudgetComponent(0)}
    inactive_entity_components[TimeBudgetComponent].is_active = False
    manager.simulation_state.entities = {"agent1": inactive_entity_components}
    manager.simulation_state.get_component.return_value = inactive_entity_components[TimeBudgetComponent]

    # --- Act ---
    await manager.run()

    # --- Assert ---
    # The loop should break on the very first step.
    # Neither the system update nor the entity processing should have been called.
    manager.system_manager.update_all.assert_not_called()
    mock_dependencies["action_generator"].generate.assert_not_called()

    # The final state should still be saved.
    manager.mock_create_snapshot.assert_called_once()
    manager.mock_file_store.save.assert_called_once()


def test_load_state_replaces_simulation_state(sim_manager_with_mocks):
    """
    Tests that loading from a checkpoint correctly replaces the existing
    simulation state with a new one created from a snapshot.
    """
    # --- Arrange ---
    manager = sim_manager_with_mocks
    original_sim_state = manager.simulation_state

    # Mock the return value of loading a file
    mock_snapshot = MagicMock(spec=SimulationSnapshot)
    manager.mock_file_store.load.return_value = mock_snapshot

    # Mock the class method that creates a state from a snapshot
    with patch("agent_engine.simulation.engine.SimulationState.from_snapshot") as mock_from_snapshot:
        new_mock_state = MagicMock(spec=SimulationState)
        new_mock_state.current_tick = 100 # Give it a different tick to verify
        mock_from_snapshot.return_value = new_mock_state

        # --- Act ---
        manager.load_state("/fake/path/to/snapshot.json")

        # --- Assert ---
        # Verify the file store was used to load the data
        manager.mock_file_store.load.assert_called_once_with()

        # Verify the factory method was called with the loaded data
        mock_from_snapshot.assert_called_once()

        # Verify the manager's simulation_state attribute was updated to the new instance
        assert manager.simulation_state is new_mock_state
        assert manager.simulation_state is not original_sim_state
        assert manager.simulation_state.current_tick == 100
