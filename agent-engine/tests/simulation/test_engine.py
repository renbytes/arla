# tests/simulation/test_engine.py

from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from omegaconf import OmegaConf

# Subject under test
from agent_engine.simulation.engine import SimulationManager
from agent_core.core.ecs.component import TimeBudgetComponent, ActionPlanComponent

# --- Fixtures ---


@pytest.fixture
def mock_config():
    """Provides a mock OmegaConf config object."""
    conf_dict = {
        "simulation": {"steps": 3, "random_seed": 123},
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
    }


@pytest.fixture
@patch("agent_engine.simulation.engine.SystemManager")
@patch("agent_engine.simulation.engine.SimulationState")
@patch("agent_engine.simulation.engine.CognitiveScaffold")
@patch("agent_engine.simulation.engine.EventBus")
def sim_manager(mock_event_bus, mock_scaffold, mock_sim_state, mock_system_manager, mock_config, mock_dependencies):
    """Provides an initialized SimulationManager with all dependencies mocked."""

    # Configure the mock SimulationState to have one active entity
    active_entity_components = {TimeBudgetComponent: TimeBudgetComponent(100)}
    mock_sim_state.return_value.entities = {"agent1": active_entity_components}

    # Configure the mock SystemManager's update_all method to be awaitable
    mock_system_manager.return_value.update_all = AsyncMock()

    # Configure the mock DecisionSelector to return an action plan
    mock_dependencies["decision_selector"].select.return_value = ActionPlanComponent()

    # Configure the environment mock's to_dict() method
    # Tell the mock to return an empty dictionary when to_dict() is called.
    mock_dependencies["environment"].to_dict.return_value = {}

    manager = SimulationManager(config=mock_config, **mock_dependencies)
    return manager


# --- Test Cases ---


def test_initialization(sim_manager, mock_dependencies):
    """
    Tests that the SimulationManager correctly initializes and calls the scenario loader.
    """
    # Assert
    assert sim_manager.config["simulation"]["steps"] == 3
    # The scenario loader's load method should be called at the end of initialization
    mock_dependencies["scenario_loader"].load.assert_called_once()
    assert sim_manager.system_manager is not None


async def test_run_loop_executes_correct_number_of_steps(sim_manager):
    """
    Tests that the main run loop iterates for the number of steps specified in the config.
    """
    # Act
    await sim_manager.run()

    # Assert
    # The manager is configured for 3 steps. The system manager's update method
    # should be called exactly 3 times.
    assert sim_manager.system_manager.update_all.call_count == 3
    # Check that the calls were for ticks 0, 1, and 2
    ticks = [call.kwargs["current_tick"] for call in sim_manager.system_manager.update_all.call_args_list]
    assert ticks == [0, 1, 2]


async def test_run_loop_processes_entity_turn(sim_manager, mock_dependencies):
    """
    Tests that the logic for processing a single entity's turn is called correctly.
    """
    # Act
    await sim_manager.run()

    # Assert
    # For each of the 3 steps, the core decision-making pipeline should be called.
    assert mock_dependencies["action_generator"].generate.call_count == 3
    assert mock_dependencies["decision_selector"].select.call_count == 3

    # Check that the 'action_chosen' event was published each time
    assert sim_manager.event_bus.publish.call_count == 3
    event_name = sim_manager.event_bus.publish.call_args[0][0]
    assert event_name == "action_chosen"


async def test_run_loop_stops_when_no_active_entities(sim_manager):
    """
    Tests that the simulation ends early if all entities become inactive.
    """
    # Arrange
    # Reconfigure the mock state to have no active entities
    inactive_entity_components = {TimeBudgetComponent: TimeBudgetComponent(0)}  # budget is 0
    inactive_entity_components[TimeBudgetComponent].is_active = False
    sim_manager.simulation_state.entities = {"agent1": inactive_entity_components}

    # Act
    await sim_manager.run()

    # Assert
    # The loop should break on the first step, so no updates should be called.
    sim_manager.system_manager.update_all.assert_not_called()
