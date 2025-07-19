# FILE: tests/simulation/test_scenario_loader_emergence_sim.py

import json
from unittest.mock import MagicMock

import pytest
from pytest_mock import mocker

from simulations.emergence_sim.config.schemas import EmergenceSimAppConfig

# Assume these are the paths to your classes
from simulations.emergence_sim.simulation.scenario_loader import EmergenceScenarioLoader

# Test Fixtures
# Fixtures are reusable setup functions for tests.


@pytest.fixture
def mock_simulation_state(mocker):
    """
    Creates a mock of the SimulationState object.
    We can inspect how many times its methods (like add_component) are called.
    """
    state = MagicMock()
    state.entities = {}
    # When add_entity is called, we simulate it by adding a mock to the dict
    state.add_entity.side_effect = lambda entity_id: state.entities.update({entity_id: {}})
    return state


@pytest.fixture
def mock_environment(mocker):
    """Creates a mock of the EmergenceEnvironment."""
    env = MagicMock()
    env.width = 15
    env.height = 15
    return env


@pytest.fixture
def mock_component_factory(mocker):
    """
    Creates a mock of the EmergenceComponentFactory.
    It simulates the creation of components for a new agent.
    """
    factory = MagicMock()
    # This mock function will return a list of mock components
    factory.create_agent_from_archetype.return_value = [
        MagicMock(name="PositionComponent"),
        MagicMock(name="InventoryComponent"),
        MagicMock(name="TimeBudgetComponent"),
    ]
    return factory


@pytest.fixture
def valid_config(tmp_path):
    """
    Creates a temporary valid scenario JSON file and returns a config
    object pointing to it. tmp_path is a pytest fixture for temp directories.
    """
    scenario_content = {
        "name": "Default Emergence Scenario",
        "description": "A starting scenario with 10 agents randomly placed.",
        "agents": [{"archetype": "default", "count": 10, "placement": "random"}],
    }
    # Create a temporary directory and file
    scenario_dir = tmp_path / "scenarios"
    scenario_dir.mkdir()
    scenario_file = scenario_dir / "test_scenario.json"
    scenario_file.write_text(json.dumps(scenario_content))

    # Use a minimal, valid config structure for the test
    # This ensures our test doesn't fail due to unrelated config issues
    config_data = {
        "scenario_path": str(scenario_file),
        "agent": {
            "initial_agent_count": 10,
            "foundational": {
                "vitals": {
                    "initial_time_budget": 500.0,
                    "initial_health": 100.0,
                    "initial_resources": 5.0,
                },
                "attributes": {"initial_attack_power": 0.0, "initial_speed": 1},
            },
        },
    }
    # We can create a partial config by telling Pydantic it's okay
    return EmergenceSimAppConfig.model_validate(config_data, from_attributes=True, context={"strict": False})


# Unit Tests


def test_scenario_loader_emergence_sim_load_success(
    valid_config, mock_simulation_state, mock_environment, mock_component_factory
):
    """
    Tests the "happy path": loading a valid scenario file.
    This test would have caught the 'foundational' and 'scenario_path' errors.
    """
    # 1. ARRANGE: Create the loader with mocks and a valid config
    loader = EmergenceScenarioLoader(
        config=valid_config,
        simulation_state=mock_simulation_state,
        environment=mock_environment,
        component_factory=mock_component_factory,
    )

    # 2. ACT: Run the method we want to test
    loader.load()

    # 3. ASSERT: Verify the outcome
    # Check that we tried to create 10 agents from the 'default' archetype
    assert mock_component_factory.create_agent_from_archetype.call_count == 10
    mock_component_factory.create_agent_from_archetype.assert_called_with(
        archetype_id="default",
        position=mocker.ANY,  # Position is random, so we accept any value
    )

    # Check that we added 10 entities to the simulation state
    assert mock_simulation_state.add_entity.call_count == 10

    # Check that we added 3 components for each of the 10 agents (30 total)
    assert mock_simulation_state.add_component.call_count == 30


def test_scenario_loader_emergence_sim_file_not_found(mock_simulation_state, mock_environment, mock_component_factory):
    """
    Tests that the loader raises FileNotFoundError for a bad path.
    """
    # 1. ARRANGE: Create a config pointing to a file that doesn't exist
    bad_config_data = {"scenario_path": "path/to/nonexistent/file.json"}
    bad_config = EmergenceSimAppConfig.model_validate(bad_config_data, from_attributes=True, context={"strict": False})

    loader = EmergenceScenarioLoader(
        config=bad_config,
        simulation_state=mock_simulation_state,
        environment=mock_environment,
        component_factory=mock_component_factory,
    )

    # 2. ACT & ASSERT: Expect a FileNotFoundError when .load() is called
    with pytest.raises(FileNotFoundError):
        loader.load()


def test_scenario_loader_emergence_sim_invalid_json(
    tmp_path, mock_simulation_state, mock_environment, mock_component_factory
):
    """
    Tests that the loader raises a JSONDecodeError for a malformed file.
    """
    # 1. ARRANGE: Create a temporary file with invalid JSON content
    scenario_file = tmp_path / "invalid.json"
    scenario_file.write_text("{'this is not valid json',}")

    bad_config_data = {"scenario_path": str(scenario_file)}
    bad_config = EmergenceSimAppConfig.model_validate(bad_config_data, from_attributes=True, context={"strict": False})

    loader = EmergenceScenarioLoader(
        config=bad_config,
        simulation_state=mock_simulation_state,
        environment=mock_environment,
        component_factory=mock_component_factory,
    )

    # 2. ACT & ASSERT: Expect a JSONDecodeError
    with pytest.raises(json.JSONDecodeError):
        loader.load()


def test_scenario_loader_emergence_sim_unsupported_placement(
    valid_config, mock_simulation_state, mock_environment, mock_component_factory
):
    """
    Tests that the loader logs a warning for an unsupported placement strategy.
    This assumes your loader has a `logger` attribute.
    """
    # 1. ARRANGE: Modify the valid config to have an unknown placement strategy
    scenario_data = json.loads(open(valid_config.scenario_path).read())
    scenario_data["agents"][0]["placement"] = "magical"  # an unsupported strategy

    # Write the modified content back to the file
    with open(valid_config.scenario_path, "w") as f:
        json.dump(scenario_data, f)

    loader = EmergenceScenarioLoader(
        config=valid_config,
        simulation_state=mock_simulation_state,
        environment=mock_environment,
        component_factory=mock_component_factory,
    )
    # Mock the logger to capture its output
    loader.logger = MagicMock()

    # 2. ACT
    loader.load()

    # 3. ASSERT: Check that a warning was logged
    loader.logger.warning.assert_called_once()
    assert "Unsupported placement strategy" in loader.logger.warning.call_args[0][0]
