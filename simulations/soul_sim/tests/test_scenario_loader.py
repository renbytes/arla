import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, ANY

import pytest
from omegaconf import OmegaConf

# Classes from the core libraries needed for testing
from agent_core.core.ecs.component import TimeBudgetComponent
from agent_engine.simulation.simulation_state import SimulationState

# The class we are testing
from simulations.soul_sim.simulation.scenario_loader import ScenarioLoader

# Import some real components to verify dynamic loading
from simulations.soul_sim.components import (
    CombatComponent,
    HealthComponent,
    InventoryComponent,
    PositionComponent,
    ResourceComponent,
)

# --- Fixtures for Mocking Dependencies ---

@pytest.fixture
def mock_config() -> dict:
    """Provides a standard, nested configuration dictionary for tests."""
    return {
        "agent": {
            "cognitive": {
                "archetypes": {
                    "scout": {
                        "components": [
                            "agent_core.core.ecs.component.TimeBudgetComponent",
                            "simulations.soul_sim.components.PositionComponent",
                            "simulations.soul_sim.components.HealthComponent",
                        ]
                    },
                    "brute": {
                        "components": [
                            "agent_core.core.ecs.component.TimeBudgetComponent",
                            "simulations.soul_sim.components.PositionComponent",
                            "simulations.soul_sim.components.HealthComponent",
                            "simulations.soul_sim.components.CombatComponent",
                        ]
                    },
                    "bad_archetype": {
                        "components": [
                            "invalid.path.to.NonExistentComponent"
                        ]
                    }
                },
                "embeddings": {"main_embedding_dim": 4},
            },
            "foundational": {
                "vitals": {
                    "initial_time_budget": 1000.0,
                    "initial_health": 100.0,
                    "initial_resources": 10.0,
                },
                "attributes": {"initial_attack_power": 10.0},
            },
        },
        "learning": {
            "memory": {"affective_buffer_maxlen": 100},
            "q_learning": {"alpha": 0.001}
        }
    }


@pytest.fixture
def scenario_file(tmp_path: Path) -> Path:
    """Creates a temporary, valid scenario JSON file for testing."""
    scenario_data = {
        "name": "Test Scenario",
        "resources": {
            "resource_list": [
                {"pos": [10, 10], "params": {"value": 100, "type": "gold"}}
            ]
        },
        "groups": [
            {"type": "scout", "count": 1},
            {"type": "brute", "count": 2}
        ],
    }
    file_path = tmp_path / "test_scenario.json"
    file_path.write_text(json.dumps(scenario_data))
    return file_path


@pytest.fixture
def mock_simulation_state() -> MagicMock:
    """Provides a MagicMock of the SimulationState for tracking calls."""
    mock_env = MagicMock()
    mock_env.get_valid_positions.return_value = [(x, y) for x in range(5) for y in range(5)]

    mock_state = MagicMock(spec=SimulationState)
    mock_state.environment = mock_env
    mock_state.device = "cpu"

    # Store added components in a dictionary for detailed assertions
    mock_state.entities = {}

    def add_entity_side_effect(entity_id):
        mock_state.entities[entity_id] = {}

    def add_component_side_effect(entity_id, component):
        if entity_id not in mock_state.entities:
            add_entity_side_effect(entity_id)
        mock_state.entities[entity_id][type(component)] = component

    mock_state.add_entity.side_effect = add_entity_side_effect
    mock_state.add_component.side_effect = add_component_side_effect

    return mock_state


# --- Test Cases ---

def test_initialization_with_dict_config(scenario_file: Path, mock_config: dict):
    """Tests that the loader initializes correctly with a standard dictionary."""
    loader = ScenarioLoader(scenario_path=str(scenario_file), config=mock_config)
    assert loader.scenario_path == str(scenario_file)
    assert loader.config["agent"]["foundational"]["vitals"]["initial_health"] == 100.0
    assert loader.scenario_data["name"] == "Test Scenario"


def test_initialization_with_omegaconf(scenario_file: Path, mock_config: dict):
    """Tests that the loader correctly converts an OmegaConf DictConfig."""
    omega_conf = OmegaConf.create(mock_config)
    loader = ScenarioLoader(scenario_path=str(scenario_file), config=omega_conf)
    assert isinstance(loader.config, dict) # Should be converted
    assert loader.config["agent"]["cognitive"]["archetypes"]["scout"]["components"][0] == "agent_core.core.ecs.component.TimeBudgetComponent"


def test_load_raises_error_if_state_not_set(scenario_file: Path, mock_config: dict):
    """Tests that calling load() before setting the simulation state raises a RuntimeError."""
    loader = ScenarioLoader(scenario_path=str(scenario_file), config=mock_config)
    with pytest.raises(RuntimeError, match="SimulationState has not been set"):
        loader.load()


def test_load_happy_path(scenario_file: Path, mock_config: dict, mock_simulation_state: MagicMock):
    """Tests the successful loading of a valid scenario with multiple agent archetypes."""
    # Arrange
    loader = ScenarioLoader(scenario_path=str(scenario_file), config=mock_config)
    loader.set_simulation_state(mock_simulation_state)

    # Act
    loader.load()

    # Assert
    # 1. Check resource creation
    assert "resource_0" in mock_simulation_state.entities
    assert PositionComponent in mock_simulation_state.entities["resource_0"]

    # 2. Check agent creation counts
    assert "agent_0" in mock_simulation_state.entities  # 1st agent (scout)
    assert "agent_1" in mock_simulation_state.entities  # 2nd agent (brute)
    assert "agent_2" in mock_simulation_state.entities  # 3rd agent (brute)
    assert mock_simulation_state.add_entity.call_count == 4 # 3 agents + 1 resource

    # 3. Check component loading for different archetypes
    # Scout should have 3 components and no CombatComponent
    scout_components = mock_simulation_state.entities["agent_0"]
    assert len(scout_components) == 3
    assert TimeBudgetComponent in scout_components
    assert HealthComponent in scout_components
    assert CombatComponent not in scout_components

    # Brute should have 4 components, including CombatComponent
    brute_components = mock_simulation_state.entities["agent_1"]
    assert len(brute_components) == 4
    assert TimeBudgetComponent in brute_components
    assert HealthComponent in brute_components
    assert CombatComponent in brute_components
    assert brute_components[CombatComponent].attack_power == 10.0


def test_load_gracefully_handles_empty_scenario(tmp_path: Path, mock_config: dict, mock_simulation_state: MagicMock, caplog):
    """Tests that the loader handles a scenario file with no groups or resources without crashing."""
    # Arrange
    empty_scenario_path = tmp_path / "empty.json"
    empty_scenario_path.write_text(json.dumps({"name": "Empty", "groups": [], "resources": {}}))

    loader = ScenarioLoader(scenario_path=str(empty_scenario_path), config=mock_config)
    loader.set_simulation_state(mock_simulation_state)

    # Act
    with caplog.at_level(logging.WARNING):
        loader.load()

    # Assert
    mock_simulation_state.add_entity.assert_not_called()
    assert "Scenario file contains no agent groups" in caplog.text


def test_load_handles_unknown_archetype(tmp_path: Path, mock_config: dict, mock_simulation_state: MagicMock, caplog):
    """Tests that an agent with an archetype not in the config is created empty but does not crash."""
    # Arrange
    scenario_data = {"groups": [{"type": "unknown_archetype", "count": 1}]}
    scenario_path = tmp_path / "unknown_archetype.json"
    scenario_path.write_text(json.dumps(scenario_data))

    loader = ScenarioLoader(scenario_path=str(scenario_path), config=mock_config)
    loader.set_simulation_state(mock_simulation_state)

    # Act
    with caplog.at_level(logging.WARNING):
        loader.load()

    # Assert
    # The entity itself is created
    assert "agent_0" in mock_simulation_state.entities
    # But it has no components
    assert len(mock_simulation_state.entities["agent_0"]) == 0
    # And a warning was logged
    assert "No components listed for archetype 'unknown_archetype'" in caplog.text


def test_load_handles_bad_component_path(tmp_path: Path, mock_config: dict, mock_simulation_state: MagicMock, capsys):
    """Tests that an invalid component path in the config is skipped gracefully."""
    # Arrange
    scenario_data = {"groups": [{"type": "bad_archetype", "count": 1}]}
    scenario_path = tmp_path / "bad_archetype.json"
    scenario_path.write_text(json.dumps(scenario_data))

    loader = ScenarioLoader(scenario_path=str(scenario_path), config=mock_config)
    loader.set_simulation_state(mock_simulation_state)

    # Act
    loader.load()
    captured = capsys.readouterr()

    # Assert
    assert "agent_0" in mock_simulation_state.entities
    # The agent should be empty as its only component path was invalid
    assert len(mock_simulation_state.entities["agent_0"]) == 0
    # Check that an error was printed to the console
    assert "Failed to load component" in captured.out
    assert "invalid.path.to.NonExistentComponent" in captured.out


def test_create_agent_with_components(
    scenario_file: Path, mock_config: dict, mock_simulation_state: MagicMock
):
    """
    Unit test for the `_create_agent_with_components` method.

    This test verifies that for a given archetype, the correct components
    are dynamically imported, instantiated with the correct arguments based on
    their __init__ signature, and added to the simulation state.
    """
    # Arrange
    # Instantiate the loader, which we will use to call the private method
    loader = ScenarioLoader(scenario_path=str(scenario_file), config=mock_config)
    loader.set_simulation_state(mock_simulation_state)

    # Define the specific inputs for this test
    entity_id = "test_agent_scout"
    initial_pos = (10, 20)
    archetype_name = "scout"  # Defined in the mock_config fixture

    # Act
    # Directly call the private method we want to test
    loader._create_agent_with_components(entity_id, initial_pos, archetype_name)

    # Assert
    # 1. Verify the entity was created in the simulation state
    mock_simulation_state.add_entity.assert_called_with(entity_id)

    # 2. Verify the correct number of components were added (3 for a "scout")
    # The mock is configured to store components, so we can check its state
    created_components = mock_simulation_state.entities[entity_id]
    assert len(created_components) == 3

    # 3. Verify the PositionComponent was created correctly
    assert PositionComponent in created_components
    pos_comp = created_components[PositionComponent]
    assert isinstance(pos_comp, PositionComponent)
    assert pos_comp.position == initial_pos  # Check positional arg
    assert pos_comp.environment is mock_simulation_state.environment  # Check keyword arg

    # 4. Verify the HealthComponent was created correctly
    assert HealthComponent in created_components
    health_comp = created_components[HealthComponent]
    assert isinstance(health_comp, HealthComponent)
    # Verify it was instantiated with the correct value from config
    expected_health = mock_config["agent"]["foundational"]["vitals"]["initial_health"]
    assert health_comp.health == expected_health

    # 5. Verify the TimeBudgetComponent was created correctly
    assert TimeBudgetComponent in created_components
    time_comp = created_components[TimeBudgetComponent]
    assert isinstance(time_comp, TimeBudgetComponent)
    # Verify it was instantiated with the correct value from config
    expected_budget = mock_config["agent"]["foundational"]["vitals"]["initial_time_budget"]
    assert time_comp.initial_time_budget == expected_budget
