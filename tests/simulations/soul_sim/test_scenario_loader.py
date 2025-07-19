# simulations/soul_sim/tests/test_scenario_loader.py

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from agent_core.core.ecs.component import TimeBudgetComponent
from agent_engine.systems.components import QLearningComponent

# Import some real components to verify they are added correctly
from simulations.soul_sim.components import (
    CombatComponent,
    PositionComponent,
    ResourceComponent,
)

# Subject under test
from simulations.soul_sim.simulation.scenario_loader import ScenarioLoader

# Fixtures


@pytest.fixture
def mock_config(tmp_path):
    """Provides a mock config object using SimpleNamespace and a dummy scenario file path."""
    scenario_path = tmp_path / "test_scenario.json"
    scenario_path.touch()

    # Create a nested object that mimics the Pydantic model's structure
    return SimpleNamespace(
        scenario_path=str(scenario_path),
        agent=SimpleNamespace(
            cognitive=SimpleNamespace(embeddings=SimpleNamespace(main_embedding_dim=4, schema_embedding_dim=2)),
            foundational=SimpleNamespace(
                vitals=SimpleNamespace(
                    initial_time_budget=1000.0,
                    initial_health=100.0,
                    initial_resources=10.0,
                ),
                attributes=SimpleNamespace(initial_attack_power=10.0),
                lifespan_std_dev_percent=0.1,
            ),
        ),
        learning=SimpleNamespace(
            memory=SimpleNamespace(affective_buffer_maxlen=100),
            q_learning=SimpleNamespace(alpha=0.001, state_feature_dim=16, action_feature_dim=12),
        ),
        simulation=SimpleNamespace(random_seed=None),  # Add simulation section for completeness
    )


@pytest.fixture
def mock_simulation_state():
    """Provides a MagicMock of the SimulationState for tracking calls."""
    mock_env = MagicMock()
    mock_env.get_valid_positions.return_value = [(x, y) for x in range(20) for y in range(20)]

    mock_state = MagicMock()
    mock_state.environment = mock_env
    mock_state.device = "cpu"

    # Use a real dictionary to track entities and components for detailed assertions
    mock_state.entities = {}

    def add_entity_side_effect(entity_id):
        mock_state.entities[entity_id] = {}

    def add_component_side_effect(entity_id, component):
        mock_state.entities[entity_id][type(component)] = component

    mock_state.add_entity.side_effect = add_entity_side_effect
    mock_state.add_component.side_effect = add_component_side_effect

    return mock_state


@pytest.fixture
def scenario_file(tmp_path):
    """Creates a temporary, valid scenario JSON file for testing."""
    scenario_data = {
        "name": "Test Scenario",
        "resources": {
            "resource_list": [
                {
                    "pos": [10, 10],
                    "params": {
                        "resource_type": "gold",
                        "initial_health": 100,
                        "min_agents": 1,
                        "max_agents": 1,
                        "mining_rate": 1,
                        "reward_per_mine": 1,
                        "resource_yield": 1,
                        "respawn_time": 1,
                    },
                }
            ]
        },
        "groups": [{"type": "full_agent", "count": 2}],
    }
    file_path = tmp_path / "test_scenario.json"
    file_path.write_text(json.dumps(scenario_data))
    return file_path


# Test Cases


class TestScenarioLoader:
    def test_load_happy_path(self, mock_config, scenario_file, mock_simulation_state):
        """
        Tests the successful loading of a valid scenario with agents and resources.
        """
        # Arrange
        mock_config.scenario_path = str(scenario_file)
        # Add resource counts to the mock config for this test
        mock_config.environment = SimpleNamespace(
            num_single_resources=1,
            num_double_resources=0,
            num_triple_resources=0,
        )
        loader = ScenarioLoader(config=mock_config)
        loader.simulation_state = mock_simulation_state

        # Act
        loader.load()

        # Assert
        # 1. Check that a resource was created
        # Find the first entity that has a ResourceComponent
        resource_entity_id = next(
            (eid for eid, comps in mock_simulation_state.entities.items() if ResourceComponent in comps),
            None,
        )
        assert resource_entity_id is not None, "No resource entity was created"

        # 2. Check agent creation
        assert "full_agent_0" in mock_simulation_state.entities
        assert "full_agent_1" in mock_simulation_state.entities

        # 3. Verify components of one agent
        agent_comps = mock_simulation_state.entities["full_agent_0"]
        assert PositionComponent in agent_comps
        assert TimeBudgetComponent in agent_comps
        assert CombatComponent in agent_comps
        assert QLearningComponent in agent_comps
        assert agent_comps[CombatComponent].attack_power == 10.0

    def test_load_raises_error_if_state_not_set(self, mock_config):
        """
        Tests that calling load() before the simulation_state is injected raises a RuntimeError.
        """
        loader = ScenarioLoader(config=mock_config)
        with pytest.raises(RuntimeError, match="SimulationState must be set before calling load()"):
            loader.load()

    def test_load_raises_error_for_missing_scenario_file(self, mock_config, mock_simulation_state):
        """
        Tests that a ValueError is raised if the scenario_path in the config is invalid.
        """
        mock_config.scenario_path = "/path/to/non_existent_file.json"
        loader = ScenarioLoader(config=mock_config)
        loader.simulation_state = mock_simulation_state
        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_load_handles_empty_scenario(self, mock_config, tmp_path, mock_simulation_state, capsys):
        """
        Tests that the loader handles a scenario file with no groups or resources without crashing.
        """
        # Arrange
        empty_scenario_path = tmp_path / "empty.json"
        empty_scenario_path.write_text(json.dumps({"name": "Empty", "groups": [], "resources": {}}))
        mock_config.scenario_path = str(empty_scenario_path)
        mock_config.environment = SimpleNamespace(
            num_single_resources=0, num_double_resources=0, num_triple_resources=0
        )

        loader = ScenarioLoader(config=mock_config)
        loader.simulation_state = mock_simulation_state

        # Act
        loader.load()
        captured = capsys.readouterr()

        # Assert
        assert "full_agent_0" not in mock_simulation_state.entities
        assert "Created 0 agents" in captured.out

    def test_load_handles_unknown_archetype(self, mock_config, tmp_path, mock_simulation_state, capsys):
        """
        Tests that an agent with an archetype not defined in the loader is skipped gracefully.
        """
        # Arrange
        scenario_data = {"groups": [{"type": "unknown_archetype", "count": 1}]}
        scenario_path = tmp_path / "unknown_archetype.json"
        scenario_path.write_text(json.dumps(scenario_data))
        mock_config.scenario_path = str(scenario_path)
        mock_config.environment = SimpleNamespace(
            num_single_resources=0, num_double_resources=0, num_triple_resources=0
        )

        loader = ScenarioLoader(config=mock_config)
        loader.simulation_state = mock_simulation_state

        # Act
        loader.load()
        captured = capsys.readouterr()

        # Assert
        # Check that no AGENT entities were created
        agent_ids = [eid for eid in mock_simulation_state.entities if "agent" in eid]
        assert not agent_ids, "Agent entities were created for an unknown archetype."
        assert "Warning: Archetype 'unknown_archetype' not found" in captured.out

    @patch(
        "agent_core.agents.actions.action_registry.action_registry._actions",
        {"move": None, "extract": None, "combat": None},
    )
    def test_prepare_component_kwargs(self, mock_config, mock_simulation_state):
        """
        Tests the helper method that gathers constructor arguments from the config,
        especially verifying the dynamic calculation of Q-learning feature dimensions.
        """
        # Arrange
        loader = ScenarioLoader(config=mock_config)
        loader.simulation_state = mock_simulation_state

        # Act
        kwargs = loader._prepare_component_kwargs(initial_pos=(1, 2))

        # Assert
        # Check standard value retrieval
        assert kwargs["HealthComponent"]["initial_health"] == 100.0
        assert kwargs["environment"] is mock_simulation_state.environment

        # Check dynamic Q-learning dimensions
        q_kwargs = kwargs["QLearningComponent"]
        assert q_kwargs["action_feature_dim"] == 12
        assert q_kwargs["internal_state_dim"] == 31
        assert q_kwargs["state_feature_dim"] == 16
