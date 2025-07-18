# FILE: agent-engine/tests/simulation/test_simulation_state.py

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from agent_core.core.ecs.component import (
    Component,
    EmotionComponent,
    GoalComponent,
)
from agent_core.core.ecs.component_factory_interface import ComponentFactoryInterface
from agent_core.environment.interface import EnvironmentInterface

# Subject under test
from agent_engine.simulation.simulation_state import SimulationState

# Import snapshot models for testing
from agent_persist.models import SimulationSnapshot

# --- Fixtures ---


@pytest.fixture
def config():
    """Provides a mock config object using SimpleNamespace for clear attribute access."""
    return SimpleNamespace(
        agent=SimpleNamespace(cognitive=SimpleNamespace(embeddings=SimpleNamespace(main_embedding_dim=4)))
    )


@pytest.fixture
def sim_state(config):
    """Provides a fresh SimulationState instance for each test."""
    return SimulationState(config=config, device="cpu")


# --- Test Cases for Basic ECS Management ---


class TestECSOperations:
    """Tests for basic entity-component-system management."""

    def test_add_and_get_entity(self, sim_state):
        """
        Tests adding an entity and verifying its existence, including preventing duplicates.
        """
        sim_state.add_entity("agent1")
        assert "agent1" in sim_state.entities
        with pytest.raises(ValueError, match="Entity with ID agent1 already exists."):
            sim_state.add_entity("agent1")

    def test_add_and_get_component(self, sim_state):
        """
        Tests adding and retrieving a component for an entity.
        """

        class TestComponent(Component):
            def to_dict(self):
                return {}

            def validate(self, entity_id):
                return True, []

        sim_state.add_entity("agent1")
        test_comp = TestComponent()
        sim_state.add_component("agent1", test_comp)

        retrieved_comp = sim_state.get_component("agent1", TestComponent)
        assert retrieved_comp is test_comp
        assert sim_state.get_component("agent1", GoalComponent) is None
        with pytest.raises(ValueError, match="Entity with ID agent2 does not exist."):
            sim_state.add_component("agent2", test_comp)

    def test_remove_entity(self, sim_state):
        """
        Tests that an entity and all its components are removed correctly.
        """
        sim_state.add_entity("agent1")
        assert "agent1" in sim_state.entities
        sim_state.remove_entity("agent1")
        assert "agent1" not in sim_state.entities
        sim_state.remove_entity("non_existent_agent")

    def test_get_entities_with_components(self, sim_state):
        """
        Tests retrieving all entities that possess a specific set of components.
        """
        sim_state.add_entity("agent1")
        sim_state.add_component("agent1", EmotionComponent())
        sim_state.add_component("agent1", GoalComponent(embedding_dim=4))

        sim_state.add_entity("agent2")
        sim_state.add_component("agent2", EmotionComponent())

        sim_state.add_entity("agent3")
        sim_state.add_component("agent3", GoalComponent(embedding_dim=4))

        entities_both = sim_state.get_entities_with_components([EmotionComponent, GoalComponent])
        assert "agent1" in entities_both and len(entities_both) == 1

        entities_emotion = sim_state.get_entities_with_components([EmotionComponent])
        assert "agent1" in entities_emotion and "agent2" in entities_emotion
        assert len(entities_emotion) == 2

        all_entities = sim_state.get_entities_with_components([])
        assert len(all_entities) == 3


# CORRECTED: The entire TestInternalStateFeatures class has been deleted.
# This is because the method it was testing, 'get_internal_state_features_for_entity',
# was correctly removed from the SimulationState class. These tests are obsolete.


# --- Test Cases for Snapshotting ---


class TestSnapshotting:
    """Tests the to_snapshot and from_snapshot methods."""

    def test_to_snapshot_creates_correct_structure(self, sim_state):
        """
        Tests that to_snapshot correctly serializes the simulation state.
        """
        mock_env = MagicMock(spec=EnvironmentInterface)
        mock_env.to_dict.return_value = {"world_size": [10, 10]}
        sim_state.environment = mock_env
        sim_state.simulation_id = "test_sim_123"
        sim_state.current_tick = 150
        sim_state.add_entity("agent_01")
        sim_state.add_component("agent_01", EmotionComponent(valence=0.5, arousal=0.6))

        snapshot = sim_state.to_snapshot()

        assert isinstance(snapshot, SimulationSnapshot)
        assert snapshot.simulation_id == "test_sim_123"
        assert snapshot.current_tick == 150
        assert len(snapshot.agents) == 1
        comp_snapshot = snapshot.agents[0].components[0]
        assert comp_snapshot.component_type.endswith("EmotionComponent")
        assert comp_snapshot.data["valence"] == 0.5

    def test_from_snapshot_restores_state_correctly(self, config):
        """
        Tests that from_snapshot correctly reconstructs a SimulationState.
        """
        snapshot_data = {
            "simulation_id": "restored_sim_456",
            "current_tick": 200,
            "agents": [
                {
                    "agent_id": "agent_01",
                    "components": [
                        {
                            "component_type": "agent_core.core.ecs.component.EmotionComponent",
                            "data": {"valence": 0.8, "arousal": 0.3},
                        }
                    ],
                }
            ],
            "environment_state": {"weather": "sunny"},
        }
        snapshot = SimulationSnapshot(**snapshot_data)
        mock_factory = MagicMock(spec=ComponentFactoryInterface)
        mock_factory.create_component.return_value = EmotionComponent(valence=0.8, arousal=0.3)
        mock_env = MagicMock(spec=EnvironmentInterface)

        restored_state = SimulationState.from_snapshot(
            snapshot, config, mock_factory, mock_env, MagicMock(), MagicMock()
        )

        assert restored_state.simulation_id == "restored_sim_456"
        assert "agent_01" in restored_state.entities
        mock_env.restore_from_dict.assert_called_once_with({"weather": "sunny"})
        assert isinstance(
            restored_state.get_component("agent_01", EmotionComponent),
            EmotionComponent,
        )
