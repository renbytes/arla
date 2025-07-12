# agent-engine/tests/simulation/test_simulation_state.py

from unittest.mock import MagicMock

import numpy as np
import pytest
from agent_core.core.ecs.component import (
    AffectComponent,
    Component,
    EmotionComponent,
    GoalComponent,
    IdentityComponent,
)
from agent_core.core.ecs.component_factory_interface import ComponentFactoryInterface
from agent_core.environment.interface import EnvironmentInterface
from agent_engine.cognition.identity.domain_identity import (
    IdentityDomain,
    MultiDomainIdentity,
)

# Subject under test
from agent_engine.simulation.simulation_state import SimulationState

# Import snapshot models for testing
from agent_persist.models import SimulationSnapshot

# --- Fixtures ---


@pytest.fixture
def config():
    """Provides a sample config for the simulation state."""
    # Use a small, non-standard embedding dim to ensure it's read from config
    return {"agent": {"cognitive": {"embeddings": {"main_embedding_dim": 4}}}}


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
        # Removing a non-existent entity should not raise an error
        sim_state.remove_entity("non_existent_agent")

    def test_get_entities_with_components(self, sim_state):
        """
        Tests retrieving all entities that possess a specific set of components.
        """
        # Arrange: Create entities with different sets of components
        sim_state.add_entity("agent1")
        sim_state.add_component("agent1", EmotionComponent())
        sim_state.add_component("agent1", GoalComponent(embedding_dim=4))

        sim_state.add_entity("agent2")
        sim_state.add_component("agent2", EmotionComponent())

        sim_state.add_entity("agent3")
        sim_state.add_component("agent3", GoalComponent(embedding_dim=4))

        # Act & Assert
        # Case 1: Find entities with both Emotion and Goal components
        entities = sim_state.get_entities_with_components([EmotionComponent, GoalComponent])
        assert "agent1" in entities
        assert "agent2" not in entities
        assert "agent3" not in entities
        assert len(entities) == 1

        # Case 2: Find entities with only EmotionComponent
        entities_emotion = sim_state.get_entities_with_components([EmotionComponent])
        assert "agent1" in entities_emotion
        assert "agent2" in entities_emotion
        assert len(entities_emotion) == 2

        # Case 3: Edge case with an empty list of components (should return all)
        all_entities = sim_state.get_entities_with_components([])
        assert len(all_entities) == 3


# --- Test Cases for get_internal_state_features_for_entity ---


class TestInternalStateFeatures:
    """Tests for the internal state feature vector generation."""

    @pytest.fixture
    def full_components(self, config):
        """Provides a full set of mock cognitive components."""
        embedding_dim = config["agent"]["cognitive"]["embeddings"]["main_embedding_dim"]

        affect_comp = AffectComponent(affective_buffer_maxlen=10)
        affect_comp.prediction_delta_magnitude = 0.5
        affect_comp.predictive_delta_smooth = 0.6

        emotion_comp = EmotionComponent(valence=0.7, arousal=0.8)

        goal_comp = GoalComponent(embedding_dim=embedding_dim)
        goal_comp.current_symbolic_goal = "test_goal"
        goal_comp.symbolic_goals_data["test_goal"] = {"embedding": np.ones(embedding_dim)}

        id_comp = IdentityComponent(multi_domain_identity=MultiDomainIdentity(embedding_dim=embedding_dim))

        return affect_comp, emotion_comp, goal_comp, id_comp

    def test_get_internal_state_features_all_present(self, sim_state, full_components, config):
        """
        Tests feature generation when all cognitive components are present.
        """
        # Arrange
        affect_comp, emotion_comp, goal_comp, id_comp = full_components
        embedding_dim = config["agent"]["cognitive"]["embeddings"]["main_embedding_dim"]
        num_domains = len(IdentityDomain)
        # affect/emo + goal + identity + flags
        expected_len = 4 + embedding_dim + (embedding_dim * num_domains) + 3

        # Act
        features = sim_state.get_internal_state_features_for_entity(id_comp, affect_comp, goal_comp, emotion_comp)

        # Assert
        assert features.shape == (expected_len,)
        assert features.dtype == np.float32
        # Check flags are all 1.0, indicating all components were present
        assert np.all(features[-3:] == 1.0)
        # Check that the first part of the vector contains emotion/affect data
        assert features[0] == pytest.approx(0.7)  # valence
        assert features[3] == pytest.approx(0.6)  # predictive_delta_smooth
        # Check that goal embedding is present
        assert np.all(features[4 : 4 + embedding_dim] == 1.0)

    def test_get_internal_state_features_some_missing(self, sim_state, full_components, config):
        """
        Tests feature generation when some cognitive components are missing,
        ensuring correct zero-padding and flag setting.
        """
        # Arrange
        _, _, goal_comp, id_comp = full_components
        embedding_dim = config["agent"]["cognitive"]["embeddings"]["main_embedding_dim"]
        num_domains = len(IdentityDomain)
        expected_len = 4 + embedding_dim + (embedding_dim * num_domains) + 3

        # Act: Call with emotion and affect components as None
        features = sim_state.get_internal_state_features_for_entity(id_comp, None, goal_comp, None)

        # Assert
        assert features.shape == (expected_len,)
        # Check that the first 4 features (for affect/emotion) are zeros
        assert np.all(features[:4] == 0.0)
        # Check that the first flag is 0.0, but the others are 1.0
        flags = features[-3:]
        assert flags[0] == 0.0
        assert flags[1] == 1.0
        assert flags[2] == 1.0

    def test_get_internal_state_features_all_missing(self, sim_state, config):
        """
        Tests the edge case where all cognitive components are None.
        The resulting vector should be all zeros except for the flags.
        """
        # Arrange
        embedding_dim = config["agent"]["cognitive"]["embeddings"]["main_embedding_dim"]
        num_domains = len(IdentityDomain)
        expected_len = 4 + embedding_dim + (embedding_dim * num_domains) + 3

        # Act
        features = sim_state.get_internal_state_features_for_entity(None, None, None, None)

        # Assert
        assert features.shape == (expected_len,)
        # Check that all flags are 0.0
        assert np.all(features[-3:] == 0.0)
        # Check that all feature values (everything except the flags) are 0.0
        assert np.all(features[:-3] == 0.0)


# --- Test Cases for Snapshotting ---


class TestSnapshotting:
    """Tests the to_snapshot and from_snapshot methods."""

    def test_to_snapshot_creates_correct_structure(self, sim_state):
        """
        Tests that to_snapshot correctly serializes the simulation state.
        """
        # Arrange
        mock_env = MagicMock(spec=EnvironmentInterface)
        mock_env.to_dict.return_value = {"world_size": [10, 10]}
        sim_state.environment = mock_env

        sim_state.simulation_id = "test_sim_123"
        sim_state.current_tick = 150

        sim_state.add_entity("agent_01")
        sim_state.add_component("agent_01", EmotionComponent(valence=0.5, arousal=0.6))

        # Act
        snapshot = sim_state.to_snapshot()

        # Assert
        assert isinstance(snapshot, SimulationSnapshot)
        assert snapshot.simulation_id == "test_sim_123"
        assert snapshot.current_tick == 150
        assert snapshot.environment_state == {"world_size": [10, 10]}
        assert len(snapshot.agents) == 1

        agent_snapshot = snapshot.agents[0]
        assert agent_snapshot.agent_id == "agent_01"
        assert len(agent_snapshot.components) == 1

        comp_snapshot = agent_snapshot.components[0]
        assert comp_snapshot.component_type == "agent_core.core.ecs.component.EmotionComponent"
        assert comp_snapshot.data["valence"] == 0.5

    def test_from_snapshot_restores_state_correctly(self, config):
        """
        Tests that from_snapshot correctly reconstructs a SimulationState.
        """
        # Arrange
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

        # Mock dependencies for from_snapshot
        mock_factory = MagicMock(spec=ComponentFactoryInterface)
        mock_factory.create_component.return_value = EmotionComponent(valence=0.8, arousal=0.3)
        mock_env = MagicMock(spec=EnvironmentInterface)
        mock_bus = MagicMock()
        mock_logger = MagicMock()

        # Act
        restored_state = SimulationState.from_snapshot(snapshot, config, mock_factory, mock_env, mock_bus, mock_logger)

        # Assert
        assert restored_state.simulation_id == "restored_sim_456"
        assert restored_state.current_tick == 200
        assert "agent_01" in restored_state.entities
        mock_env.restore_from_dict.assert_called_once_with({"weather": "sunny"})
        mock_factory.create_component.assert_called_once_with(
            "agent_core.core.ecs.component.EmotionComponent",
            {"valence": 0.8, "arousal": 0.3},
        )
        assert isinstance(
            restored_state.get_component("agent_01", EmotionComponent),
            EmotionComponent,
        )

    def test_snapshot_roundtrip(self, sim_state, config):
        """
        Tests that saving a state to a snapshot and restoring it results
        in an equivalent state.
        """
        # --- Arrange ---

        # 1. Create a simple mock component factory for the test
        class MockFactory(ComponentFactoryInterface):
            def create_component(self, component_type, data):
                # Simple factory that only knows about EmotionComponent
                if component_type.endswith("EmotionComponent"):
                    return EmotionComponent(**data)
                return Component()

        # 2. Set up the original state
        original_state = sim_state
        original_state.simulation_id = "roundtrip_sim"
        original_state.current_tick = 50
        original_state.environment = MagicMock(spec=EnvironmentInterface)
        original_state.environment.to_dict.return_value = {"type": "mock"}
        original_state.add_entity("agent_x")
        original_state.add_component("agent_x", EmotionComponent(valence=0.9))

        # --- Act ---

        # 1. Save to snapshot
        snapshot = original_state.to_snapshot()

        # 2. Restore from snapshot
        restored_state = SimulationState.from_snapshot(
            snapshot,
            config=config,
            component_factory=MockFactory(),
            environment=original_state.environment,
            event_bus=MagicMock(),
            db_logger=MagicMock(),
        )

        # --- Assert ---
        assert restored_state.simulation_id == original_state.simulation_id
        assert restored_state.current_tick == original_state.current_tick

        # Compare dictionaries for a more robust check of content
        original_agent_comps = {type(c).__name__: c.to_dict() for c in original_state.entities["agent_x"].values()}
        restored_agent_comps = {type(c).__name__: c.to_dict() for c in restored_state.entities["agent_x"].values()}
        assert original_agent_comps == restored_agent_comps
