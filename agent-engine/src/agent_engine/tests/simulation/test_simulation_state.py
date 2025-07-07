# tests/simulation/test_simulation_state.py

import pytest
import numpy as np

# Subject under test
from agent_engine.simulation.simulation_state import SimulationState
from agent_core.core.ecs.component import (
    Component,
    AffectComponent,
    EmotionComponent,
    GoalComponent,
    IdentityComponent,
)
from agent_engine.cognition.identity.domain_identity import IdentityDomain, MultiDomainIdentity

# --- Fixtures ---


@pytest.fixture
def config():
    """Provides a sample config for the simulation state."""
    return {"agent": {"cognitive": {"embeddings": {"main_embedding_dim": 4}}}}


@pytest.fixture
def sim_state(config):
    """Provides a fresh SimulationState instance for each test."""
    return SimulationState(config=config, device="cpu")


# --- Test Cases for Basic ECS Management ---


def test_add_and_get_entity(sim_state):
    """
    Tests adding an entity and verifying its existence.
    """
    sim_state.add_entity("agent1")
    assert "agent1" in sim_state.entities
    with pytest.raises(ValueError):
        sim_state.add_entity("agent1")  # Cannot add duplicate


def test_add_and_get_component(sim_state):
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


def test_remove_entity(sim_state):
    """
    Tests that an entity and all its components are removed correctly.
    """
    sim_state.add_entity("agent1")
    assert "agent1" in sim_state.entities
    sim_state.remove_entity("agent1")
    assert "agent1" not in sim_state.entities


def test_get_entities_with_components(sim_state):
    """
    Tests retrieving all entities that possess a specific set of components.
    """
    sim_state.add_entity("agent1")
    sim_state.add_component("agent1", EmotionComponent())
    sim_state.add_component("agent1", GoalComponent(embedding_dim=4))

    sim_state.add_entity("agent2")
    sim_state.add_component("agent2", EmotionComponent())

    # Act
    entities = sim_state.get_entities_with_components([EmotionComponent, GoalComponent])

    # Assert
    assert "agent1" in entities
    assert "agent2" not in entities
    assert len(entities) == 1


# --- Test Cases for get_internal_state_features_for_entity ---


@pytest.fixture
def full_components(config):
    """Provides a full set of mock cognitive components."""
    affect_comp = AffectComponent(affective_buffer_maxlen=10)
    affect_comp.prediction_delta_magnitude = 0.5
    affect_comp.predictive_delta_smooth = 0.6

    emotion_comp = EmotionComponent(valence=0.7, arousal=0.8)

    goal_comp = GoalComponent(embedding_dim=4)
    goal_comp.current_symbolic_goal = "test_goal"
    goal_comp.symbolic_goals_data["test_goal"] = {"embedding": np.ones(4)}

    id_comp = IdentityComponent(multi_domain_identity=MultiDomainIdentity(embedding_dim=4))

    return affect_comp, emotion_comp, goal_comp, id_comp


def test_get_internal_state_features_all_present(sim_state, full_components):
    """
    Tests feature generation when all cognitive components are present.
    """
    # Arrange
    affect_comp, emotion_comp, goal_comp, id_comp = full_components
    num_domains = len(IdentityDomain)
    expected_len = 4 + 4 + (4 * num_domains) + 3  # affect/emo + goal + identity + flags

    # Act
    features = sim_state.get_internal_state_features_for_entity(id_comp, affect_comp, goal_comp, emotion_comp)

    # Assert
    assert features.shape == (expected_len,)
    # Check flags are all 1.0, indicating all components were present
    assert np.all(features[-3:] == 1.0)
    # Check that the first part of the vector contains emotion/affect data
    assert features[0] == pytest.approx(0.7)  # valence
    assert features[3] == pytest.approx(0.6)  # predictive_delta_smooth


def test_get_internal_state_features_some_missing(sim_state, full_components):
    """
    Tests feature generation when some cognitive components are missing.
    """
    # Arrange
    _, _, goal_comp, id_comp = full_components
    num_domains = len(IdentityDomain)
    expected_len = 4 + 4 + (4 * num_domains) + 3

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
