# tests/cognition/identity/test_domain_identity.py

from unittest.mock import MagicMock
import numpy as np
import pytest

# Subject under test
from agent_engine.cognition.identity.domain_identity import (
    MultiDomainIdentity,
    IdentityDomain,
    SocialValidationCollector,
)


# --- Fixtures ---


@pytest.fixture
def identity():
    """Provides a fresh MultiDomainIdentity instance for each test."""
    return MultiDomainIdentity(embedding_dim=4)


@pytest.fixture
def social_validator():
    """Provides a fresh SocialValidationCollector instance."""
    return SocialValidationCollector()


# --- Test Cases for MultiDomainIdentity ---


def test_initialization(identity):
    """
    Tests that all identity domains are initialized with default, non-zero embeddings.
    """
    assert len(identity.domains) == len(IdentityDomain)
    for domain in IdentityDomain:
        assert domain in identity.domains
        domain_id = identity.domains[domain]
        assert domain_id.embedding.shape == (4,)
        assert not np.allclose(domain_id.embedding, 0)
        assert domain_id.confidence == 0.3
        assert domain_id.stability == 0.25


def test_get_global_identity_embedding(identity):
    """
    Tests that the global embedding is a correctly weighted average of domain embeddings.
    """
    # Arrange: Set known embeddings and confidences
    identity.domains[IdentityDomain.SOCIAL].embedding = np.array([1, 0, 0, 0])
    identity.domains[IdentityDomain.SOCIAL].confidence = 1.0
    identity.domains[IdentityDomain.MORAL].embedding = np.array([0, 1, 0, 0])
    identity.domains[IdentityDomain.MORAL].confidence = 0.5
    # Other domains have low confidence and will contribute less

    # Act
    global_embedding = identity.get_global_identity_embedding()

    # Assert
    # The global embedding should be skewed towards the SOCIAL domain due to higher confidence.
    assert global_embedding.shape == (4,)
    assert np.argmax(global_embedding) == 0
    assert global_embedding[0] > global_embedding[1]


def test_get_identity_coherence(identity):
    """
    Tests the calculation of identity coherence based on domain similarity.
    """
    # Arrange: Make two domains identical and one opposite
    identity.domains[IdentityDomain.SOCIAL].embedding = np.array([1, 0, 0, 0])
    identity.domains[IdentityDomain.COMPETENCE].embedding = np.array([1, 0, 0, 0])
    identity.domains[IdentityDomain.MORAL].embedding = np.array([-1, 0, 0, 0])

    # Act
    coherence = identity.get_identity_coherence()

    # Assert
    # Coherence should be between 0 (completely incoherent) and 1 (identical).
    # The two identical domains will have high similarity, the opposite ones low similarity.
    assert 0 < coherence < 1


def test_get_identity_stability(identity):
    """
    Tests the calculation of overall identity stability.
    """
    # Arrange
    identity.domains[IdentityDomain.SOCIAL].stability = 0.8
    identity.domains[IdentityDomain.COMPETENCE].stability = 0.6
    # Other domains are at the default 0.25

    # Act
    stability = identity.get_identity_stability()

    # Assert
    expected_stability = np.mean([0.8, 0.6, 0.25, 0.25, 0.25])
    assert stability == pytest.approx(expected_stability)


def test_update_domain_identity_successful_update(identity):
    """
    Tests a scenario where an identity update should occur due to high support.
    """
    # Arrange
    domain_to_update = IdentityDomain.COMPETENCE
    new_traits = np.array([0, 1, 0, 1], dtype=np.float32)
    social_feedback = {"positive_social_responses": 0.9, "social_approval_rating": 0.8}
    original_embedding = identity.get_domain_embedding(domain_to_update).copy()

    # Act
    updated, _, _ = identity.update_domain_identity(
        domain=domain_to_update,
        new_traits=new_traits,
        social_feedback=social_feedback,
        current_tick=10,
    )

    # Assert
    assert updated is True
    new_embedding = identity.get_domain_embedding(domain_to_update)
    # The embedding should have changed from the original
    assert not np.array_equal(original_embedding, new_embedding)
    # The new embedding should be closer to the new_traits vector
    original_dist = np.linalg.norm(original_embedding - new_traits)
    new_dist = np.linalg.norm(new_embedding - new_traits)
    assert new_dist < original_dist


def test_update_domain_identity_resisted_update(identity):
    """
    Tests a scenario where an identity update is resisted due to low support and high stability.
    """
    # Arrange
    domain_to_update = IdentityDomain.MORAL
    identity.domains[domain_to_update].stability = 0.9  # Very stable
    identity.domains[domain_to_update].confidence = 0.95  # Very confident

    # New traits are very different from existing identity
    new_traits = -identity.get_domain_embedding(domain_to_update)
    social_feedback = {"negative_social_responses": 0.8}  # Negative feedback
    original_embedding = identity.get_domain_embedding(domain_to_update).copy()

    # Act
    updated, _, _ = identity.update_domain_identity(
        domain=domain_to_update,
        new_traits=new_traits,
        social_feedback=social_feedback,
        current_tick=10,
    )

    # Assert
    assert updated is False
    new_embedding = identity.get_domain_embedding(domain_to_update)
    # The embedding should NOT have changed
    np.testing.assert_array_equal(original_embedding, new_embedding)


# --- Test Cases for SocialValidationCollector ---


def test_collect_social_feedback_with_data(social_validator):
    """
    Tests that social feedback is correctly aggregated from social schemas.
    """
    # Arrange
    mock_schema_positive = MagicMock()
    mock_schema_positive.impression_valence = 0.8
    mock_schema_positive.interaction_count = 5

    mock_schema_negative = MagicMock()
    mock_schema_negative.impression_valence = -0.6
    mock_schema_negative.interaction_count = 2

    social_schemas = {
        "agent_pos": mock_schema_positive,
        "agent_neg": mock_schema_negative,
    }

    # Act
    feedback = social_validator.collect_social_feedback("agent1", {}, social_schemas, 100)

    # Assert
    assert feedback["positive_social_responses"] == pytest.approx(5 / 7)
    assert feedback["negative_social_responses"] == pytest.approx(2 / 7)
    assert feedback["social_approval_rating"] > 0.5  # (0.8 - 0.6)/2 -> positive avg valence
    assert feedback["peer_recognition"] > 0


def test_collect_social_feedback_no_data(social_validator):
    """
    Tests that the collector returns a neutral default when there are no social schemas.
    """
    # Act
    feedback = social_validator.collect_social_feedback("agent1", {}, {}, 100)

    # Assert
    assert feedback["positive_social_responses"] == 0.0
    assert feedback["negative_social_responses"] == 0.0
    assert feedback["social_approval_rating"] == 0.0
    assert feedback["peer_recognition"] == 0.0
