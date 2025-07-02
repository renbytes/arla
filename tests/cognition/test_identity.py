# --- tests/cognition/test_identity.py ---

import numpy as np
import pytest

# FIX: Correct the import path
from agent_core.cognition.identity.domain_identity import MultiDomainIdentity, IdentityDomain


def test_identity_initialization():
    """Tests that the MultiDomainIdentity initializes with all domains."""
    identity = MultiDomainIdentity(embedding_dim=16)
    assert len(identity.domains) == len(IdentityDomain)
    for domain in IdentityDomain:
        assert domain in identity.domains
        assert identity.domains[domain].embedding.shape == (16,)
        assert identity.domains[domain].confidence == pytest.approx(0.3)
        assert identity.domains[domain].stability == pytest.approx(0.25)


def test_get_global_embedding_weighted_average():
    """Tests the weighted average calculation for the global identity."""
    identity = MultiDomainIdentity(embedding_dim=2)

    # Set known values for testing
    identity.domains[IdentityDomain.SOCIAL].embedding = np.array([1.0, 0.0])
    identity.domains[IdentityDomain.SOCIAL].confidence = 1.0

    identity.domains[IdentityDomain.COMPETENCE].embedding = np.array([0.0, 1.0])
    identity.domains[IdentityDomain.COMPETENCE].confidence = 0.5

    # Make other confidences zero so they don't affect the average
    for domain in IdentityDomain:
        if domain not in [IdentityDomain.SOCIAL, IdentityDomain.COMPETENCE]:
            identity.domains[domain].confidence = 0.0

    global_embedding = identity.get_global_identity_embedding()

    # Expected: ( [1,0]*1.0 + [0,1]*0.5 ) / (1.0 + 0.5) = [1, 0.5] / 1.5 = [0.666, 0.333]
    expected = np.array([1.0, 0.5]) / 1.5
    np.testing.assert_allclose(global_embedding, expected, rtol=1e-5)


def test_identity_coherence():
    """Tests the identity coherence calculation."""
    identity = MultiDomainIdentity(embedding_dim=2)

    # Make all embeddings identical for perfect coherence
    for domain in IdentityDomain:
        identity.domains[domain].embedding = np.array([1.0, 0.0])
        identity.domains[domain].confidence = 1.0
    assert identity.get_identity_coherence() == pytest.approx(1.0)

    # Make one embedding orthogonal for lower coherence
    identity.domains[IdentityDomain.SOCIAL].embedding = np.array([0.0, 1.0])
    assert identity.get_identity_coherence() < 1.0


def test_update_domain_identity_simple_case():
    """Tests a simple, successful update to a domain identity."""
    identity = MultiDomainIdentity(embedding_dim=2)
    original_embedding = identity.domains[IdentityDomain.SOCIAL].embedding.copy()

    update_occurred, _, _ = identity.update_domain_identity(
        domain=IdentityDomain.SOCIAL,
        new_traits=np.array([0.5, 0.5]),
        social_feedback={"positive_social_responses": 0.8, "social_approval_rating": 0.9},
        current_tick=10,
    )

    assert update_occurred is True
    # Check that the embedding has changed from its original state
    assert not np.array_equal(original_embedding, identity.domains[IdentityDomain.SOCIAL].embedding)
    assert identity.domains[IdentityDomain.SOCIAL].last_updated == 10
