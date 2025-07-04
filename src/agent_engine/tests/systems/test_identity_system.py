# tests/systems/test_identity_system.py
"""Unit-tests for :pymod:`agent_engine.systems.identity_system.IdentitySystem`.

The key change compared to the previous version is that we patch the **exact**
symbol used inside the production module –
``get_embedding_with_cache`` – instead of the lower-level
``get_embedding_from_llm``.  This prevents any accidental network calls during
unit-testing and eliminates the 401 error spam.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import os
from typing import Dict, List

import numpy as np
import pytest

# Subject under test ---------------------------------------------------------
from agent_engine.systems.identity_system import IdentitySystem
from agent_engine.cognition.identity.domain_identity import (
    IdentityDomain,
    MultiDomainIdentity,
)
from agent_core.core.ecs.component import (
    IdentityComponent,
    SocialMemoryComponent,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_simulation_state():
    """Return a fake :class:`SimulationState` with minimal data."""
    state = MagicMock()

    # Real *instances* so ``isinstance`` checks in production code pass.
    mock_mdi = MagicMock(spec=MultiDomainIdentity)
    id_comp_instance = IdentityComponent(multi_domain_identity=mock_mdi)

    mock_social_comp = MagicMock(spec=SocialMemoryComponent)
    mock_social_comp.schemas = {"agent_b": MagicMock()}

    state.entities = {
        "agent1": {
            IdentityComponent: id_comp_instance,
            SocialMemoryComponent: mock_social_comp,
        }
    }

    def get_component_side_effect(entity_id: str, comp_type):
        return state.entities.get(entity_id, {}).get(comp_type)

    state.get_component.side_effect = get_component_side_effect
    return state


@pytest.fixture
def mock_cognitive_scaffold():
    """Stub scaffold that returns a deterministic structured reply."""
    scaffold = MagicMock()
    scaffold.query.return_value = """
        SOCIAL:
        - cooperative: 0.8
        - friendly: 0.7
        COMPETENCE:
        - skilled_builder: 0.9
        MORAL:
        - honest: 0.6
        """
    return scaffold


@pytest.fixture
def mock_event_bus():
    """Bare-bones EventBus replacement."""
    return MagicMock()


@pytest.fixture
def mock_openai_key(mocker):
    """Set a fake API key so OpenAI client creation never crashes."""
    mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})


# ---------------------------------------------------------------------------
# IdentitySystem fixture – patch the correct symbol
# ---------------------------------------------------------------------------


@pytest.fixture
@patch("agent_engine.systems.identity_system.SocialValidationCollector")
@patch("agent_engine.systems.identity_system.get_embedding_with_cache")
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Patch the alias actually *imported* by IdentitySystem.
# -------------------------------------------------------------------
# Note: order of the decorators = order of arguments (rightmost patch
# becomes first parameter).

def identity_system(
    mock_get_embedding_with_cache,  # type: MagicMock
    mock_validator_class,  # type: MagicMock
    mock_simulation_state,
    mock_cognitive_scaffold,
    mock_event_bus,
    mock_openai_key,
):
    """Return an initialised :class:`IdentitySystem` with networking stubbed."""

    # Every embedding request yields the same 4-d vector.
    mock_get_embedding_with_cache.return_value = np.ones(4, dtype=np.float32)

    # Inject the fake bus.
    mock_simulation_state.event_bus = mock_event_bus

    return IdentitySystem(
        simulation_state=mock_simulation_state,
        config={"llm": {}, "agent": {"cognitive": {"embeddings": {"main_embedding_dim": 4}}}},
        cognitive_scaffold=mock_cognitive_scaffold,
    )


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def test_on_reflection_completed_successful_update(identity_system, mock_simulation_state, mock_cognitive_scaffold):
    """`on_reflection_completed` should update three identity domains."""

    event_data: Dict[str, object] = {
        "entity_id": "agent1",
        "llm_final_account": "I worked with others to build a shelter.",
        "tick": 100,
    }

    id_comp_instance: IdentityComponent = mock_simulation_state.entities["agent1"][IdentityComponent]

    identity_system.on_reflection_completed(event_data)

    # LLM was queried once.
    mock_cognitive_scaffold.query.assert_called_once()

    # Assert that update_domain_identity was called for SOCIAL, COMPETENCE, MORAL.
    calls = id_comp_instance.multi_domain_identity.update_domain_identity.call_args_list
    updated_domains: List[IdentityDomain] = [c.kwargs["domain"] for c in calls]

    assert len(updated_domains) == 3
    assert {
        IdentityDomain.SOCIAL,
        IdentityDomain.COMPETENCE,
        IdentityDomain.MORAL,
    } == set(updated_domains)

    # Salient traits cache should include items from the LLM response.
    cache = id_comp_instance.salient_traits_cache
    assert cache["cooperative"] == 0.8
    assert "skilled_builder" in cache


def test_on_reflection_completed_missing_component(identity_system, mock_simulation_state):
    """Handler should do nothing when the entity lacks an IdentityComponent."""

    mock_simulation_state.entities["agent_no_id_comp"] = {}

    event_data = {
        "entity_id": "agent_no_id_comp",
        "llm_final_account": "…",
        "tick": 100,
    }

    # Reset mock so we can assert later.
    id_comp = mock_simulation_state.entities["agent1"][IdentityComponent]
    id_comp.multi_domain_identity.update_domain_identity.reset_mock()

    identity_system.on_reflection_completed(event_data)

    id_comp.multi_domain_identity.update_domain_identity.assert_not_called()


@pytest.mark.parametrize(
    "llm_response, expected_domains",
    [
        ("", {}),
        ("SOCIAL:\n- friendly: 0.8", {IdentityDomain.SOCIAL: {"friendly": 0.8}}),
        ("INVALID_DOMAIN:\n- trait: 1.0", {}),
        ("COMPETENCE:\n- invalid-line", {IdentityDomain.COMPETENCE: {}}),
        ("MORAL:\n- honest: 99.9", {IdentityDomain.MORAL: {"honest": 1.0}}),
    ],
)
def test_parse_structured_llm_traits(identity_system, llm_response: str, expected_domains: Dict):
    """Parser should always return a dict keyed by every IdentityDomain."""

    parsed = identity_system._parse_structured_llm_traits(llm_response)

    # Length must match the enum size.
    assert len(parsed) == len(IdentityDomain)

    for domain, traits in expected_domains.items():
        assert parsed[domain] == traits
