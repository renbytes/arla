# tests/systems/test_identity_system.py
"""Unit-tests for :pymod:`agent_engine.systems.identity_system.IdentitySystem`.

The key change compared to the previous version is that we patch the **exact**
symbol used inside the production module –
``get_embedding_with_cache`` – instead of the lower-level
``get_embedding_from_llm``.  This prevents any accidental network calls during
unit-testing and eliminates the 401 error spam.
"""

from unittest.mock import MagicMock, patch
from typing import Any, Dict, Iterator

import numpy as np
import pytest

# Subject under test
from agent_engine.systems.identity_system import IdentitySystem
from agent_engine.cognition.identity.domain_identity import (
    IdentityDomain,
    MultiDomainIdentity,
)
from agent_core.core.ecs.component import (
    IdentityComponent,
)

# --- Fixtures ---


@pytest.fixture
def mock_simulation_state() -> MagicMock:
    """Return a mock SimulationState with a realistic IdentityComponent."""
    state = MagicMock()

    # Use a real MultiDomainIdentity instance but mock its method
    # to make the test more robust and realistic.
    mock_mdi = MultiDomainIdentity(embedding_dim=4)
    mock_mdi.update_domain_identity = MagicMock()

    id_comp_instance = IdentityComponent(multi_domain_identity=mock_mdi)

    state.entities = {"agent1": {IdentityComponent: id_comp_instance}}

    def get_component_side_effect(entity_id: str, comp_type):
        return state.entities.get(entity_id, {}).get(comp_type)

    state.get_component.side_effect = get_component_side_effect
    return state


@pytest.fixture
def mock_cognitive_scaffold() -> MagicMock:
    """Stub scaffold that returns a deterministic structured reply."""
    scaffold = MagicMock()
    scaffold.query.return_value = """
        SOCIAL:
        - cooperative: 0.8
        COMPETENCE:
        - skilled_builder: 0.9
        MORAL:
        - honest: 0.6
        """
    return scaffold


@pytest.fixture
def mock_event_bus() -> MagicMock:
    """Bare-bones EventBus replacement."""
    return MagicMock()


@pytest.fixture
def identity_system(
    mock_simulation_state: MagicMock,
    mock_cognitive_scaffold: MagicMock,
    mock_event_bus: MagicMock,
) -> Iterator[IdentitySystem]:
    """
    Return an initialized IdentitySystem with dependencies stubbed.
    No more patching is needed as the system is fully decoupled.
    """
    mock_simulation_state.event_bus = mock_event_bus

    # We still need a patch for the external network call
    with patch("agent_engine.systems.identity_system.get_embedding_with_cache") as mock_get_embedding:
        mock_get_embedding.return_value = np.ones(4, dtype=np.float32)
        system = IdentitySystem(
            simulation_state=mock_simulation_state,
            config={"llm": {}, "agent": {"cognitive": {"embeddings": {"main_embedding_dim": 4}}}},
            cognitive_scaffold=mock_cognitive_scaffold,
        )
        yield system


# --- Test Cases ---


def test_on_reflection_completed_updates_identity(
    identity_system: IdentitySystem, mock_simulation_state: MagicMock, mock_cognitive_scaffold: MagicMock
):
    """
    GIVEN a reflection event with a context dictionary
    WHEN on_reflection_completed is called
    THEN the system should query the LLM and update the agent's identity domains.
    """
    # ARRANGE: The mock context must now include 'llm_final_account'
    # to accurately simulate the event produced by ReflectionSystem.
    event_data: Dict[str, Any] = {
        "entity_id": "agent1",
        "context": {
            "narrative": "I worked with others to build a shelter.",
            "llm_final_account": "Reflecting on my actions, I see I am becoming a cooperative and skilled builder.",
            "social_feedback": {"positive_social_responses": 0.5},
        },
        "tick": 100,
    }

    id_comp_instance: IdentityComponent = mock_simulation_state.entities["agent1"][IdentityComponent]

    # ACT
    identity_system.on_reflection_completed(event_data)

    # ASSERT
    # 1. The LLM was queried to infer traits from the synthesized reflection.
    mock_cognitive_scaffold.query.assert_called_once()
    assert "cooperative and skilled builder" in mock_cognitive_scaffold.query.call_args[1]["prompt"]

    # 2. update_domain_identity was called for the domains found in the LLM response.
    calls = id_comp_instance.multi_domain_identity.update_domain_identity.call_args_list
    updated_domains = {c.kwargs["domain"] for c in calls}

    assert len(updated_domains) == 3
    assert {IdentityDomain.SOCIAL, IdentityDomain.COMPETENCE, IdentityDomain.MORAL} == updated_domains

    # 3. The generic context from the event was passed through to the identity model.
    call_context = calls[0].kwargs["context"]
    assert call_context["narrative"] == "I worked with others to build a shelter."
    assert call_context["social_feedback"]["positive_social_responses"] == 0.5

    # 4. The salient traits cache was updated.
    assert "cooperative" in id_comp_instance.salient_traits_cache


def test_on_reflection_completed_missing_component(identity_system: IdentitySystem, mock_simulation_state: MagicMock):
    """
    GIVEN an event for an entity without an IdentityComponent
    WHEN on_reflection_completed is called
    THEN the system should not perform any updates and exit gracefully.
    """
    # ARRANGE
    mock_simulation_state.entities["agent_no_id_comp"] = {}
    event_data = {
        "entity_id": "agent_no_id_comp",
        "context": {"narrative": "...",
        "llm_final_account": "some reflection"},
        "tick": 100,
    }
    id_comp = mock_simulation_state.entities["agent1"][IdentityComponent]
    id_comp.multi_domain_identity.update_domain_identity.reset_mock()

    # ACT
    identity_system.on_reflection_completed(event_data)

    # ASSERT
    id_comp.multi_domain_identity.update_domain_identity.assert_not_called()


@pytest.mark.parametrize(
    "llm_response, expected_domains",
    [
        ("", {}),  # Empty response
        ("SOCIAL:\n- friendly: 0.8", {IdentityDomain.SOCIAL: {"friendly": 0.8}}),  # Valid
        ("INVALID_DOMAIN:\n- trait: 1.0", {}),  # Skips invalid domains
        ("COMPETENCE:\n- invalid-line", {IdentityDomain.COMPETENCE: {}}),  # Handles malformed lines
        ("MORAL:\n- honest: 99.9", {IdentityDomain.MORAL: {"honest": 1.0}}),  # Clips scores to 1.0
    ],
)
def test_parse_structured_llm_traits(identity_system: IdentitySystem, llm_response: str, expected_domains: Dict):
    """
    Tests that the internal LLM response parser correctly handles various formats.
    """
    parsed = identity_system._parse_structured_llm_traits(llm_response)

    # Assert that the output dictionary always contains all possible domains
    assert len(parsed) == len(IdentityDomain)

    # Assert that the parsed traits match the expected output for this test case
    for domain, traits in expected_domains.items():
        assert parsed[domain] == traits
