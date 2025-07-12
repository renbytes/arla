from unittest.mock import MagicMock

import pytest
from agent_core.core.ecs.component import IdentityComponent

# The class to test
from simulations.soul_sim.component_factory import SoulSimComponentFactory

# Components to verify
from simulations.soul_sim.components import HealthComponent, PositionComponent


@pytest.fixture
def factory_setup():
    mock_env = MagicMock()
    mock_config = {"agent": {"cognitive": {"embeddings": {"identity_dim": 4}}}}
    factory = SoulSimComponentFactory(environment=mock_env, config=mock_config)
    return factory, mock_env


def test_factory_creates_generic_component(factory_setup):
    """Tests that the factory can create a simple component from its data."""
    factory, _ = factory_setup
    # The HealthComponent constructor only takes `initial_health`.
    # The `current_health` is initialized from that value internally.
    # This test now correctly reflects how a component would be restored.
    comp = factory.create_component("simulations.soul_sim.components.HealthComponent", {"initial_health": 80.0})
    assert isinstance(comp, HealthComponent)
    # The current_health should be equal to initial_health upon creation.
    assert comp.current_health == 80.0


def test_factory_creates_position_component_with_env(factory_setup):
    """Tests the special case for PositionComponent, ensuring it gets the environment."""
    factory, mock_env = factory_setup
    comp = factory.create_component("simulations.soul_sim.components.PositionComponent", {"position": (10, 20)})
    assert isinstance(comp, PositionComponent)
    assert comp.position == (10, 20)
    assert comp.environment is mock_env  # Crucial check


def test_factory_creates_identity_component_with_mdi(factory_setup):
    """Tests the special case for IdentityComponent, ensuring it gets a MultiDomainIdentity object."""
    factory, _ = factory_setup
    comp = factory.create_component(
        "agent_core.core.ecs.component.IdentityComponent",
        {},  # Data can be empty for this test
    )
    assert isinstance(comp, IdentityComponent)
    assert comp.multi_domain_identity is not None  # Crucial check
