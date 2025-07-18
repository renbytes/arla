# FILE: tests/providers/test_simulation_providers.py

from unittest.mock import MagicMock

import numpy as np
import pytest

# Mock the components that the encoder depends on
from simulations.emergence_sim.components import (
    ConceptualSpaceComponent,
    DebtLedgerComponent,
    InventoryComponent,
    PositionComponent,
    SocialCreditComponent,
)

# The class we are testing
from simulations.emergence_sim.providers.simulation_providers import (
    EmergenceStateEncoder,
)

# --- Test Fixtures ---


@pytest.fixture
def state_encoder():
    """Returns an instance of the EmergenceStateEncoder."""
    return EmergenceStateEncoder()


@pytest.fixture
def mock_config():
    """
    Provides a mock configuration object with the necessary dimensions
    for state vector validation.
    """
    config = MagicMock()
    config.learning.q_learning.state_feature_dim = 16
    config.learning.q_learning.internal_state_dim = 8
    return config


# --- Unit Tests ---


def test_encode_state_returns_correct_shape(state_encoder, mock_config):
    """
    Tests that encode_state returns a numpy array with the shape
    defined by config.learning.q_learning.state_feature_dim.
    """
    # 1. ARRANGE: Set up a mock simulation state with necessary components
    mock_sim_state = MagicMock()
    mock_entity_id = "agent_1"

    # Mock the nested structure the encoder will access
    mock_sim_state.entities = {
        mock_entity_id: {
            PositionComponent: MagicMock(position=(5, 5)),
            InventoryComponent: MagicMock(current_resources=20),
        }
    }
    mock_sim_state.environment.width = 10
    mock_sim_state.environment.height = 10
    mock_sim_state.config = mock_config  # Link the config

    # 2. ACT: Call the method under test
    feature_vector = state_encoder.encode_state(
        simulation_state=mock_sim_state,
        entity_id=mock_entity_id,
        config=mock_config,
    )

    # 3. ASSERT: Verify the output shape and type
    assert isinstance(feature_vector, np.ndarray)
    assert feature_vector.shape == (mock_config.learning.q_learning.state_feature_dim,)
    assert feature_vector.dtype == np.float32


def test_encode_internal_state_returns_correct_shape(state_encoder, mock_config):
    """
    Tests that encode_internal_state returns a numpy array with the shape
    defined by config.learning.q_learning.internal_state_dim.
    """
    # 1. ARRANGE: Create a dictionary of mock components for an agent
    mock_components = {
        SocialCreditComponent: MagicMock(score=0.75),
        DebtLedgerComponent: MagicMock(obligations=[1, 2, 3]),  # Just need the length
        ConceptualSpaceComponent: MagicMock(concepts={"symbol1": {}, "symbol2": {}}),
    }

    # 2. ACT: Call the method under test
    feature_vector = state_encoder.encode_internal_state(components=mock_components, config=mock_config)

    # 3. ASSERT: Verify the output shape and type
    assert isinstance(feature_vector, np.ndarray)
    assert feature_vector.shape == (mock_config.learning.q_learning.internal_state_dim,)
    assert feature_vector.dtype == np.float32


def test_encode_internal_state_handles_missing_components(state_encoder, mock_config):
    """
    Tests that encode_internal_state correctly handles cases where an agent
    is missing some or all of the relevant cognitive components. It should
    return a correctly shaped zero-vector, not crash.
    """
    # 1. ARRANGE: Create an empty dictionary of components
    mock_components = {}

    # 2. ACT: Call the method with no components
    feature_vector = state_encoder.encode_internal_state(components=mock_components, config=mock_config)

    # 3. ASSERT: Verify the output is a correctly shaped zero-vector
    assert isinstance(feature_vector, np.ndarray)
    assert feature_vector.shape == (mock_config.learning.q_learning.internal_state_dim,)
    assert np.all(feature_vector == 0)  # Check that all elements are zero


def test_encode_state_handles_missing_components(state_encoder, mock_config):
    """
    Tests that encode_state handles agents missing physical components
    and still returns a correctly shaped zero-vector.
    """
    # 1. ARRANGE: Set up a mock state where the agent has no components
    mock_sim_state = MagicMock()
    mock_entity_id = "agent_1"
    mock_sim_state.entities = {mock_entity_id: {}}  # Agent exists but has no components
    mock_sim_state.config = mock_config

    # 2. ACT: Call the method
    feature_vector = state_encoder.encode_state(
        simulation_state=mock_sim_state,
        entity_id=mock_entity_id,
        config=mock_config,
    )

    # 3. ASSERT: Verify the output is a correctly shaped zero-vector
    assert isinstance(feature_vector, np.ndarray)
    assert feature_vector.shape == (mock_config.learning.q_learning.state_feature_dim,)
    assert np.all(feature_vector == 0)
