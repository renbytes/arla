# tests/utils/test_config_utils.py

import pytest

# Subject under test
from agent_engine.utils.config_utils import get_config_value


# --- Fixtures ---


@pytest.fixture
def nested_config():
    """Provides a sample nested dictionary for testing."""
    return {
        "agent": {
            "cognitive": {
                "embeddings": {
                    "main_embedding_dim": 1536,
                },
                "memory": {
                    "reflection_interval": 50,
                },
            },
            "emotional_noise_std": 0.02,
        },
        "simulation": {
            "steps": 100,
            "random_seed": None,  # Test retrieving a None value
        },
    }


# --- Test Cases ---


def test_get_config_value_deeply_nested(nested_config):
    """
    Tests retrieving a value that is several levels deep in the dictionary.
    """
    path = "agent.cognitive.embeddings.main_embedding_dim"
    value = get_config_value(nested_config, path)
    assert value == 1536


def test_get_config_value_top_level(nested_config):
    """
    Tests retrieving a dictionary that is at the top level.
    """
    path = "simulation"
    value = get_config_value(nested_config, path)
    assert value == {"steps": 100, "random_seed": None}


def test_get_config_value_invalid_path_returns_default(nested_config):
    """
    Tests that a default value is returned when the path does not exist.
    """
    path = "agent.cognitive.nonexistent_key"
    # Test with a specific default value
    value = get_config_value(nested_config, path, default="default_val")
    assert value == "default_val"
    # Test with the default of None
    value_none = get_config_value(nested_config, path)
    assert value_none is None


def test_get_config_value_path_through_non_dict_returns_default(nested_config):
    """
    Tests that a default value is returned if an intermediate key in the path
    is not a dictionary.
    """
    path = "agent.emotional_noise_std.some_deeper_key"
    value = get_config_value(nested_config, path, default="default_val")
    assert value == "default_val"


def test_get_config_value_retrieves_explicit_none(nested_config):
    """
    Tests that an explicit value of None in the config is correctly returned,
    instead of the default value.
    """
    path = "simulation.random_seed"
    value = get_config_value(nested_config, path, default=12345)
    assert value is None


def test_get_config_value_empty_path(nested_config):
    """
    Tests that providing an empty path returns the default value.
    """
    # An empty path is not valid, so it should return the default.
    value = get_config_value(nested_config, "", default="default_val")
    assert value == "default_val"
