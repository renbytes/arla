# tests/cognition/reflection/test_episode.py

import pytest

# Subject under test
from agent_engine.cognition.reflection.episode import Episode

# Fixtures


@pytest.fixture
def sample_episode():
    """Provides a sample Episode instance for testing."""
    events = [
        {"tick": 10, "action_type": "MOVE", "reward": 0.1},
        {"tick": 12, "action_type": "EXTRACT", "reward": 5.0},
        {"tick": 15, "action_type": "COMMUNICATE", "reward": 0.5},
        {
            "tick": 18,
            "action_type": "REST",
            "reward": 0.0,
        },  # This event should not be in the preview
    ]

    episode = Episode(
        start_tick=10,
        end_tick=20,
        theme="A day of work",
        emotional_valence_curve=[0.1, 0.3, 0.2],
        events=events,
        goal_at_start="gather_resources",
        goal_at_end="store_resources",
    )
    return episode


# Test Cases


def test_episode_initialization(sample_episode):
    """
    Tests that the Episode dataclass initializes correctly.
    """
    assert sample_episode.start_tick == 10
    assert sample_episode.end_tick == 20
    assert sample_episode.theme == "A day of work"
    assert len(sample_episode.events) == 4
    assert sample_episode.goal_at_start == "gather_resources"


def test_episode_to_dict_serialization(sample_episode):
    """
    Tests that the to_dict method correctly serializes the episode's data,
    including the logic for event previews.
    """
    # Act
    episode_dict = sample_episode.to_dict()

    # Assert
    assert episode_dict["start_tick"] == 10
    assert episode_dict["theme"] == "A day of work"
    assert episode_dict["goal_at_end"] == "store_resources"
    assert episode_dict["event_count"] == 4

    # Check the event previews
    assert len(episode_dict["event_previews"]) == 3
    assert "Tick 10: MOVE" in episode_dict["event_previews"][0]
    assert "Tick 12: EXTRACT" in episode_dict["event_previews"][1]
    assert "Tick 15: COMMUNICATE" in episode_dict["event_previews"][2]


def test_episode_to_dict_with_few_events():
    """
    Tests that to_dict handles cases with fewer than 3 events correctly.
    """
    # Arrange
    events = [{"tick": 5, "action_type": "IDLE"}]
    episode = Episode(start_tick=5, end_tick=6, theme="A quiet moment", events=events)

    # Act
    episode_dict = episode.to_dict()

    # Assert
    assert episode_dict["event_count"] == 1
    assert len(episode_dict["event_previews"]) == 1
    assert "Tick 5: IDLE" in episode_dict["event_previews"][0]
