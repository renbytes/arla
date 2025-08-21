# tests/agent_persist/test_store.py

import json
from pathlib import Path

import pytest
from agent_persist.models import SimulationSnapshot

# Subject under test
from agent_persist.store import FileStateStore

# Fixtures


@pytest.fixture
def snapshot_instance():
    """Provides a valid instance of the SimulationSnapshot model."""
    return SimulationSnapshot(
        simulation_id="sim_save_load_test",
        current_tick=50,
        agents=[
            {
                "agent_id": "agent1",
                "components": [{"component_type": "TestComp", "data": {"value": 1}}],
            }
        ],
    )


@pytest.fixture
def store(tmp_path: Path) -> FileStateStore:
    """Provides a FileStateStore instance using a temporary directory."""
    test_file = tmp_path / "snapshots" / "test_snapshot.json"
    return FileStateStore(file_path=test_file)


# Test Cases


def test_save_creates_file_and_directories(
    store: FileStateStore, snapshot_instance: SimulationSnapshot
):
    """
    Tests that the save method correctly creates the file and any necessary
    parent directories.
    """
    # Act
    store.save(snapshot_instance)

    # Assert
    assert store.file_path.exists()
    assert store.file_path.is_file()


def test_save_and_load_successful_roundtrip(
    store: FileStateStore, snapshot_instance: SimulationSnapshot
):
    """
    Tests that a snapshot can be saved and then loaded back, with the data
    remaining identical.
    """
    # Act: Save the snapshot
    store.save(snapshot_instance)

    # Act: Load the snapshot back
    loaded_snapshot = store.load()

    # Assert
    assert isinstance(loaded_snapshot, SimulationSnapshot)
    assert loaded_snapshot == snapshot_instance
    assert loaded_snapshot.simulation_id == "sim_save_load_test"
    assert loaded_snapshot.agents[0].agent_id == "agent1"


def test_load_raises_file_not_found(store: FileStateStore):
    """
    Tests that load() correctly raises FileNotFoundError if the file does not exist.
    """
    # Act & Assert
    with pytest.raises(FileNotFoundError):
        store.load()


def test_load_raises_value_error_for_malformed_json(store: FileStateStore):
    """
    Tests that load() raises a ValueError for a file with invalid JSON content.
    """
    # Arrange: Create a malformed JSON file
    store.file_path.parent.mkdir(exist_ok=True)
    # This is not valid JSON because it uses single quotes.
    store.file_path.write_text("{'invalid_json': True}")

    # Act & Assert
    # The test should check for a generic ValueError that wraps the Pydantic error.
    # The original code was too specific. Pydantic's validation error is the
    # root cause, which is then wrapped in a ValueError by the store.
    with pytest.raises(ValueError, match="Data validation error"):
        store.load()


def test_load_raises_value_error_for_invalid_schema(store: FileStateStore):
    """
    Tests that load() raises a ValueError (from Pydantic's ValidationError)
    if the JSON data does not match the expected schema.
    """
    # Arrange: Create a file with valid JSON but the wrong structure
    invalid_data = {
        "simulation_name": "wrong_key",
        "tick": 100,
    }  # Missing required fields
    store.file_path.parent.mkdir(exist_ok=True)
    store.file_path.write_text(json.dumps(invalid_data))

    # Act & Assert
    with pytest.raises(ValueError, match="Data validation error"):
        store.load()


def test_load_file_not_found(tmp_path: Path):
    store = FileStateStore(tmp_path / "non_existent.json")
    with pytest.raises(FileNotFoundError):
        store.load()
