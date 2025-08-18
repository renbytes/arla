# tests/agent-engine/utils/test_manifest.py

import subprocess
from unittest.mock import patch

from agent_engine.utils.manifest import create_run_manifest, get_git_commit_hash

# --- Tests for get_git_commit_hash ---


@patch("subprocess.check_output")
@patch("pathlib.Path.exists", return_value=True)
def test_get_git_commit_hash_success(mock_path_exists, mock_subprocess):
    """
    Tests that get_git_commit_hash returns the correct hash when git
    command succeeds.
    """
    # Arrange
    expected_hash = "a1b2c3d4e5f6"
    mock_subprocess.return_value = f"{expected_hash}\n".encode("utf-8")

    # Act
    commit_hash = get_git_commit_hash()

    # Assert
    assert commit_hash == expected_hash
    mock_path_exists.assert_called_once()
    mock_subprocess.assert_called_once()


@patch("subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "git"))
@patch("pathlib.Path.exists", return_value=True)
def test_get_git_commit_hash_failure(mock_path_exists, mock_subprocess_error):
    """
    Tests that get_git_commit_hash returns None when the git command fails.
    """
    # Act
    commit_hash = get_git_commit_hash()

    # Assert
    assert commit_hash is None


@patch("pathlib.Path.exists", return_value=False)
def test_get_git_commit_hash_no_git_directory(mock_path_exists):
    """
    Tests that get_git_commit_hash returns None when the .git directory
    is not found.
    """
    # Act
    commit_hash = get_git_commit_hash()

    # Assert
    assert commit_hash is None


# --- Tests for create_run_manifest ---


@patch("agent_engine.utils.manifest.get_git_commit_hash", return_value="test_hash")
def test_create_run_manifest(mock_get_hash):
    """
    Tests that create_run_manifest correctly assembles the manifest
    dictionary with all required fields.
    """
    # Arrange
    run_id = "sim_123"
    experiment_id = "exp_456"
    task_id = "task_789"
    config = {"simulation": {"random_seed": 42}}

    # Act
    manifest = create_run_manifest(run_id, experiment_id, task_id, config)

    # Assert
    assert manifest["run_id"] == run_id
    assert manifest["experiment_id"] == experiment_id
    assert manifest["task_id"] == task_id
    assert manifest["git_commit_hash"] == "test_hash"
    assert manifest["random_seed"] == 42
    assert "timestamp_utc" in manifest
    mock_get_hash.assert_called_once()
