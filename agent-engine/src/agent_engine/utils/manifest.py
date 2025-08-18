# agent-engine/src/agent_engine/utils/manifest.py
"""
Utilities for creating and managing simulation run manifests.

This module provides functions to capture metadata about a simulation run,
such as the Git commit hash and configuration, to ensure reproducibility.
"""

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def get_git_commit_hash() -> str | None:
    """
    Gets the current git commit hash of the repository.

    This function is designed to be robust, even when the script is not run
    from the repository's root directory.

    Returns:
        The git commit hash as a string, or None if it cannot be determined.
    """
    try:
        # Navigate to the project root to find the .git directory
        project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
        git_dir = project_root / ".git"

        if not git_dir.exists():
            print("Warning: .git directory not found. Cannot determine commit hash.")
            return None

        commit_hash = (
            subprocess.check_output(
                ["git", f"--git-dir={git_dir}", "rev-parse", "HEAD"],
                stderr=subprocess.STDOUT,
            )
            .strip()
            .decode("utf-8")
        )
        return commit_hash
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: Could not get git commit hash. Is git installed and in a repo?")
        return None


def create_run_manifest(
    run_id: str,
    experiment_id: Optional[str],
    task_id: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Creates a dictionary containing the run manifest data.

    Args:
        run_id: The unique ID for the simulation run.
        experiment_id: The ID of the experiment this run belongs to. Can be None.
        task_id: The Celery task ID or local task identifier.
        config: The fully resolved configuration dictionary for the run.

    Returns:
        A dictionary containing the complete run manifest.
    """
    return {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "task_id": task_id,
        "timestamp_utc": datetime.utcnow().isoformat(),
        "git_commit_hash": get_git_commit_hash(),
        "random_seed": config.get("simulation", {}).get("random_seed"),
    }
