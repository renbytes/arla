# src/agent_persist/store.py

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from pydantic import ValidationError

from .models import SimulationSnapshot


class StateStore(ABC):
    """
    Abstract Base Class defining the interface for a state persistence store.
    """

    @abstractmethod
    def save(self, snapshot: SimulationSnapshot) -> None:
        """
        Saves a simulation snapshot to the persistence layer.

        Args:
            snapshot: A SimulationSnapshot object to be saved.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self) -> SimulationSnapshot:
        """
        Loads a simulation snapshot from the persistence layer.

        Returns:
            A populated SimulationSnapshot object.
        """
        raise NotImplementedError


class FileStateStore(StateStore):
    """
    A concrete implementation of StateStore that saves and loads simulation
    snapshots to and from a local JSON file.
    """

    def __init__(self, file_path: Union[str, Path]):
        """
        Initializes the FileStateStore with the path to the snapshot file.

        Args:
            file_path: The path (as a string or Path object) where the
                       snapshot file is stored.
        """
        self.file_path = Path(file_path)

    def save(self, snapshot: SimulationSnapshot) -> None:
        """
        Serializes the SimulationSnapshot to a JSON string and writes it to the file.

        This method will create the parent directory if it does not exist.

        Args:
            snapshot: The SimulationSnapshot object to save.

        Raises:
            IOError: If there is an issue writing the file to disk.
        """
        try:
            # Ensure the parent directory exists
            self.file_path.parent.mkdir(parents=True, exist_ok=True)

            # Use Pydantic's model_dump_json for robust serialization
            json_data = snapshot.model_dump_json(indent=2)

            self.file_path.write_text(json_data, encoding="utf-8")
            print(f"Successfully saved snapshot to {self.file_path}")

        except (IOError, TypeError) as e:
            print(f"Error: Could not save snapshot to {self.file_path}. Reason: {e}")
            raise

    def load(self) -> SimulationSnapshot:
        """
        Loads and validates a simulation snapshot from the JSON file.

        Returns:
            A populated SimulationSnapshot object.

        Raises:
            FileNotFoundError: If the snapshot file does not exist.
            ValueError: If the file contains invalid JSON or does not match the
                        SimulationSnapshot schema.
        """
        if not self.file_path.is_file():
            raise FileNotFoundError(f"Snapshot file not found at: {self.file_path}")

        try:
            json_text = self.file_path.read_text(encoding="utf-8")

            # Use Pydantic's model_validate_json for robust parsing and validation
            snapshot = SimulationSnapshot.model_validate_json(json_text)
            print(f"Successfully loaded snapshot from {self.file_path}")
            return snapshot

        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {self.file_path}: {e}")
        except ValidationError as e:
            raise ValueError(
                f"Data validation error when loading snapshot from {self.file_path}: {e}"
            )
        except IOError as e:
            print(f"Error: Could not read snapshot from {self.file_path}. Reason: {e}")
            raise
