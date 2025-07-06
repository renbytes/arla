# src/agent-core/persistence/interface.py

from abc import ABC, abstractmethod
from typing import Any


class StateStoreInterface(ABC):
    @abstractmethod
    def save(self, state: Any, path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> Any:
        raise NotImplementedError
