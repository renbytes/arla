# src/agent_persist/__init__.py
"""
A library for persisting and restoring agent-based simulation states.

This package provides a `StateStore` interface and concrete implementations
for saving and loading simulation snapshots using Pydantic models for
robust data validation and serialization.
"""

# Import the core classes to make them accessible at the package level.
# e.g., from agent_persist import StateStore, FileStateStore
from .models import SimulationSnapshot
from .store import FileStateStore, StateStore

# Define what is exposed when a user does 'from agent_persist import *'
__all__ = ["StateStore", "FileStateStore", "SimulationSnapshot"]
