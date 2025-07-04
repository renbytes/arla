# src/agent_concurrent/__init__.py
"""
A library for concurrent and parallel execution of agent-based simulation systems.

This package provides runner classes that can execute a list of systems
serially or concurrently using Python's asyncio library.
"""

# Import the core runner classes to make them accessible at the package level.
# e.g., from agent_concurrent import AsyncSystemRunner
from .runners import AsyncSystemRunner, SerialSystemRunner

# Define what is exposed when a user does 'from agent_concurrent import *'
__all__ = ["AsyncSystemRunner", "SerialSystemRunner"]
