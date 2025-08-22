# agent-concurrent/src/agent_concurrent/__init__.py
"""
Initializes the agent_concurrent package.

This file exposes the core components of the library, making them
easily accessible for import. It prioritizes importing the compiled
Rust extension for performance.
"""

try:
    # This imports the ParallelSystemRunner class from the compiled
    # `agent_concurrent_core.so` or `.pyd` file that our build script
    # places in this directory.
    from .agent_concurrent_core import ParallelSystemRunner

except ImportError:
    # This block can be used as a fallback if the Rust extension has not been
    # compiled. For now, we'll define a placeholder that raises an error.
    class ParallelSystemRunner:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Failed to import the Rust-based ParallelSystemRunner. "
                "Please compile the project first by running 'poetry install'."
            )

        def run(self, *args, **kwargs):
            raise NotImplementedError


# This tells Python what names to export when a user does `from agent_concurrent import *`
__all__ = [
    "ParallelSystemRunner",
]
