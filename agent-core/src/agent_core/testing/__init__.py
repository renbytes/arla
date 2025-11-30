# src/agent_core/testing/__init__.py

"""
A package providing tools for testing and analyzing agent behavior.
This includes the Behavioral Query Language (BQL).
"""

# [FIX] Only import and expose the BQL class.
# The assert_always/assert_eventually methods are accessed via the BQL instance.
from .bql import BQL

__all__ = ["BQL"]
