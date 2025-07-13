# agent-core/tests/conftest.py
"""
Configuration file for the pytest test suite.
This file adds the project's source directory ('src') to the Python path,
allowing tests to import modules from the application using absolute paths.
"""

import os
import sys

# Get the absolute path to the 'agent-core' directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Get the absolute path to the source directory within 'agent-core'
SOURCE_PATH = os.path.join(PROJECT_ROOT, "src")

# Prepend the source path to sys.path to ensure local modules are found first
if SOURCE_PATH not in sys.path:
    sys.path.insert(0, SOURCE_PATH)
