# tests/conftest.py
"""
Configuration file for the pytest test suite.

This file adds the project's source directory ('src') to the Python path,
allowing tests to import modules from the application using absolute paths.
"""

import os
import sys

# Get the absolute path to the project's root directory (agent-sim)
# This makes the test setup independent of where you run pytest from.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Get the absolute path to the source directory
SOURCE_PATH = os.path.join(PROJECT_ROOT, "src")

# Prepend the source path to sys.path
# This ensures that Python looks in your local 'src' directory for modules
# BEFORE looking in the installed site-packages. This is crucial for
# avoiding issues with stale editable installs.
if SOURCE_PATH not in sys.path:
    sys.path.insert(0, SOURCE_PATH)
