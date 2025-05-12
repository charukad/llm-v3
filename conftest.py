"""
Test configuration and fixtures.
"""
import os
import sys
import pytest

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Configure pytest-asyncio
def pytest_configure(config):
    """Configure pytest."""
    config.option.asyncio_mode = "strict"
    config.option.asyncio_default_fixture_loop_scope = "function" 