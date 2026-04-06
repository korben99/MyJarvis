"""pytest configuration for Jarvis tests."""
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: tests requiring a running Jarvis server on localhost:8000",
    )
