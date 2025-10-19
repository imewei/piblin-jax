"""
Pytest configuration and fixtures for quantiq tests.
"""

import pytest
import numpy as np


# Pytest configuration hooks
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests requiring GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks performance benchmarks"
    )
    config.addinivalue_line(
        "markers", "visual: marks visual regression tests (pytest-mpl)"
    )


# Common fixtures for testing

@pytest.fixture
def sample_1d_data():
    """Generate sample 1D dataset for testing."""
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + 0.1 * np.random.randn(100)
    return x, y


@pytest.fixture
def sample_2d_data():
    """Generate sample 2D dataset for testing."""
    x = np.linspace(0, 5, 50)
    y = np.linspace(0, 5, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.cos(Y) + 0.1 * np.random.randn(50, 50)
    return x, y, Z


@pytest.fixture
def sample_metadata():
    """Generate sample metadata for testing."""
    conditions = {
        "temperature": 25.0,
        "sample": "Sample A",
        "replicate": 1,
    }
    details = {
        "operator": "Test Operator",
        "instrument": "Test Instrument",
        "date": "2025-10-18",
    }
    return conditions, details


@pytest.fixture
def tolerance():
    """Numerical tolerance for floating point comparisons."""
    return 1e-10


# Hypothesis configuration for property-based testing
from hypothesis import settings, Verbosity

# Register custom hypothesis profiles
settings.register_profile("ci", max_examples=1000, deadline=None)
settings.register_profile("dev", max_examples=100, deadline=None)
settings.register_profile("debug", max_examples=10, verbosity=Verbosity.verbose)

# Load the appropriate profile
settings.load_profile("dev")
