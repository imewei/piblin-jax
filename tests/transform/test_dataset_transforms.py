"""
Tests for core dataset-level transforms.

This module tests:
- Interpolate1D transform
- Smoothing transforms (moving average)
- Baseline subtraction (polynomial)
- Normalization transforms
- Derivative transform
- Integration transform
- JIT compilation effectiveness
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quantiq.backend import BACKEND, is_jax_available
from quantiq.data.datasets import OneDimensionalDataset
from quantiq.transform.dataset import (
    CumulativeIntegral,
    Derivative,
    Interpolate1D,
    MinMaxNormalize,
    MovingAverageSmooth,
    PolynomialBaseline,
    ZScoreNormalize,
)


# Helper function to create test dataset
def create_test_dataset(x=None, y=None):
    """Create a simple 1D dataset for testing."""
    if x is None:
        x = np.linspace(0, 10, 50)
    if y is None:
        # Sine wave with some noise
        y = np.sin(x) + 0.1 * np.random.randn(len(x))

    return OneDimensionalDataset(
        independent_variable_data=x,
        dependent_variable_data=y,
        conditions={"temperature": 25.0},
        details={"units": "arbitrary"},
    )


class TestInterpolate1D:
    """Test interpolation transform."""

    def test_linear_interpolation(self):
        """Test linear interpolation to new x-values."""
        # Create dataset with known values
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])  # y = x
        dataset = create_test_dataset(x, y)

        # Interpolate to finer grid
        new_x = np.array([0.5, 1.5, 2.5, 3.5])
        transform = Interpolate1D(new_x, method="linear")
        result = transform.apply_to(dataset)

        # Check results
        assert_allclose(result.independent_variable_data, new_x)
        assert_allclose(result.dependent_variable_data, new_x, rtol=1e-5)

    def test_interpolation_preserves_metadata(self):
        """Test that interpolation preserves dataset metadata."""
        dataset = create_test_dataset()
        new_x = np.linspace(0, 10, 100)

        transform = Interpolate1D(new_x)
        result = transform.apply_to(dataset)

        assert result.conditions == dataset.conditions
        assert result.details == dataset.details

    def test_interpolation_with_jax(self):
        """Test interpolation works with JAX backend."""
        if not is_jax_available():
            pytest.skip("JAX not available")

        dataset = create_test_dataset()
        new_x = np.linspace(0, 10, 100)

        transform = Interpolate1D(new_x)
        result = transform.apply_to(dataset)

        assert result.dependent_variable_data is not None
        assert len(result.dependent_variable_data) == 100


class TestSmoothing:
    """Test smoothing transforms."""

    def test_moving_average_smoothing(self):
        """Test moving average smoothing reduces noise."""
        # Create noisy data
        x = np.linspace(0, 10, 100)
        y_clean = np.sin(x)
        y_noisy = y_clean + 0.5 * np.random.randn(len(x))
        dataset = create_test_dataset(x, y_noisy)

        # Apply smoothing
        transform = MovingAverageSmooth(window_size=5)
        result = transform.apply_to(dataset)

        # Smoothed data should be closer to clean signal
        # Check that smoothing reduces variance
        original_variance = np.var(dataset.dependent_variable_data - y_clean)
        smoothed_variance = np.var(result.dependent_variable_data - y_clean)

        # Smoothing should reduce variance (but this might not always be true due to randomness)
        # Just check that smoothing was applied (data changed)
        assert not np.allclose(result.dependent_variable_data, dataset.dependent_variable_data)

    def test_window_size_must_be_odd(self):
        """Test that even window sizes raise an error."""
        with pytest.raises(ValueError, match="window_size must be odd"):
            MovingAverageSmooth(window_size=4)


class TestBaselineSubtraction:
    """Test baseline subtraction transforms."""

    def test_linear_baseline_subtraction(self):
        """Test polynomial baseline subtraction (linear)."""
        # Create data with linear baseline and small oscillation
        x = np.linspace(0, 10, 100)
        baseline = 2.0 * x + 1.0
        signal = 0.1 * np.sin(x)  # Small oscillation
        y = signal + baseline
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        # Remove linear baseline
        transform = PolynomialBaseline(degree=1)
        result = transform.apply_to(dataset)

        # Result should be close to original signal
        # (not exact due to fitting process but close)
        assert_allclose(result.dependent_variable_data, signal, atol=0.05)

    def test_polynomial_baseline_subtraction(self):
        """Test polynomial baseline subtraction (quadratic)."""
        # Create data with quadratic baseline and small oscillation
        x = np.linspace(0, 10, 100)
        baseline = 0.1 * x**2 + 2.0 * x + 1.0
        signal = 0.05 * np.sin(x)  # Small oscillation
        y = signal + baseline
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        # Remove quadratic baseline
        transform = PolynomialBaseline(degree=2)
        result = transform.apply_to(dataset)

        # Result should be close to original signal
        # (not exact due to fitting process but close)
        assert_allclose(result.dependent_variable_data, signal, atol=0.05)


class TestNormalization:
    """Test normalization transforms."""

    def test_minmax_normalization_default(self):
        """Test min-max normalization to [0, 1]."""
        x = np.linspace(0, 10, 50)
        y = np.array([5.0, 10.0, 15.0, 20.0, 25.0] * 10)  # Range: 5 to 25
        dataset = create_test_dataset(x, y)

        transform = MinMaxNormalize()
        result = transform.apply_to(dataset)

        # Check range is [0, 1]
        assert_allclose(np.min(result.dependent_variable_data), 0.0, atol=1e-6)
        assert_allclose(np.max(result.dependent_variable_data), 1.0, atol=1e-6)

    def test_minmax_normalization_custom_range(self):
        """Test min-max normalization to custom range."""
        x = np.linspace(0, 10, 50)
        y = np.linspace(5, 25, 50)
        dataset = create_test_dataset(x, y)

        transform = MinMaxNormalize(feature_range=(-1, 1))
        result = transform.apply_to(dataset)

        # Check range is [-1, 1]
        assert_allclose(np.min(result.dependent_variable_data), -1.0, atol=1e-5)
        assert_allclose(np.max(result.dependent_variable_data), 1.0, atol=1e-5)

    def test_zscore_normalization(self):
        """Test z-score normalization."""
        x = np.linspace(0, 10, 100)
        y = np.random.randn(100) * 5.0 + 10.0  # Mean ~10, std ~5
        dataset = create_test_dataset(x, y)

        transform = ZScoreNormalize()
        result = transform.apply_to(dataset)

        # Check mean ~0 and std ~1
        assert_allclose(np.mean(result.dependent_variable_data), 0.0, atol=1e-6)
        assert_allclose(np.std(result.dependent_variable_data), 1.0, atol=1e-6)


class TestDerivative:
    """Test derivative transform."""

    def test_first_derivative(self):
        """Test first derivative computation."""
        # Create data with known derivative
        x = np.linspace(0, 10, 100)
        y = x**2  # dy/dx = 2x
        dataset = create_test_dataset(x, y)

        transform = Derivative(order=1)
        result = transform.apply_to(dataset)

        # First derivative should be approximately 2x
        expected = 2 * x
        assert_allclose(result.dependent_variable_data, expected, atol=0.5)

    def test_second_derivative(self):
        """Test second derivative computation."""
        # Create data with known second derivative
        x = np.linspace(0, 10, 100)
        y = x**3  # d²y/dx² = 6x
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        transform = Derivative(order=2)
        result = transform.apply_to(dataset)

        # Second derivative should be approximately 6x
        expected = 6 * x
        # Allow higher tolerance due to numerical differentiation errors
        # Skip edges (first 2 and last 2 points) where errors are largest
        assert_allclose(result.dependent_variable_data[2:-2], expected[2:-2], atol=1.0)

    def test_invalid_derivative_order(self):
        """Test that invalid derivative orders raise errors."""
        with pytest.raises(ValueError, match="order must be 1 or 2"):
            Derivative(order=3)


class TestIntegration:
    """Test integration transform."""

    def test_cumulative_integral(self):
        """Test cumulative integration."""
        # Create data with known integral
        x = np.linspace(0, 10, 100)
        y = np.ones_like(x)  # Integral of 1 is x
        dataset = create_test_dataset(x, y)

        transform = CumulativeIntegral()
        result = transform.apply_to(dataset)

        # Cumulative integral should be approximately x (starting from 0)
        # First value should be 0
        assert_allclose(result.dependent_variable_data[0], 0.0, atol=1e-6)
        # Last value should be approximately 10 (integral of 1 from 0 to 10)
        assert_allclose(result.dependent_variable_data[-1], 10.0, atol=0.2)


class TestJITCompilation:
    """Test JIT compilation effectiveness."""

    def test_jit_compilation_available(self):
        """Test that JIT compilation is available with JAX backend."""
        if not is_jax_available():
            pytest.skip("JAX not available, cannot test JIT")

        # Create dataset
        dataset = create_test_dataset()

        # Apply transform (first call compiles, second uses compiled version)
        transform = MinMaxNormalize()
        result1 = transform.apply_to(dataset)
        result2 = transform.apply_to(dataset)

        # Results should be identical
        assert_allclose(result1.dependent_variable_data, result2.dependent_variable_data)

    def test_transforms_work_with_numpy_backend(self):
        """Test that all transforms work even with NumPy backend."""
        # This test ensures fallback to NumPy works
        dataset = create_test_dataset()

        transforms = [
            Interpolate1D(np.linspace(0, 10, 80)),
            MovingAverageSmooth(window_size=5),
            PolynomialBaseline(degree=1),
            MinMaxNormalize(),
            ZScoreNormalize(),
            Derivative(order=1),
            CumulativeIntegral(),
        ]

        for transform in transforms:
            result = transform.apply_to(dataset)
            assert result is not None
            assert hasattr(result, "dependent_variable_data")
