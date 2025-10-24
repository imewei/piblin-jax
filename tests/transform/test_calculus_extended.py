"""
Extended tests for calculus transforms to achieve 95%+ coverage.

This module tests uncovered paths in calculus.py including:
- Forward and backward difference methods (lines 113-114, 120-121)
- Second derivative computation (lines 149-156)
- DefiniteIntegral edge cases (lines 335-392)
- Different integration methods and bounds
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from piblin_jax.backend import jnp
from piblin_jax.data.datasets import OneDimensionalDataset
from piblin_jax.transform.dataset import CumulativeIntegral, DefiniteIntegral, Derivative


def create_polynomial_dataset():
    """Create dataset with known polynomial for testing derivatives."""
    x = np.linspace(0, 10, 100)
    # y = x^2, so dy/dx = 2x, d2y/dx2 = 2
    y = x**2
    return OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)


def create_sine_dataset():
    """Create dataset with sine wave for testing integration."""
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)
    return OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)


class TestDerivativeForwardBackward:
    """Test forward and backward difference methods."""

    def test_forward_difference_method(self):
        """Test forward difference derivative method (lines 113-114, 149-152)."""
        dataset = create_polynomial_dataset()

        # Apply forward difference
        transform = Derivative(method="forward", order=1)
        result = transform.apply_to(dataset)

        # For y = x^2, dy/dx = 2x
        x = dataset.independent_variable_data
        expected = 2 * x

        # Forward difference should be reasonably close
        assert_allclose(result.dependent_variable_data, expected, rtol=0.1, atol=0.5)

    def test_backward_difference_method(self):
        """Test backward difference derivative method (lines 120-121, 153-154)."""
        dataset = create_polynomial_dataset()

        # Apply backward difference
        transform = Derivative(method="backward", order=1)
        result = transform.apply_to(dataset)

        # For y = x^2, dy/dx = 2x
        x = dataset.independent_variable_data
        expected = 2 * x

        # Backward difference should be reasonably close
        assert_allclose(result.dependent_variable_data, expected, rtol=0.1, atol=0.5)

    def test_gradient_method_default(self):
        """Test gradient method (central differences)."""
        dataset = create_polynomial_dataset()

        # Apply gradient (default method)
        transform = Derivative(method="gradient", order=1)
        result = transform.apply_to(dataset)

        # For y = x^2, dy/dx = 2x
        x = dataset.independent_variable_data
        expected = 2 * x

        # Central differences should be more accurate
        assert_allclose(result.dependent_variable_data, expected, rtol=0.05, atol=0.2)

    def test_invalid_derivative_method(self):
        """Test invalid derivative method raises error (line 156)."""
        dataset = create_polynomial_dataset()

        # Invalid method should raise ValueError
        transform = Derivative(method="invalid", order=1)
        with pytest.raises(ValueError, match="Unknown method"):
            transform.apply_to(dataset)


class TestSecondDerivative:
    """Test second derivative computation."""

    def test_second_derivative_gradient_method(self):
        """Test second derivative with gradient method (line 159+)."""
        dataset = create_polynomial_dataset()

        # Apply second derivative
        transform = Derivative(method="gradient", order=2)
        result = transform.apply_to(dataset)

        # For y = x^2, d2y/dx2 = 2 (constant)
        # Second derivative should be approximately 2
        expected = 2.0
        mean_second_deriv = np.mean(result.dependent_variable_data)

        assert_allclose(mean_second_deriv, expected, rtol=0.2, atol=0.5)

    def test_second_derivative_forward_method(self):
        """Test second derivative with forward difference."""
        dataset = create_polynomial_dataset()

        # Apply second derivative with forward method
        transform = Derivative(method="forward", order=2)
        result = transform.apply_to(dataset)

        # Should produce reasonable result
        # For y = x^2, d2y/dx2 = 2
        mean_second_deriv = np.mean(result.dependent_variable_data)
        assert_allclose(mean_second_deriv, 2.0, rtol=0.3, atol=1.0)

    def test_second_derivative_backward_method(self):
        """Test second derivative with backward difference."""
        dataset = create_polynomial_dataset()

        # Apply second derivative with backward method
        transform = Derivative(method="backward", order=2)
        result = transform.apply_to(dataset)

        # Should produce reasonable result
        mean_second_deriv = np.mean(result.dependent_variable_data)
        assert_allclose(mean_second_deriv, 2.0, rtol=0.3, atol=1.0)

    def test_second_derivative_preserves_metadata(self):
        """Test that second derivative preserves metadata."""
        x = np.linspace(0, 10, 50)
        y = x**3
        dataset = OneDimensionalDataset(
            independent_variable_data=x,
            dependent_variable_data=y,
            conditions={"temperature": 25.0},
            details={"sample": "test"},
        )

        transform = Derivative(method="gradient", order=2)
        result = transform.apply_to(dataset)

        # Metadata should be preserved
        assert result.conditions == dataset.conditions
        assert result.details == dataset.details


class TestDefiniteIntegral:
    """Test definite integral with different configurations."""

    def test_definite_integral_full_range(self):
        """Test definite integral over full data range (lines 335-392)."""
        dataset = create_sine_dataset()

        # Integrate over full range (no bounds specified)
        transform = DefiniteIntegral(x_min=None, x_max=None, method="trapezoid")
        result = transform.apply_to(dataset)

        # Should store integral in details
        assert "integral_value" in result.details
        assert "integral_x_min" in result.details
        assert "integral_x_max" in result.details

        # Integral of sin(x) from 0 to 2π should be close to 0
        integral_value = result.details["integral_value"]
        assert_allclose(integral_value, 0.0, atol=0.1)

    def test_definite_integral_partial_range(self):
        """Test definite integral over partial range."""
        dataset = create_sine_dataset()

        # Integrate from 0 to π (should be positive)
        transform = DefiniteIntegral(x_min=0.0, x_max=np.pi, method="trapezoid")
        result = transform.apply_to(dataset)

        # Integral of sin(x) from 0 to π should be close to 2
        integral_value = result.details["integral_value"]
        assert integral_value > 1.5  # Should be positive and > 1.5

    def test_definite_integral_only_x_min(self):
        """Test definite integral with only x_min specified."""
        dataset = create_sine_dataset()

        # Integrate from π to end
        transform = DefiniteIntegral(x_min=np.pi, x_max=None, method="trapezoid")
        result = transform.apply_to(dataset)

        # Should work
        assert "integral_value" in result.details
        assert result.details["integral_x_min"] >= np.pi

    def test_definite_integral_only_x_max(self):
        """Test definite integral with only x_max specified."""
        dataset = create_sine_dataset()

        # Integrate from start to π
        transform = DefiniteIntegral(x_min=None, x_max=np.pi, method="trapezoid")
        result = transform.apply_to(dataset)

        # Should work
        assert "integral_value" in result.details
        assert result.details["integral_x_max"] <= np.pi

    def test_definite_integral_small_region(self):
        """Test definite integral with very small region (line 377-378)."""
        dataset = create_sine_dataset()

        # Create a region with less than 2 points
        x_data = dataset.independent_variable_data
        x_min = x_data[50]
        x_max = x_data[50] + 0.001  # Very small range, might have < 2 points

        transform = DefiniteIntegral(x_min=x_min, x_max=x_max, method="trapezoid")
        result = transform.apply_to(dataset)

        # Should handle gracefully
        assert "integral_value" in result.details

    def test_definite_integral_single_point_region(self):
        """Test definite integral with region containing single point."""
        x = np.linspace(0, 10, 20)
        y = np.sin(x)
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        # Region with exactly one point (or no points)
        x_min = -1.0  # Before data range
        x_max = -0.5  # Before data range

        transform = DefiniteIntegral(x_min=x_min, x_max=x_max, method="trapezoid")
        result = transform.apply_to(dataset)

        # Should return 0 for empty region
        assert result.details["integral_value"] == 0.0

    def test_definite_integral_invalid_method(self):
        """Test definite integral with invalid method (line 383)."""
        dataset = create_sine_dataset()

        # Invalid method should raise ValueError
        transform = DefiniteIntegral(x_min=None, x_max=None, method="invalid")
        with pytest.raises(ValueError, match="Unknown method"):
            transform.apply_to(dataset)

    def test_definite_integral_preserves_original_data(self):
        """Test that definite integral preserves original dataset data."""
        dataset = create_sine_dataset()

        # Store original data
        original_y = dataset.dependent_variable_data.copy()

        # Apply integration
        transform = DefiniteIntegral(x_min=None, x_max=None, method="trapezoid")
        result = transform.apply_to(dataset)

        # Original dependent variable should be unchanged
        np.testing.assert_array_equal(result.dependent_variable_data, original_y)

    def test_definite_integral_adds_to_existing_details(self):
        """Test that definite integral adds to existing details."""
        # Create dataset with existing details
        x = np.linspace(0, 2 * np.pi, 100)
        y = np.sin(x)
        dataset = OneDimensionalDataset(
            independent_variable_data=x,
            dependent_variable_data=y,
            conditions={"temperature": 25.0},
            details={"existing_key": "existing_value"},
        )

        # Apply integration
        transform = DefiniteIntegral(x_min=None, x_max=None, method="trapezoid")
        result = transform.apply_to(dataset)

        # Should preserve existing details and add new ones
        assert "existing_key" in result.details
        assert "integral_value" in result.details
        assert result.details["existing_key"] == "existing_value"

    def test_definite_integral_with_none_details(self):
        """Test definite integral when dataset has None details (line 386-387)."""
        x = np.linspace(0, 10, 50)
        y = np.sin(x)
        dataset = OneDimensionalDataset(
            independent_variable_data=x, dependent_variable_data=y, details=None
        )

        # Apply integration
        transform = DefiniteIntegral(x_min=None, x_max=None, method="trapezoid")
        result = transform.apply_to(dataset)

        # Should create details dict
        assert result.details is not None
        assert "integral_value" in result.details


class TestCumulativeIntegralExtended:
    """Extended tests for cumulative integral."""

    def test_cumulative_integral_trapezoid(self):
        """Test cumulative integral with trapezoid method."""
        # Create dataset where we know the integral
        x = np.linspace(0, 10, 100)
        y = np.ones(100)  # Constant 1, integral should be x
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        # Apply cumulative integration
        transform = CumulativeIntegral(method="trapezoid")
        result = transform.apply_to(dataset)

        # Cumulative integral should be approximately linear
        # For constant y=1, cumulative integral should be ≈ x
        expected = x - x[0]
        assert_allclose(result.dependent_variable_data, expected, rtol=0.05, atol=0.1)

    def test_cumulative_integral_preserves_length(self):
        """Test that cumulative integral preserves data length."""
        dataset = create_sine_dataset()

        transform = CumulativeIntegral(method="trapezoid")
        result = transform.apply_to(dataset)

        # Length should be preserved
        assert len(result.dependent_variable_data) == len(dataset.dependent_variable_data)
        assert len(result.independent_variable_data) == len(dataset.independent_variable_data)
