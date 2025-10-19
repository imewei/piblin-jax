"""
Tests for lambda transforms and dynamic parameter transforms.

This module tests:
- LambdaTransform with simple functions
- LambdaTransform with JAX-compatible functions for JIT
- Dynamic parameter transforms (AutoScaleTransform, AutoBaselineTransform)
- Error handling for invalid functions
- JIT compilation effectiveness
"""

import pytest
import numpy as np
from quantiq.backend import jnp, is_jax_available
from quantiq.data.datasets import OneDimensionalDataset
from quantiq.transform.lambda_transform import (
    LambdaTransform,
    DynamicTransform,
    AutoScaleTransform,
    AutoBaselineTransform,
)


class TestLambdaTransform:
    """Test LambdaTransform with various functions."""

    def test_lambda_transform_simple_function(self):
        """LambdaTransform applies simple function correctly."""
        # Create test dataset
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = jnp.array([2.0, 4.0, 6.0, 8.0, 10.0])
        dataset = OneDimensionalDataset(
            independent_variable_data=x,
            dependent_variable_data=y
        )

        # Apply lambda transform: multiply by 2
        transform = LambdaTransform(lambda y: y * 2.0)
        result = transform.apply_to(dataset, make_copy=True)

        # Check result
        expected = y * 2.0
        np.testing.assert_allclose(result.dependent_variable_data, expected)

    def test_lambda_transform_with_x_and_y(self):
        """LambdaTransform applies function using both x and y."""
        # Create test dataset
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = jnp.array([2.0, 4.0, 6.0, 8.0, 10.0])
        dataset = OneDimensionalDataset(
            independent_variable_data=x,
            dependent_variable_data=y
        )

        # Apply lambda transform: divide y by max(x)
        transform = LambdaTransform(
            lambda x, y: y / jnp.max(x),
            use_x=True
        )
        result = transform.apply_to(dataset, make_copy=True)

        # Check result
        expected = y / jnp.max(x)
        np.testing.assert_allclose(result.dependent_variable_data, expected)

    def test_lambda_transform_jax_compatible(self):
        """LambdaTransform works with JAX-compatible functions."""
        # Create test dataset
        x = jnp.array([0.0, 0.5, 1.0, 1.5, 2.0])
        y = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dataset = OneDimensionalDataset(
            independent_variable_data=x,
            dependent_variable_data=y
        )

        # Apply JAX-compatible function: exp(y) * sin(y)
        transform = LambdaTransform(
            lambda y: jnp.exp(y) * jnp.sin(y),
            jit_compile=True
        )
        result = transform.apply_to(dataset, make_copy=True)

        # Check result
        expected = jnp.exp(y) * jnp.sin(y)
        np.testing.assert_allclose(result.dependent_variable_data, expected, rtol=1e-6)

    def test_lambda_transform_no_jit(self):
        """LambdaTransform works without JIT compilation."""
        # Create test dataset
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([2.0, 4.0, 6.0])
        dataset = OneDimensionalDataset(
            independent_variable_data=x,
            dependent_variable_data=y
        )

        # Apply without JIT
        transform = LambdaTransform(
            lambda y: y + 1.0,
            jit_compile=False
        )
        result = transform.apply_to(dataset, make_copy=True)

        # Check result
        expected = y + 1.0
        np.testing.assert_allclose(result.dependent_variable_data, expected)

    def test_lambda_transform_invalid_function(self):
        """LambdaTransform raises TypeError for non-callable."""
        with pytest.raises(TypeError, match="func must be callable"):
            LambdaTransform("not a function")

    def test_lambda_transform_preserves_x_data(self):
        """LambdaTransform preserves independent variable data."""
        # Create test dataset
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([2.0, 4.0, 6.0])
        dataset = OneDimensionalDataset(
            independent_variable_data=x,
            dependent_variable_data=y
        )

        # Apply transform
        transform = LambdaTransform(lambda y: y * 2.0)
        result = transform.apply_to(dataset, make_copy=True)

        # X should be unchanged
        np.testing.assert_allclose(result.independent_variable_data, x)


class TestAutoScaleTransform:
    """Test AutoScaleTransform with dynamic parameters."""

    def test_autoscale_to_zero_one(self):
        """AutoScaleTransform scales data to [0, 1]."""
        # Create test dataset
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
        dataset = OneDimensionalDataset(
            independent_variable_data=x,
            dependent_variable_data=y
        )

        # Apply autoscale
        transform = AutoScaleTransform(target_min=0.0, target_max=1.0)
        result = transform.apply_to(dataset, make_copy=True)

        # Check result scaled to [0, 1]
        result_y = result.dependent_variable_data
        assert float(jnp.min(result_y)) == pytest.approx(0.0)
        assert float(jnp.max(result_y)) == pytest.approx(1.0)

    def test_autoscale_to_minus_one_one(self):
        """AutoScaleTransform scales data to [-1, 1]."""
        # Create test dataset
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([5.0, 10.0, 15.0])
        dataset = OneDimensionalDataset(
            independent_variable_data=x,
            dependent_variable_data=y
        )

        # Apply autoscale
        transform = AutoScaleTransform(target_min=-1.0, target_max=1.0)
        result = transform.apply_to(dataset, make_copy=True)

        # Check result scaled to [-1, 1]
        result_y = result.dependent_variable_data
        assert float(jnp.min(result_y)) == pytest.approx(-1.0)
        assert float(jnp.max(result_y)) == pytest.approx(1.0)

    def test_autoscale_constant_data(self):
        """AutoScaleTransform handles constant data gracefully."""
        # Create dataset with constant values
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([5.0, 5.0, 5.0])
        dataset = OneDimensionalDataset(
            independent_variable_data=x,
            dependent_variable_data=y
        )

        # Apply autoscale
        transform = AutoScaleTransform(target_min=0.0, target_max=1.0)
        result = transform.apply_to(dataset, make_copy=True)

        # Should be scaled to target_min
        result_y = result.dependent_variable_data
        np.testing.assert_allclose(result_y, jnp.ones_like(y) * 0.0)


class TestAutoBaselineTransform:
    """Test AutoBaselineTransform with dynamic baseline correction."""

    def test_autobaseline_first(self):
        """AutoBaselineTransform subtracts baseline from first N points."""
        # Create test dataset with baseline offset
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        y = jnp.array([5.0, 5.1, 4.9, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0])
        dataset = OneDimensionalDataset(
            independent_variable_data=x,
            dependent_variable_data=y
        )

        # Apply baseline correction (first 4 points)
        transform = AutoBaselineTransform(n_points=4, method='first')
        result = transform.apply_to(dataset, make_copy=True)

        # Baseline should be ~5.0, so first points should be ~0
        result_y = result.dependent_variable_data
        assert float(jnp.mean(result_y[:4])) == pytest.approx(0.0, abs=0.1)

    def test_autobaseline_last(self):
        """AutoBaselineTransform subtracts baseline from last N points."""
        # Create test dataset
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        y = jnp.array([10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 5.0, 5.1, 4.9, 5.0])
        dataset = OneDimensionalDataset(
            independent_variable_data=x,
            dependent_variable_data=y
        )

        # Apply baseline correction (last 4 points)
        transform = AutoBaselineTransform(n_points=4, method='last')
        result = transform.apply_to(dataset, make_copy=True)

        # Baseline should be ~5.0, so last points should be ~0
        result_y = result.dependent_variable_data
        assert float(jnp.mean(result_y[-4:])) == pytest.approx(0.0, abs=0.1)

    def test_autobaseline_min(self):
        """AutoBaselineTransform subtracts minimum value as baseline."""
        # Create test dataset
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = jnp.array([10.0, 5.0, 15.0, 8.0, 12.0])
        dataset = OneDimensionalDataset(
            independent_variable_data=x,
            dependent_variable_data=y
        )

        # Apply baseline correction (min)
        transform = AutoBaselineTransform(method='min')
        result = transform.apply_to(dataset, make_copy=True)

        # Minimum should be 0
        result_y = result.dependent_variable_data
        assert float(jnp.min(result_y)) == pytest.approx(0.0)

    def test_autobaseline_invalid_method(self):
        """AutoBaselineTransform raises ValueError for invalid method."""
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([2.0, 4.0, 6.0])
        dataset = OneDimensionalDataset(
            independent_variable_data=x,
            dependent_variable_data=y
        )

        transform = AutoBaselineTransform(method='invalid')
        with pytest.raises(ValueError, match="Unknown method"):
            transform.apply_to(dataset, make_copy=True)


class TestDynamicTransformCombinations:
    """Test combining lambda and dynamic transforms."""

    def test_pipeline_with_lambda_and_dynamic(self):
        """Lambda and dynamic transforms work in pipeline."""
        from quantiq.transform.pipeline import Pipeline

        # Create test dataset
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = jnp.array([15.0, 15.1, 14.9, 20.0, 25.0])
        dataset = OneDimensionalDataset(
            independent_variable_data=x,
            dependent_variable_data=y
        )

        # Create pipeline: baseline correction -> scale -> multiply by 2
        pipeline = Pipeline([
            AutoBaselineTransform(n_points=3, method='first'),
            AutoScaleTransform(target_min=0.0, target_max=1.0),
            LambdaTransform(lambda y: y * 2.0),
        ])

        # Apply pipeline
        result = pipeline.apply_to(dataset, make_copy=True)

        # Result should be baseline corrected, scaled, then doubled
        # Max should be 2.0 (scaled to 1.0, then doubled)
        result_y = result.dependent_variable_data
        assert float(jnp.max(result_y)) == pytest.approx(2.0)
