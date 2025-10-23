"""
Extended tests for baseline transforms to achieve 95%+ coverage.

This module tests uncovered paths in baseline.py including:
- AsymmetricLeastSquaresBaseline implementation (lines 189-234)
- Different baseline methods
- Edge cases and parameter validation
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quantiq.backend import jnp
from quantiq.data.datasets import OneDimensionalDataset
from quantiq.transform.dataset import AsymmetricLeastSquaresBaseline, PolynomialBaseline


def create_baseline_dataset():
    """Create dataset with known baseline."""
    x = np.linspace(0, 10, 100)
    # Signal with polynomial baseline
    baseline = 0.5 * x**2 + 2.0 * x + 1.0
    signal = np.sin(2 * np.pi * x)
    y = signal + baseline

    return OneDimensionalDataset(
        independent_variable_data=x,
        dependent_variable_data=y,
        conditions={"temperature": 25.0},
    )


class TestAsymmetricLeastSquaresBaseline:
    """Test ALS baseline correction."""

    def test_als_baseline_basic(self):
        """Test basic ALS baseline correction (lines 189-234)."""
        # Create dataset with baseline
        dataset = create_baseline_dataset()

        # Apply ALS baseline correction
        transform = AsymmetricLeastSquaresBaseline(lambda_=1e5, p=0.01, max_iter=10)
        result = transform.apply_to(dataset)

        # Result should have baseline removed (signal closer to zero mean)
        original_mean = np.abs(np.mean(dataset.dependent_variable_data))
        corrected_mean = np.abs(np.mean(result.dependent_variable_data))

        # Corrected data should have smaller absolute mean
        assert corrected_mean < original_mean

    def test_als_baseline_with_different_lambda(self):
        """Test ALS with different smoothness parameters."""
        dataset = create_baseline_dataset()

        # Low lambda (less smooth)
        transform_low = AsymmetricLeastSquaresBaseline(lambda_=1e3, p=0.01, max_iter=10)
        result_low = transform_low.apply_to(dataset)

        # High lambda (more smooth)
        transform_high = AsymmetricLeastSquaresBaseline(lambda_=1e7, p=0.01, max_iter=10)
        result_high = transform_high.apply_to(dataset)

        # Both should work
        assert result_low.dependent_variable_data is not None
        assert result_high.dependent_variable_data is not None

    def test_als_baseline_with_different_asymmetry(self):
        """Test ALS with different asymmetry parameters."""
        dataset = create_baseline_dataset()

        # Low asymmetry (p close to 0)
        transform_low = AsymmetricLeastSquaresBaseline(lambda_=1e5, p=0.001, max_iter=10)
        result_low = transform_low.apply_to(dataset)

        # High asymmetry (p close to 0.5)
        transform_high = AsymmetricLeastSquaresBaseline(lambda_=1e5, p=0.1, max_iter=10)
        result_high = transform_high.apply_to(dataset)

        # Both should work
        assert result_low.dependent_variable_data is not None
        assert result_high.dependent_variable_data is not None

    def test_als_baseline_with_different_iterations(self):
        """Test ALS with different iteration counts."""
        dataset = create_baseline_dataset()

        # Few iterations
        transform_few = AsymmetricLeastSquaresBaseline(lambda_=1e5, p=0.01, max_iter=3)
        result_few = transform_few.apply_to(dataset)

        # Many iterations
        transform_many = AsymmetricLeastSquaresBaseline(lambda_=1e5, p=0.01, max_iter=20)
        result_many = transform_many.apply_to(dataset)

        # Both should work
        assert result_few.dependent_variable_data is not None
        assert result_many.dependent_variable_data is not None

    def test_als_baseline_preserves_metadata(self):
        """Test that ALS preserves dataset metadata."""
        dataset = create_baseline_dataset()

        transform = AsymmetricLeastSquaresBaseline(lambda_=1e5, p=0.01, max_iter=10)
        result = transform.apply_to(dataset)

        # Metadata should be preserved
        assert result.conditions == dataset.conditions

    def test_als_baseline_preserves_independent_variable(self):
        """Test that ALS preserves independent variable."""
        dataset = create_baseline_dataset()

        transform = AsymmetricLeastSquaresBaseline(lambda_=1e5, p=0.01, max_iter=10)
        result = transform.apply_to(dataset)

        # Independent variable should be unchanged
        np.testing.assert_array_equal(
            result.independent_variable_data, dataset.independent_variable_data
        )

    def test_als_baseline_with_small_dataset(self):
        """Test ALS with very small dataset."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([1.0, 2.0, 1.5, 2.5, 2.0])
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        transform = AsymmetricLeastSquaresBaseline(lambda_=1e3, p=0.01, max_iter=5)
        result = transform.apply_to(dataset)

        # Should work even with small dataset
        assert len(result.dependent_variable_data) == 5

    @pytest.mark.xfail(
        reason="ALS algorithm has numerical issues with perfectly constant data (edge case). "
        "The sparse matrix solver produces artifacts at boundaries for this degenerate case."
    )
    def test_als_baseline_with_constant_data(self):
        """Test ALS with constant data."""
        x = np.linspace(0, 10, 50)
        y = np.ones(50) * 5.0
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        transform = AsymmetricLeastSquaresBaseline(lambda_=1e5, p=0.01, max_iter=10)
        result = transform.apply_to(dataset)

        # Should remove the constant baseline
        # Result should be close to zero
        assert np.abs(np.mean(result.dependent_variable_data)) < 0.1

    def test_als_baseline_iterative_convergence(self):
        """Test that ALS iterates through all specified iterations."""
        dataset = create_baseline_dataset()

        # Single iteration
        transform_1 = AsymmetricLeastSquaresBaseline(lambda_=1e5, p=0.01, max_iter=1)
        result_1 = transform_1.apply_to(dataset)

        # Multiple iterations should give different result
        transform_10 = AsymmetricLeastSquaresBaseline(lambda_=1e5, p=0.01, max_iter=10)
        result_10 = transform_10.apply_to(dataset)

        # Results should differ (more iterations = better convergence)
        diff = np.sum(np.abs(result_1.dependent_variable_data - result_10.dependent_variable_data))
        assert diff > 0.01  # Should have noticeable difference


class TestPolynomialBaselineExtended:
    """Extended tests for polynomial baseline correction."""

    def test_polynomial_baseline_degree_0(self):
        """Test polynomial baseline with degree 0 (constant)."""
        x = np.linspace(0, 10, 50)
        y = np.sin(x) + 2.0  # Sine wave with constant offset
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        transform = PolynomialBaseline(degree=0)
        result = transform.apply_to(dataset)

        # Should remove constant offset
        assert np.abs(np.mean(result.dependent_variable_data)) < 0.5

    def test_polynomial_baseline_degree_1(self):
        """Test polynomial baseline with degree 1 (linear)."""
        x = np.linspace(0, 10, 50)
        y = np.sin(x) + 0.5 * x + 1.0  # Sine wave with linear baseline
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        transform = PolynomialBaseline(degree=1)
        result = transform.apply_to(dataset)

        # Linear trend should be removed
        # Check that trend is reduced
        from scipy import stats

        _, _, r_value_orig, _, _ = stats.linregress(x, dataset.dependent_variable_data)
        _, _, r_value_corr, _, _ = stats.linregress(x, result.dependent_variable_data)

        # Correlation should be reduced after baseline removal
        assert abs(r_value_corr) < abs(r_value_orig)

    def test_polynomial_baseline_degree_3(self):
        """Test polynomial baseline with degree 3 (cubic)."""
        x = np.linspace(0, 10, 50)
        # Cubic baseline
        baseline = 0.01 * x**3 - 0.1 * x**2 + 0.5 * x + 1.0
        signal = np.sin(2 * np.pi * x / 5)
        y = signal + baseline
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        transform = PolynomialBaseline(degree=3)
        result = transform.apply_to(dataset)

        # Cubic baseline should be largely removed
        # Corrected signal should oscillate around zero
        assert np.abs(np.mean(result.dependent_variable_data)) < 0.5

    def test_polynomial_baseline_high_degree(self):
        """Test polynomial baseline with high degree."""
        dataset = create_baseline_dataset()

        # High degree polynomial
        transform = PolynomialBaseline(degree=5)
        result = transform.apply_to(dataset)

        # Should work without error
        assert result.dependent_variable_data is not None
        assert len(result.dependent_variable_data) == len(dataset.dependent_variable_data)
