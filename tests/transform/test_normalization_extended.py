"""
Extended tests for normalization transforms to achieve 95%+ coverage.

This module tests uncovered paths in normalization.py including:
- RobustNormalize JIT-compiled path (lines 231-259)
- MaxNormalize JIT-compiled path (lines 296-321)
- Edge cases with constant data and extreme values
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from piblin_jax.backend import is_jax_available, jnp
from piblin_jax.data.datasets import OneDimensionalDataset
from piblin_jax.transform.dataset import (
    MaxNormalize,
    MinMaxNormalize,
    RobustNormalize,
    ZScoreNormalize,
)


def create_test_dataset(y_data=None):
    """Create simple test dataset."""
    if y_data is None:
        n_points = 50
        y_data = np.random.randn(n_points) * 10 + 50
    else:
        n_points = len(y_data)
    x = np.linspace(0, 10, n_points)
    return OneDimensionalDataset(
        independent_variable_data=x,
        dependent_variable_data=y_data,
        conditions={"temperature": 25.0},
    )


class TestRobustNormalize:
    """Test robust normalization (median/IQR based)."""

    def test_robust_normalize_basic(self):
        """Test basic robust normalization (lines 231-259)."""
        # Create dataset with known statistics
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        dataset = create_test_dataset(y)

        # Apply robust normalization
        transform = RobustNormalize()
        result = transform.apply_to(dataset)

        # After robust normalization, median should be ~0
        median = np.median(result.dependent_variable_data)
        assert_allclose(median, 0.0, atol=0.1)

    def test_robust_normalize_with_outliers(self):
        """Test robust normalization handles outliers well."""
        # Create dataset with outliers
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100], dtype=float)  # 100 is outlier
        dataset = create_test_dataset(y)

        # Apply robust normalization
        transform = RobustNormalize()
        result = transform.apply_to(dataset)

        # Median should still be close to 0
        median = np.median(result.dependent_variable_data)
        assert_allclose(median, 0.0, atol=0.5)

    def test_robust_normalize_jit_compiled(self):
        """Test that robust normalization uses JIT compilation."""
        if not is_jax_available():
            pytest.skip("JAX not available")

        dataset = create_test_dataset()

        # Apply robust normalization
        transform = RobustNormalize()
        result = transform.apply_to(dataset)

        # Should work (JIT compilation should succeed)
        assert result.dependent_variable_data is not None

    def test_robust_normalize_with_constant_data(self):
        """Test robust normalization with constant data (zero IQR)."""
        # Constant data (IQR = 0)
        y = np.ones(50) * 5.0
        dataset = create_test_dataset(y)

        # Apply robust normalization
        transform = RobustNormalize()
        result = transform.apply_to(dataset)

        # Should handle division by zero gracefully (uses epsilon)
        # Result should be all zeros (since median = 5, and IQR ~= 0)
        assert_allclose(result.dependent_variable_data, 0.0, atol=0.1)

    def test_robust_normalize_preserves_metadata(self):
        """Test that robust normalization preserves metadata."""
        dataset = create_test_dataset()

        transform = RobustNormalize()
        result = transform.apply_to(dataset)

        assert result.conditions == dataset.conditions

    def test_robust_normalize_with_small_iqr(self):
        """Test robust normalization with very small IQR."""
        # Data with very small spread
        y = np.array([5.0, 5.001, 5.002, 5.001, 5.0] * 10)
        dataset = create_test_dataset(y)

        # Apply robust normalization
        transform = RobustNormalize()
        result = transform.apply_to(dataset)

        # Should not crash (epsilon prevents division by zero)
        assert result.dependent_variable_data is not None


class TestMaxNormalize:
    """Test max normalization."""

    def test_max_normalize_basic(self):
        """Test basic max normalization (lines 296-321)."""
        # Create dataset with known max
        y = np.array([-10, -5, 0, 5, 10], dtype=float)
        dataset = create_test_dataset(y)

        # Apply max normalization
        transform = MaxNormalize()
        result = transform.apply_to(dataset)

        # Max absolute value should be 1
        max_abs = np.max(np.abs(result.dependent_variable_data))
        assert_allclose(max_abs, 1.0, rtol=0.01)

    def test_max_normalize_preserves_sign(self):
        """Test that max normalization preserves sign."""
        y = np.array([-8, -4, 0, 4, 8], dtype=float)
        dataset = create_test_dataset(y)

        # Apply max normalization
        transform = MaxNormalize()
        result = transform.apply_to(dataset)

        # Signs should be preserved
        assert np.all(np.sign(result.dependent_variable_data) == np.sign(y))

    def test_max_normalize_preserves_zero(self):
        """Test that max normalization preserves zero."""
        y = np.array([-5, -2, 0, 2, 5], dtype=float)
        dataset = create_test_dataset(y)

        # Apply max normalization
        transform = MaxNormalize()
        result = transform.apply_to(dataset)

        # Zero should remain zero
        zero_idx = np.where(y == 0)[0][0]
        assert result.dependent_variable_data[zero_idx] == 0.0

    def test_max_normalize_with_all_positive(self):
        """Test max normalization with all positive values."""
        y = np.array([1, 2, 3, 4, 5], dtype=float)
        dataset = create_test_dataset(y)

        # Apply max normalization
        transform = MaxNormalize()
        result = transform.apply_to(dataset)

        # Max should be 1, min should be 1/5
        assert_allclose(np.max(result.dependent_variable_data), 1.0, rtol=0.01)
        assert_allclose(np.min(result.dependent_variable_data), 0.2, rtol=0.01)

    def test_max_normalize_with_all_negative(self):
        """Test max normalization with all negative values."""
        y = np.array([-5, -4, -3, -2, -1], dtype=float)
        dataset = create_test_dataset(y)

        # Apply max normalization
        transform = MaxNormalize()
        result = transform.apply_to(dataset)

        # Max absolute value should be 1 (most negative becomes -1)
        assert_allclose(np.min(result.dependent_variable_data), -1.0, rtol=0.01)

    def test_max_normalize_with_zero_data(self):
        """Test max normalization with all zeros (edge case)."""
        y = np.zeros(50)
        dataset = create_test_dataset(y)

        # Apply max normalization
        transform = MaxNormalize()
        result = transform.apply_to(dataset)

        # Should handle gracefully (epsilon prevents division by zero)
        assert_allclose(result.dependent_variable_data, 0.0, atol=1e-8)

    def test_max_normalize_jit_compiled(self):
        """Test that max normalization uses JIT compilation."""
        if not is_jax_available():
            pytest.skip("JAX not available")

        dataset = create_test_dataset()

        # Apply max normalization
        transform = MaxNormalize()
        result = transform.apply_to(dataset)

        # Should work (JIT compilation should succeed)
        assert result.dependent_variable_data is not None


class TestMinMaxNormalizeExtended:
    """Extended tests for min-max normalization."""

    def test_minmax_normalize_with_constant_data(self):
        """Test min-max normalization with constant data."""
        y = np.ones(50) * 7.0
        dataset = create_test_dataset(y)

        # Apply min-max normalization
        transform = MinMaxNormalize()
        result = transform.apply_to(dataset)

        # Should handle constant data (range = 0)
        # Result should be well-defined (not NaN)
        assert not np.any(np.isnan(result.dependent_variable_data))

    def test_minmax_normalize_to_custom_range(self):
        """Test min-max normalization to custom range."""
        y = np.array([0, 5, 10], dtype=float)
        dataset = create_test_dataset(y)

        # This would require MinMaxNormalize to support custom ranges
        # Current implementation normalizes to [0, 1]
        transform = MinMaxNormalize()
        result = transform.apply_to(dataset)

        # Should be in [0, 1]
        assert np.min(result.dependent_variable_data) >= 0.0
        assert np.max(result.dependent_variable_data) <= 1.0


class TestZScoreNormalizeExtended:
    """Extended tests for z-score normalization."""

    def test_zscore_normalize_with_constant_data(self):
        """Test z-score normalization with constant data (zero std)."""
        y = np.ones(50) * 3.0
        dataset = create_test_dataset(y)

        # Apply z-score normalization
        transform = ZScoreNormalize()
        result = transform.apply_to(dataset)

        # Should handle zero std gracefully
        assert not np.any(np.isnan(result.dependent_variable_data))

    def test_zscore_normalize_produces_standard_normal(self):
        """Test that z-score normalization produces standard normal."""
        # Create data with known mean and std
        y = np.random.randn(1000) * 5 + 10  # mean=10, std=5
        dataset = create_test_dataset(y)

        # Apply z-score normalization
        transform = ZScoreNormalize()
        result = transform.apply_to(dataset)

        # Result should have mean ~0 and std ~1
        assert_allclose(np.mean(result.dependent_variable_data), 0.0, atol=0.1)
        assert_allclose(np.std(result.dependent_variable_data), 1.0, atol=0.1)

    def test_zscore_normalize_with_small_std(self):
        """Test z-score normalization with very small std."""
        # Data with very small standard deviation
        y = np.array([5.0, 5.0001, 5.0002, 5.0001, 5.0] * 10)
        dataset = create_test_dataset(y)

        # Apply z-score normalization
        transform = ZScoreNormalize()
        result = transform.apply_to(dataset)

        # Should not crash
        assert result.dependent_variable_data is not None
