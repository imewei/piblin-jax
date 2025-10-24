"""
Tests for uncertainty propagation through transforms and pipelines.

This module tests Task Group 14:
- Uncertainty propagation through single transforms
- Uncertainty propagation through pipelines
- Bootstrap method for uncertainty quantification
"""

import numpy as np
import pytest

from piblin_jax.data.datasets import OneDimensionalDataset
from piblin_jax.transform.base import DatasetTransform
from piblin_jax.transform.pipeline import LazyPipeline, Pipeline


class MultiplyTransform(DatasetTransform):
    """Simple transform that multiplies dependent variable by a factor."""

    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def _apply(self, dataset):
        """Multiply dependent variable by factor."""
        dataset._dependent_variable_data = dataset._dependent_variable_data * self.factor
        return dataset


class AddTransform(DatasetTransform):
    """Simple transform that adds a constant to dependent variable."""

    def __init__(self, constant):
        super().__init__()
        self.constant = constant

    def _apply(self, dataset):
        """Add constant to dependent variable."""
        dataset._dependent_variable_data = dataset._dependent_variable_data + self.constant
        return dataset


class TestUncertaintyPropagationSingleTransform:
    """Test uncertainty propagation through a single transform."""

    def test_transform_without_uncertainty_propagation(self):
        """Test that transform works without uncertainty propagation."""
        # Create dataset
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 1.0
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        # Apply transform
        transform = MultiplyTransform(2.0)
        result = transform.apply_to(dataset, propagate_uncertainty=False)

        # Check result
        np.testing.assert_allclose(result.dependent_variable_data, 4.0 * x + 2.0)
        assert not result.has_uncertainty

    def test_transform_with_uncertainty_propagation_flag(self):
        """Test that transform accepts propagate_uncertainty parameter."""
        # Create dataset with uncertainty
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 1.0
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        # Apply transform with propagate_uncertainty=True
        transform = MultiplyTransform(2.0)
        result = transform.apply_to(dataset, propagate_uncertainty=True)

        # Should work even without uncertainty (just applies transform normally)
        np.testing.assert_allclose(result.dependent_variable_data, 4.0 * x + 2.0)

    def test_transform_preserves_uncertainty_samples(self):
        """Test that transform preserves uncertainty samples."""
        # Create dataset with uncertainty using bootstrap
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 1.0 + 0.1 * np.random.randn(len(x))
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        # Add uncertainty with bootstrap
        dataset_with_unc = dataset.with_uncertainty(
            n_samples=100, method="bootstrap", keep_samples=True
        )

        # Apply transform
        transform = MultiplyTransform(2.0)
        result = transform.apply_to(dataset_with_unc, propagate_uncertainty=True)

        # Check that uncertainty is preserved
        assert result.has_uncertainty
        assert result.uncertainty_samples is not None


class TestUncertaintyPropagationPipeline:
    """Test uncertainty propagation through pipelines."""

    def test_pipeline_without_uncertainty_propagation(self):
        """Test pipeline works without uncertainty propagation."""
        # Create dataset
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 1.0
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        # Create pipeline
        pipeline = Pipeline([MultiplyTransform(2.0), AddTransform(5.0)])

        # Apply pipeline
        result = pipeline.apply_to(dataset, propagate_uncertainty=False)

        # Check result: (2x + 1) * 2 + 5 = 4x + 7
        np.testing.assert_allclose(result.dependent_variable_data, 4.0 * x + 7.0)

    def test_pipeline_with_uncertainty_propagation(self):
        """Test pipeline with uncertainty propagation parameter."""
        # Create dataset
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 1.0
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        # Create pipeline
        pipeline = Pipeline([MultiplyTransform(2.0), AddTransform(5.0)])

        # Apply with propagate_uncertainty=True
        result = pipeline.apply_to(dataset, propagate_uncertainty=True)

        # Should work (just applies transforms normally without uncertainty)
        np.testing.assert_allclose(result.dependent_variable_data, 4.0 * x + 7.0)

    def test_pipeline_preserves_uncertainty(self):
        """Test that pipeline preserves uncertainty information."""
        # Create dataset with uncertainty
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 1.0 + 0.1 * np.random.randn(len(x))
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        # Add uncertainty with bootstrap
        dataset_with_unc = dataset.with_uncertainty(
            n_samples=100, method="bootstrap", keep_samples=True
        )

        # Create pipeline
        pipeline = Pipeline([MultiplyTransform(2.0), AddTransform(5.0)])

        # Apply with uncertainty propagation
        result = pipeline.apply_to(dataset_with_unc, propagate_uncertainty=True)

        # Check that uncertainty is preserved
        assert result.has_uncertainty
        assert result.credible_intervals is not None

    def test_lazy_pipeline_with_uncertainty(self):
        """Test lazy pipeline with uncertainty propagation."""
        # Create dataset
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 1.0
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        # Create lazy pipeline
        pipeline = LazyPipeline([MultiplyTransform(2.0), AddTransform(5.0)])

        # Apply with propagate_uncertainty
        result = pipeline.apply_to(dataset, propagate_uncertainty=True)

        # Access result (triggers computation)
        y_result = result.dependent_variable_data

        # Check result
        np.testing.assert_allclose(y_result, 4.0 * x + 7.0)


class TestBootstrapMethod:
    """Test bootstrap method for uncertainty quantification."""

    def test_bootstrap_basic(self):
        """Test basic bootstrap uncertainty quantification."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 1.0 + 0.1 * np.random.randn(len(x))
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        # Add uncertainty with bootstrap
        dataset_with_unc = dataset.with_uncertainty(
            n_samples=100, method="bootstrap", keep_samples=False
        )

        # Check that uncertainty is added
        assert dataset_with_unc.has_uncertainty
        assert dataset_with_unc.credible_intervals is not None

    def test_bootstrap_with_samples(self):
        """Test bootstrap with keep_samples=True."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 1.0 + 0.1 * np.random.randn(len(x))
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        # Add uncertainty with bootstrap, keeping samples
        dataset_with_unc = dataset.with_uncertainty(
            n_samples=100, method="bootstrap", keep_samples=True
        )

        # Check that samples are stored
        assert dataset_with_unc.has_uncertainty
        assert dataset_with_unc.uncertainty_samples is not None
        assert "bootstrap_samples" in dataset_with_unc.uncertainty_samples

        # Check shape of bootstrap samples
        samples = dataset_with_unc.uncertainty_samples["bootstrap_samples"]
        assert samples.shape == (100, len(x))

    def test_bootstrap_credible_intervals(self):
        """Test that bootstrap produces reasonable credible intervals."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        # Add noise to data
        y = 2.0 * x + 1.0 + 0.5 * np.random.randn(len(x))
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        # Add uncertainty with bootstrap
        dataset_with_unc = dataset.with_uncertainty(n_samples=200, method="bootstrap", level=0.95)

        # Get credible intervals
        lower, upper = dataset_with_unc.credible_intervals

        # Check that intervals are arrays of correct shape
        assert lower.shape == y.shape
        assert upper.shape == y.shape

        # Check that lower < upper
        assert np.all(lower <= upper)

    def test_bootstrap_different_levels(self):
        """Test bootstrap with different credible interval levels."""
        np.random.seed(42)
        x = np.linspace(0, 10, 30)
        y = 2.0 * x + 1.0 + 0.5 * np.random.randn(len(x))
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        # 95% interval
        dataset_95 = dataset.with_uncertainty(n_samples=100, method="bootstrap", level=0.95)
        lower_95, upper_95 = dataset_95.credible_intervals

        # 68% interval (should be narrower)
        dataset_68 = dataset.with_uncertainty(n_samples=100, method="bootstrap", level=0.68)
        lower_68, upper_68 = dataset_68.credible_intervals

        # 68% interval should be narrower than 95% interval
        width_95 = np.mean(upper_95 - lower_95)
        width_68 = np.mean(upper_68 - lower_68)
        assert width_68 < width_95
