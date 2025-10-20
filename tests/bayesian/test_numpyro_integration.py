"""
Tests for NumPyro integration foundation (Task Group 12).

This module tests:
- BayesianModel base class functionality
- MCMC sampling with NumPyro
- Credible interval calculation
- Uncertainty storage in datasets
- with_uncertainty() API
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from numpy.testing import assert_allclose

from quantiq.bayesian.base import BayesianModel
from quantiq.data.datasets import OneDimensionalDataset


class SimpleBayesianLinearModel(BayesianModel):
    """Simple linear regression model for testing."""

    def model(self, x, y=None):
        """
        Simple Bayesian linear regression: y = slope * x + intercept + noise.

        Parameters
        ----------
        x : array_like
            Independent variable
        y : array_like | None
            Dependent variable (observations, None for prediction)
        """
        # Priors
        slope = numpyro.sample("slope", dist.Normal(0.0, 10.0))
        intercept = numpyro.sample("intercept", dist.Normal(0.0, 10.0))
        sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))

        # Likelihood
        mu = slope * x + intercept
        with numpyro.plate("data", x.shape[0]):
            numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)

    def predict(self, x, credible_interval=0.95):
        """
        Predict with uncertainty using posterior samples.

        Parameters
        ----------
        x : array_like
            Points to predict at
        credible_interval : float
            Credible interval level (default: 0.95)

        Returns
        -------
        dict
            {'mean': ..., 'lower': ..., 'upper': ..., 'samples': ...}
        """
        if self._samples is None:
            raise RuntimeError("Model must be fit before prediction")

        x = jnp.asarray(x)

        # Generate predictions from posterior
        slope_samples = self._samples["slope"]
        intercept_samples = self._samples["intercept"]
        sigma_samples = self._samples["sigma"]

        # Predict for each posterior sample
        n_samples = slope_samples.shape[0]
        predictions = jnp.zeros((n_samples, x.shape[0]))

        for i in range(n_samples):
            mu = slope_samples[i] * x + intercept_samples[i]
            predictions = predictions.at[i, :].set(mu)

        # Compute statistics
        mean_pred = jnp.mean(predictions, axis=0)
        alpha = 1 - credible_interval
        lower = jnp.percentile(predictions, 100 * alpha / 2, axis=0)
        upper = jnp.percentile(predictions, 100 * (1 - alpha / 2), axis=0)

        return {
            "mean": np.array(mean_pred),
            "lower": np.array(lower),
            "upper": np.array(upper),
            "samples": np.array(predictions),
        }


class TestBayesianModelBaseClass:
    """Test BayesianModel abstract base class functionality."""

    def test_instantiation(self):
        """Test that BayesianModel can be instantiated with a concrete subclass."""
        model = SimpleBayesianLinearModel(n_samples=100, n_warmup=50, n_chains=1)
        assert model.n_samples == 100
        assert model.n_warmup == 50
        assert model.n_chains == 1
        assert model.random_seed == 0

    def test_custom_random_seed(self):
        """Test custom random seed initialization."""
        model = SimpleBayesianLinearModel(random_seed=42)
        assert model.random_seed == 42

    def test_samples_none_before_fit(self):
        """Test that samples are None before fitting."""
        model = SimpleBayesianLinearModel()
        assert model.samples is None


class TestMCMCSampling:
    """Test MCMC sampling with NumPyro."""

    @pytest.fixture
    def linear_data(self):
        """Generate simple linear data for testing."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        true_slope = 2.0
        true_intercept = 1.0
        noise = 0.1
        y = true_slope * x + true_intercept + noise * np.random.randn(len(x))
        return x, y, true_slope, true_intercept

    def test_fit_simple_model(self, linear_data):
        """Test fitting a simple Bayesian linear model."""
        x, y, _true_slope, _true_intercept = linear_data

        model = SimpleBayesianLinearModel(n_samples=500, n_warmup=250, n_chains=1)
        model.fit(x, y)

        # Check that samples were stored
        assert model.samples is not None
        assert "slope" in model.samples
        assert "intercept" in model.samples
        assert "sigma" in model.samples

        # Check sample shapes
        assert model.samples["slope"].shape == (500,)
        assert model.samples["intercept"].shape == (500,)
        assert model.samples["sigma"].shape == (500,)

    def test_posterior_recovery(self, linear_data):
        """Test that MCMC recovers true parameters (within reasonable tolerance)."""
        x, y, true_slope, true_intercept = linear_data

        model = SimpleBayesianLinearModel(n_samples=1000, n_warmup=500, n_chains=2)
        model.fit(x, y)

        # Check posterior means are close to true values
        slope_mean = np.mean(model.samples["slope"])
        intercept_mean = np.mean(model.samples["intercept"])

        assert_allclose(slope_mean, true_slope, atol=0.5)
        assert_allclose(intercept_mean, true_intercept, atol=0.5)


class TestCredibleIntervals:
    """Test credible interval calculation."""

    @pytest.fixture
    def fitted_model(self):
        """Fixture providing a fitted model."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 1.0 + 0.1 * np.random.randn(len(x))

        model = SimpleBayesianLinearModel(n_samples=1000, n_warmup=500, n_chains=1)
        model.fit(x, y)
        return model

    def test_get_credible_intervals_eti(self, fitted_model):
        """Test equal-tailed interval calculation."""
        lower, upper = fitted_model.get_credible_intervals("slope", level=0.95, method="eti")

        assert isinstance(lower, (float, np.floating))
        assert isinstance(upper, (float, np.floating))
        assert lower < upper

        # Check that interval contains reasonable values
        assert 1.0 < lower < 3.0
        assert 1.0 < upper < 3.0

    def test_get_credible_intervals_custom_level(self, fitted_model):
        """Test credible intervals with custom confidence level."""
        lower_95, upper_95 = fitted_model.get_credible_intervals("slope", level=0.95, method="eti")
        lower_68, upper_68 = fitted_model.get_credible_intervals("slope", level=0.68, method="eti")

        # 68% interval should be narrower than 95% interval
        assert (upper_68 - lower_68) < (upper_95 - lower_95)

    def test_get_credible_intervals_all_parameters(self, fitted_model):
        """Test credible intervals for all model parameters."""
        for param in ["slope", "intercept", "sigma"]:
            lower, upper = fitted_model.get_credible_intervals(param, level=0.95)
            assert lower < upper

    def test_get_credible_intervals_before_fit(self):
        """Test that credible intervals fail before fitting."""
        model = SimpleBayesianLinearModel()
        with pytest.raises(RuntimeError, match="Model must be fit first"):
            model.get_credible_intervals("slope")

    def test_get_credible_intervals_invalid_parameter(self, fitted_model):
        """Test error for invalid parameter name."""
        with pytest.raises(ValueError, match="Unknown parameter"):
            fitted_model.get_credible_intervals("nonexistent_param")


class TestUncertaintySamples:
    """Test uncertainty samples storage and prediction."""

    @pytest.fixture
    def fitted_model(self):
        """Fixture providing a fitted model."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 1.0 + 0.1 * np.random.randn(len(x))

        model = SimpleBayesianLinearModel(n_samples=500, n_warmup=250, n_chains=1)
        model.fit(x, y)
        return model

    def test_predict_with_uncertainty(self, fitted_model):
        """Test prediction with uncertainty quantification."""
        x_new = np.array([5.0, 7.5, 10.0])
        result = fitted_model.predict(x_new, credible_interval=0.95)

        assert "mean" in result
        assert "lower" in result
        assert "upper" in result
        assert "samples" in result

        # Check shapes
        assert result["mean"].shape == (3,)
        assert result["lower"].shape == (3,)
        assert result["upper"].shape == (3,)
        assert result["samples"].shape == (500, 3)

        # Check that uncertainty bounds are reasonable
        for i in range(3):
            assert result["lower"][i] < result["mean"][i] < result["upper"][i]

    def test_predict_before_fit(self):
        """Test that prediction fails before fitting."""
        model = SimpleBayesianLinearModel()
        with pytest.raises(RuntimeError, match="Model must be fit before prediction"):
            model.predict(np.array([1.0, 2.0, 3.0]))


class TestDatasetUncertaintyStorage:
    """Test uncertainty storage in Dataset classes."""

    def test_dataset_uncertainty_attributes(self):
        """Test that datasets have uncertainty attributes."""
        x = np.linspace(0, 10, 50)
        y = np.sin(x)
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        # Check uncertainty attributes exist
        assert hasattr(dataset, "_uncertainty_samples")
        assert hasattr(dataset, "_credible_intervals")
        assert hasattr(dataset, "_uncertainty_method")
        assert hasattr(dataset, "has_uncertainty")

    def test_dataset_no_uncertainty_initially(self):
        """Test that datasets have no uncertainty initially."""
        x = np.linspace(0, 10, 50)
        y = np.sin(x)
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        assert dataset.has_uncertainty is False
        assert dataset.uncertainty_samples is None
        assert dataset.credible_intervals is None


class TestWithUncertaintyAPI:
    """Test with_uncertainty() API for datasets."""

    @pytest.fixture
    def simple_dataset(self):
        """Simple dataset for testing."""
        np.random.seed(42)
        x = np.linspace(0, 10, 30)
        y = 2.0 * x + 1.0 + 0.2 * np.random.randn(len(x))
        return OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

    def test_with_uncertainty_bayesian_method(self, simple_dataset):
        """Test with_uncertainty() using Bayesian method."""
        # This is a simple smoke test - full implementation in 12.4
        dataset_with_unc = simple_dataset.with_uncertainty(
            n_samples=200, method="bayesian", keep_samples=False, level=0.95
        )

        # Should return a new dataset
        assert isinstance(dataset_with_unc, OneDimensionalDataset)
        assert dataset_with_unc.has_uncertainty is True
        assert dataset_with_unc._uncertainty_method == "bayesian"

    def test_with_uncertainty_keep_samples(self, simple_dataset):
        """Test with_uncertainty() with keep_samples=True."""
        dataset_with_unc = simple_dataset.with_uncertainty(
            n_samples=200, method="bayesian", keep_samples=True
        )

        assert dataset_with_unc.uncertainty_samples is not None
        assert "sigma" in dataset_with_unc.uncertainty_samples

    def test_with_uncertainty_no_keep_samples(self, simple_dataset):
        """Test with_uncertainty() with keep_samples=False."""
        dataset_with_unc = simple_dataset.with_uncertainty(
            n_samples=200, method="bayesian", keep_samples=False
        )

        # Samples should not be stored
        assert dataset_with_unc.uncertainty_samples is None
        # But credible intervals should be computed
        assert dataset_with_unc.credible_intervals is not None

    def test_with_uncertainty_unsupported_method(self, simple_dataset):
        """Test with_uncertainty() with unsupported method."""
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            simple_dataset.with_uncertainty(n_samples=100, method="invalid_method")
