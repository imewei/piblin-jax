"""
Extended tests for BayesianModel base class (Task Group 10).

This module provides comprehensive coverage for:
- BayesianModel base class edge cases
- MCMC error handling and validation
- Credible interval computation methods (ETI and HPD)
- Summary statistics with edge cases
- Parameter validation and error conditions
- Prior sampling (y=None scenarios)
"""

import pytest

# Skip all tests in this module if JAX is not available
pytest.importorskip("jax", reason="JAX required for bayesian tests")

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpy.testing import assert_allclose

from quantiq.bayesian.base import BayesianModel


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


class TestBayesianModelEdgeCases:
    """Test BayesianModel base class edge cases and error handling."""

    @pytest.fixture
    def linear_data(self):
        """Generate simple linear data for testing."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        true_slope = 2.0
        true_intercept = 1.0
        noise = 0.1
        y = true_slope * x + true_intercept + noise * np.random.randn(len(x))
        return x, y

    def test_fit_with_use_nlsq_init_true(self, linear_data):
        """Test fit() with use_nlsq_init=True (experimental feature)."""
        x, y = linear_data

        # This should work without error even though use_nlsq_init is not implemented
        model = SimpleBayesianLinearModel(n_samples=100, n_warmup=50, n_chains=1)
        model.fit(x, y, use_nlsq_init=True)

        # Should still produce samples
        assert model.samples is not None
        assert "slope" in model.samples

    def test_fit_with_use_nlsq_init_false(self, linear_data):
        """Test fit() with use_nlsq_init=False (default behavior)."""
        x, y = linear_data

        model = SimpleBayesianLinearModel(n_samples=100, n_warmup=50, n_chains=1)
        model.fit(x, y, use_nlsq_init=False)

        assert model.samples is not None
        assert "slope" in model.samples

    def test_predict_before_fit_raises_error(self):
        """Test that predict() raises RuntimeError before fitting."""
        model = SimpleBayesianLinearModel()

        with pytest.raises(RuntimeError, match="Model must be fit before prediction"):
            model.predict(np.array([1.0, 2.0, 3.0]))

    def test_samples_property_before_fit(self):
        """Test that samples property returns None before fitting."""
        model = SimpleBayesianLinearModel()
        assert model.samples is None

    def test_samples_property_after_fit(self, linear_data):
        """Test that samples property returns dict after fitting."""
        x, y = linear_data
        model = SimpleBayesianLinearModel(n_samples=100, n_warmup=50, n_chains=1)
        model.fit(x, y)

        samples = model.samples
        assert samples is not None
        assert isinstance(samples, dict)
        assert "slope" in samples
        assert "intercept" in samples
        assert "sigma" in samples


class TestCredibleIntervalsExtended:
    """Extended tests for credible interval computation."""

    @pytest.fixture
    def fitted_model(self):
        """Fixture providing a fitted model."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 1.0 + 0.1 * np.random.randn(len(x))

        model = SimpleBayesianLinearModel(n_samples=1000, n_warmup=500, n_chains=1)
        model.fit(x, y)
        return model

    def test_get_credible_intervals_hpd_method(self, fitted_model):
        """Test HPD (highest posterior density) method for credible intervals."""
        # HPD method should work (even though it's simplified implementation)
        lower, upper = fitted_model.get_credible_intervals("slope", level=0.95, method="hpd")

        assert isinstance(lower, (float, np.floating))
        assert isinstance(upper, (float, np.floating))
        assert lower < upper

    def test_get_credible_intervals_eti_method(self, fitted_model):
        """Test ETI (equal-tailed interval) method for credible intervals."""
        lower, upper = fitted_model.get_credible_intervals("slope", level=0.95, method="eti")

        assert isinstance(lower, (float, np.floating))
        assert isinstance(upper, (float, np.floating))
        assert lower < upper

    def test_get_credible_intervals_invalid_method(self, fitted_model):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            fitted_model.get_credible_intervals("slope", level=0.95, method="invalid")

    def test_get_credible_intervals_before_fit(self):
        """Test that get_credible_intervals raises RuntimeError before fitting."""
        model = SimpleBayesianLinearModel()

        with pytest.raises(RuntimeError, match="Model must be fit first"):
            model.get_credible_intervals("slope")

    def test_get_credible_intervals_invalid_parameter(self, fitted_model):
        """Test that invalid parameter name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown parameter"):
            fitted_model.get_credible_intervals("nonexistent_param")

    def test_get_credible_intervals_different_levels(self, fitted_model):
        """Test credible intervals at different confidence levels."""
        # Test 50%, 68%, 95%, and 99% intervals
        levels = [0.50, 0.68, 0.95, 0.99]
        intervals = []

        for level in levels:
            lower, upper = fitted_model.get_credible_intervals("slope", level=level)
            intervals.append((lower, upper))

        # Higher confidence levels should have wider intervals
        for i in range(len(intervals) - 1):
            width_i = intervals[i][1] - intervals[i][0]
            width_i_plus_1 = intervals[i + 1][1] - intervals[i + 1][0]
            assert width_i < width_i_plus_1

    def test_get_credible_intervals_all_parameters(self, fitted_model):
        """Test credible intervals for all model parameters."""
        for param in ["slope", "intercept", "sigma"]:
            lower, upper = fitted_model.get_credible_intervals(param, level=0.95)
            assert lower < upper
            assert isinstance(lower, (float, np.floating))
            assert isinstance(upper, (float, np.floating))


class TestSummaryStatistics:
    """Test summary statistics computation."""

    @pytest.fixture
    def fitted_model(self):
        """Fixture providing a fitted model."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 1.0 + 0.1 * np.random.randn(len(x))

        model = SimpleBayesianLinearModel(n_samples=1000, n_warmup=500, n_chains=1)
        model.fit(x, y)
        return model

    def test_summary_before_fit(self):
        """Test that summary() raises RuntimeError before fitting."""
        model = SimpleBayesianLinearModel()

        with pytest.raises(RuntimeError, match="Model must be fit first"):
            model.summary()

    def test_summary_structure(self, fitted_model):
        """Test that summary returns correct structure."""
        summary = fitted_model.summary()

        assert isinstance(summary, dict)
        assert "slope" in summary
        assert "intercept" in summary
        assert "sigma" in summary

        # Check that each parameter has required statistics
        for _param_name, stats in summary.items():
            assert "mean" in stats
            assert "std" in stats
            assert "q_2.5" in stats
            assert "q_50" in stats
            assert "q_97.5" in stats

    def test_summary_value_ranges(self, fitted_model):
        """Test that summary statistics have reasonable values."""
        summary = fitted_model.summary()

        # For slope (true value is 2.0)
        slope_stats = summary["slope"]
        assert 1.5 < slope_stats["mean"] < 2.5
        assert slope_stats["std"] > 0
        assert slope_stats["q_2.5"] < slope_stats["q_50"] < slope_stats["q_97.5"]

        # For intercept (true value is 1.0)
        intercept_stats = summary["intercept"]
        assert 0.5 < intercept_stats["mean"] < 1.5
        assert intercept_stats["std"] > 0

        # For sigma (should be small, around 0.1)
        sigma_stats = summary["sigma"]
        assert sigma_stats["mean"] > 0
        assert sigma_stats["std"] > 0

    def test_summary_quantile_ordering(self, fitted_model):
        """Test that quantiles are properly ordered."""
        summary = fitted_model.summary()

        for _param_name, stats in summary.items():
            # q_2.5 < q_50 < q_97.5
            assert stats["q_2.5"] < stats["q_50"]
            assert stats["q_50"] < stats["q_97.5"]

    def test_summary_all_values_finite(self, fitted_model):
        """Test that all summary values are finite."""
        summary = fitted_model.summary()

        for _param_name, stats in summary.items():
            for _stat_name, value in stats.items():
                assert np.isfinite(value)

    def test_summary_return_types(self, fitted_model):
        """Test that summary returns proper Python float types."""
        summary = fitted_model.summary()

        for _param_name, stats in summary.items():
            for _stat_name, value in stats.items():
                # Should be Python float, not numpy float
                assert isinstance(value, float)


class TestMCMCConfiguration:
    """Test MCMC configuration and initialization."""

    def test_default_configuration(self):
        """Test default MCMC configuration."""
        model = SimpleBayesianLinearModel()

        assert model.n_samples == 1000
        assert model.n_warmup == 500
        assert model.n_chains == 2
        assert model.random_seed == 0

    def test_custom_configuration(self):
        """Test custom MCMC configuration."""
        model = SimpleBayesianLinearModel(n_samples=2000, n_warmup=1000, n_chains=4, random_seed=42)

        assert model.n_samples == 2000
        assert model.n_warmup == 1000
        assert model.n_chains == 4
        assert model.random_seed == 42

    def test_internal_state_before_fit(self):
        """Test that internal state is properly initialized."""
        model = SimpleBayesianLinearModel()

        assert model._mcmc is None
        assert model._samples is None

    def test_internal_state_after_fit(self):
        """Test that internal state is properly set after fitting."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 1.0 + 0.1 * np.random.randn(len(x))

        model = SimpleBayesianLinearModel(n_samples=100, n_warmup=50, n_chains=1)
        model.fit(x, y)

        assert model._mcmc is not None
        assert model._samples is not None


class TestModelAbstraction:
    """Test abstract base class behavior."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that BayesianModel cannot be instantiated directly."""
        # BayesianModel is abstract, so we can't instantiate it
        # This test ensures subclasses implement required methods
        with pytest.raises(TypeError):
            BayesianModel()

    def test_subclass_must_implement_model(self):
        """Test that subclasses must implement model() method."""

        class IncompleteBayesianModel(BayesianModel):
            # Missing model() implementation
            def predict(self, x, credible_interval=0.95):
                return {}

        # Should raise TypeError when trying to instantiate
        with pytest.raises(TypeError):
            IncompleteBayesianModel()

    def test_subclass_must_implement_predict(self):
        """Test that subclasses must implement predict() method."""

        class IncompleteBayesianModel(BayesianModel):
            # Missing predict() implementation
            def model(self, x, y=None):
                pass

        # Should raise TypeError when trying to instantiate
        with pytest.raises(TypeError):
            IncompleteBayesianModel()


class TestFitReturnValue:
    """Test that fit() returns self for method chaining."""

    def test_fit_returns_self(self):
        """Test that fit() returns self for method chaining."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 1.0 + 0.1 * np.random.randn(len(x))

        model = SimpleBayesianLinearModel(n_samples=100, n_warmup=50, n_chains=1)
        result = model.fit(x, y)

        # fit() should return self
        assert result is model

    def test_method_chaining(self):
        """Test that fit() enables method chaining."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 1.0 + 0.1 * np.random.randn(len(x))

        model = SimpleBayesianLinearModel(n_samples=100, n_warmup=50, n_chains=1)

        # Should be able to chain fit() and other methods
        x_new = np.array([5.0, 7.5, 10.0])
        predictions = model.fit(x, y).predict(x_new)

        assert "mean" in predictions
        assert "lower" in predictions
        assert "upper" in predictions
