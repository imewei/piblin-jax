"""
Tests for NLSQ integration.

This module tests Task Group 15:
- NLSQ wrapper functionality
- Scipy fallback
- BayesianModel integration with use_nlsq_init
"""

import numpy as np
import pytest

from piblin_jax.backend import is_jax_available
from piblin_jax.fitting.nlsq import estimate_initial_parameters, fit_curve


def linear_model(x, a, b):
    """Simple linear model: y = a*x + b"""
    return a * x + b


def quadratic_model(x, a, b, c):
    """Quadratic model: y = a*x^2 + b*x + c"""
    return a * x**2 + b * x + c


def exponential_model(x, a, b, c):
    """Exponential model: y = a * exp(b*x) + c"""
    return a * np.exp(b * x) + c


class TestFitCurve:
    """Test fit_curve function."""

    def test_fit_linear_model(self):
        """Test fitting a simple linear model."""
        # Generate data
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y_true = 2.5 * x + 1.0
        y = y_true + 0.1 * np.random.randn(len(x))

        # Fit
        result = fit_curve(linear_model, x, y, p0=np.array([1.0, 0.0]))

        # Check that fit succeeded
        assert result["success"]
        assert result["method"] in ["nlsq", "scipy"]

        # Check parameter recovery
        params = result["params"]
        assert len(params) == 2
        assert np.abs(params[0] - 2.5) < 0.1  # Slope close to 2.5
        assert np.abs(params[1] - 1.0) < 0.2  # Intercept close to 1.0

    def test_fit_quadratic_model(self):
        """Test fitting a quadratic model."""
        np.random.seed(42)
        x = np.linspace(-5, 5, 50)
        y_true = 0.5 * x**2 + 1.0 * x - 2.0
        y = y_true + 0.2 * np.random.randn(len(x))

        # Fit
        result = fit_curve(quadratic_model, x, y, p0=np.array([1.0, 1.0, 0.0]))

        # Check success
        assert result["success"]
        assert result["params"] is not None
        assert len(result["params"]) == 3

    def test_fit_with_weights(self):
        """Test fitting with weighted data (sigma parameter)."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 1.0 + 0.5 * np.random.randn(len(x))

        # Create weights (inverse variance)
        sigma = 0.5 * np.ones_like(y)

        # Fit with weights
        result = fit_curve(
            linear_model, x, y, p0=np.array([1.0, 0.0]), sigma=sigma, absolute_sigma=True
        )

        assert result["success"]
        assert result["params"] is not None

    def test_fit_returns_residuals(self):
        """Test that fit returns residuals."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 1.0 + 0.1 * np.random.randn(len(x))

        result = fit_curve(linear_model, x, y, p0=np.array([1.0, 0.0]))

        assert "residuals" in result
        assert result["residuals"] is not None
        assert len(result["residuals"]) == len(y)

    def test_fit_returns_covariance(self):
        """Test that fit returns parameter covariance matrix."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 1.0 + 0.1 * np.random.randn(len(x))

        result = fit_curve(linear_model, x, y, p0=np.array([1.0, 0.0]))

        assert "covariance" in result
        # Covariance might be None for some methods, but should be present
        if result["covariance"] is not None:
            assert result["covariance"].shape == (2, 2)

    def test_scipy_fallback(self):
        """Test that scipy fallback works."""
        # This test ensures scipy fallback is functional
        # NLSQ may or may not be available, but scipy should always work
        np.random.seed(42)
        x = np.linspace(0, 10, 30)
        y = 2.0 * x + 1.0 + 0.1 * np.random.randn(len(x))

        result = fit_curve(linear_model, x, y, p0=np.array([1.0, 0.0]))

        # Should succeed regardless of whether NLSQ is available
        assert result["success"]
        assert result["method"] in ["nlsq", "scipy"]


class TestEstimateInitialParameters:
    """Test initial parameter estimation."""

    def test_estimate_single_parameter(self):
        """Test estimation for single parameter model."""

        def constant_model(x, a):
            return a * np.ones_like(x)

        x = np.linspace(0, 10, 50)
        y = 5.0 * np.ones_like(x)

        p0 = estimate_initial_parameters(constant_model, x, y)

        assert len(p0) == 1
        assert p0[0] > 0  # Should be positive (using mean)

    def test_estimate_two_parameters(self):
        """Test estimation for two-parameter model (linear)."""
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 3.0

        p0 = estimate_initial_parameters(linear_model, x, y)

        assert len(p0) == 2
        # Should provide reasonable initial guesses
        assert np.abs(p0[0] - 2.0) < 5.0  # Slope estimate
        assert np.abs(p0[1] - 3.0) < 10.0  # Intercept estimate

    def test_estimate_multiple_parameters(self):
        """Test estimation for multiple parameters."""
        x = np.linspace(0, 10, 50)
        y = 0.5 * x**2 + 1.0 * x - 2.0

        p0 = estimate_initial_parameters(quadratic_model, x, y)

        assert len(p0) == 3
        # Just check that estimates are provided
        assert all(np.isfinite(p0))

    def test_estimate_with_bounds(self):
        """Test estimation with parameter bounds."""
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 3.0

        bounds = (np.array([0.0, 0.0]), np.array([10.0, 10.0]))
        p0 = estimate_initial_parameters(linear_model, x, y, bounds=bounds)

        # All parameters should be within bounds
        assert np.all(p0 >= bounds[0])
        assert np.all(p0 <= bounds[1])


@pytest.mark.skipif(not is_jax_available(), reason="JAX required for BayesianModel tests")
class TestBayesianModelIntegration:
    """Test BayesianModel integration with NLSQ."""

    def test_bayesian_model_use_nlsq_init_parameter(self):
        """Test that BayesianModel accepts use_nlsq_init parameter."""
        import numpyro
        import numpyro.distributions as dist

        from piblin_jax.bayesian.base import BayesianModel

        class SimpleLinearModel(BayesianModel):
            def model(self, x, y=None):
                slope = numpyro.sample("slope", dist.Normal(0, 10))
                intercept = numpyro.sample("intercept", dist.Normal(0, 10))
                sigma = numpyro.sample("sigma", dist.HalfNormal(1))

                mu = slope * x + intercept
                with numpyro.plate("data", x.shape[0]):
                    numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)

            def predict(self, x, credible_interval=0.95):
                pass  # Not needed for this test

        # Create model
        model = SimpleLinearModel(n_samples=100, n_warmup=50, n_chains=1)

        # Generate data
        np.random.seed(42)
        x = np.linspace(0, 10, 30)
        y = 2.0 * x + 1.0 + 0.1 * np.random.randn(len(x))

        # Fit with use_nlsq_init (should not raise error)
        model.fit(x, y, use_nlsq_init=True)

        # Should have samples
        assert model.samples is not None

    def test_bayesian_model_without_nlsq_init(self):
        """Test that BayesianModel still works without use_nlsq_init."""
        import numpyro
        import numpyro.distributions as dist

        from piblin_jax.bayesian.base import BayesianModel

        class SimpleLinearModel(BayesianModel):
            def model(self, x, y=None):
                slope = numpyro.sample("slope", dist.Normal(0, 10))
                intercept = numpyro.sample("intercept", dist.Normal(0, 10))
                sigma = numpyro.sample("sigma", dist.HalfNormal(1))

                mu = slope * x + intercept
                with numpyro.plate("data", x.shape[0]):
                    numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)

            def predict(self, x, credible_interval=0.95):
                pass

        model = SimpleLinearModel(n_samples=100, n_warmup=50, n_chains=1)

        np.random.seed(42)
        x = np.linspace(0, 10, 30)
        y = 2.0 * x + 1.0 + 0.1 * np.random.randn(len(x))

        # Fit without use_nlsq_init (default behavior)
        model.fit(x, y)

        assert model.samples is not None
