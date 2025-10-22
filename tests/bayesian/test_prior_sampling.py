"""
Tests for prior sampling (y=None scenarios) in Bayesian models (Task Group 10).

This module tests prior sampling for all rheological models to achieve
complete coverage of the sigma_scale conditional branches.
"""

import pytest

# Skip all tests in this module if JAX is not available
pytest.importorskip("jax", reason="JAX required for bayesian tests")

import numpy as np
import numpyro
from jax import random

from quantiq.bayesian.models import (
    ArrheniusModel,
    CarreauYasudaModel,
    CrossModel,
    PowerLawModel,
)


class TestPriorSampling:
    """Test prior sampling (y=None) for all models."""

    def test_power_law_model_prior_sampling(self):
        """Test PowerLawModel.model() with y=None (prior sampling)."""
        # This tests the model() method directly with y=None
        # to trigger the else branch in sigma_scale logic

        model_obj = PowerLawModel(n_samples=100, n_warmup=50, n_chains=1)

        # Test the model method directly with y=None
        shear_rate = np.logspace(-1, 2, 20)

        # Sample from prior by calling model with y=None
        from numpyro.infer import Predictive

        rng_key = random.PRNGKey(0)
        predictive = Predictive(model_obj.model, num_samples=50)
        prior_samples = predictive(rng_key, x=shear_rate, y=None)

        # Should have samples from priors
        assert "K" in prior_samples
        assert "n" in prior_samples
        assert "sigma" in prior_samples

        # Check shapes
        assert prior_samples["K"].shape == (50,)
        assert prior_samples["n"].shape == (50,)
        assert prior_samples["sigma"].shape == (50,)

    def test_arrhenius_model_prior_sampling(self):
        """Test ArrheniusModel.model() with y=None (prior sampling)."""
        model_obj = ArrheniusModel(n_samples=100, n_warmup=50, n_chains=1)

        # Test the model method directly with y=None
        temperature = np.linspace(250, 400, 20)

        # Sample from prior by calling model with y=None
        from numpyro.infer import Predictive

        rng_key = random.PRNGKey(0)
        predictive = Predictive(model_obj.model, num_samples=50)
        prior_samples = predictive(rng_key, x=temperature, y=None)

        # Should have samples from priors
        assert "A" in prior_samples
        assert "Ea" in prior_samples
        assert "sigma" in prior_samples

        # Check shapes
        assert prior_samples["A"].shape == (50,)
        assert prior_samples["Ea"].shape == (50,)
        assert prior_samples["sigma"].shape == (50,)

    def test_cross_model_prior_sampling(self):
        """Test CrossModel.model() with y=None (prior sampling)."""
        model_obj = CrossModel(n_samples=100, n_warmup=50, n_chains=1)

        # Test the model method directly with y=None
        shear_rate = np.logspace(-2, 3, 20)

        # Sample from prior by calling model with y=None
        from numpyro.infer import Predictive

        rng_key = random.PRNGKey(0)
        predictive = Predictive(model_obj.model, num_samples=50)
        prior_samples = predictive(rng_key, x=shear_rate, y=None)

        # Should have samples from priors
        assert "eta0" in prior_samples
        assert "eta_inf" in prior_samples
        assert "lambda_" in prior_samples
        assert "m" in prior_samples
        assert "sigma" in prior_samples

        # Check shapes
        assert prior_samples["eta0"].shape == (50,)
        assert prior_samples["eta_inf"].shape == (50,)
        assert prior_samples["lambda_"].shape == (50,)
        assert prior_samples["m"].shape == (50,)
        assert prior_samples["sigma"].shape == (50,)

    def test_carreau_yasuda_model_prior_sampling(self):
        """Test CarreauYasudaModel.model() with y=None (prior sampling)."""
        model_obj = CarreauYasudaModel(n_samples=100, n_warmup=50, n_chains=1)

        # Test the model method directly with y=None
        shear_rate = np.logspace(-2, 3, 20)

        # Sample from prior by calling model with y=None
        from numpyro.infer import Predictive

        rng_key = random.PRNGKey(0)
        predictive = Predictive(model_obj.model, num_samples=50)
        prior_samples = predictive(rng_key, x=shear_rate, y=None)

        # Should have samples from priors
        assert "eta0" in prior_samples
        assert "eta_inf" in prior_samples
        assert "lambda_" in prior_samples
        assert "a" in prior_samples
        assert "n" in prior_samples
        assert "sigma" in prior_samples

        # Check shapes
        assert prior_samples["eta0"].shape == (50,)
        assert prior_samples["eta_inf"].shape == (50,)
        assert prior_samples["lambda_"].shape == (50,)
        assert prior_samples["a"].shape == (50,)
        assert prior_samples["n"].shape == (50,)
        assert prior_samples["sigma"].shape == (50,)

    def test_prior_predictive_sampling(self):
        """Test prior predictive sampling for PowerLawModel."""
        model_obj = PowerLawModel(n_samples=100, n_warmup=50, n_chains=1)

        shear_rate = np.logspace(-1, 2, 20)

        # Sample from prior predictive distribution
        from numpyro.infer import Predictive

        rng_key = random.PRNGKey(0)
        predictive = Predictive(model_obj.model, num_samples=100)
        prior_predictive = predictive(rng_key, x=shear_rate, y=None)

        # Should have observations from prior predictive
        assert "obs" in prior_predictive
        assert prior_predictive["obs"].shape == (100, 20)

        # All samples should be finite
        assert np.all(np.isfinite(prior_predictive["obs"]))
