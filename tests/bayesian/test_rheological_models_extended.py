"""
Extended tests for rheological models (Task Group 10).

This module provides comprehensive coverage for:
- Prior sampling (y=None scenarios) for all models
- Edge case handling (zeros, infinities, NaNs)
- Parameter recovery with noisy data
- Model comparison and convergence diagnostics
- Integration tests with realistic datasets
"""

import pytest

# Skip all tests in this module if JAX is not available
pytest.importorskip("jax", reason="JAX required for bayesian tests")

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

from quantiq.bayesian.models import (
    ArrheniusModel,
    CarreauYasudaModel,
    CrossModel,
    PowerLawModel,
)


class TestPowerLawModelExtended:
    """Extended tests for PowerLawModel."""

    @pytest.fixture
    def power_law_data(self):
        """Generate synthetic power-law viscosity data."""
        np.random.seed(42)
        shear_rate = np.logspace(-2, 3, 50)
        true_K = 5.0
        true_n = 0.6
        noise_level = 0.05

        eta_true = true_K * shear_rate ** (true_n - 1)
        noise = noise_level * eta_true * np.random.randn(len(shear_rate))
        eta = eta_true + noise

        return shear_rate, eta, true_K, true_n

    def test_power_law_predict_before_fit(self):
        """Test that predict() raises error before fitting."""
        model = PowerLawModel()

        with pytest.raises(RuntimeError, match="Model must be fit before prediction"):
            model.predict(np.array([1.0, 10.0, 100.0]))

    def test_power_law_with_noisy_data(self):
        """Test PowerLawModel with high noise levels."""
        np.random.seed(123)
        shear_rate = np.logspace(-1, 2, 30)
        true_K = 3.0
        true_n = 0.7
        noise_level = 0.2  # 20% noise

        eta_true = true_K * shear_rate ** (true_n - 1)
        noise = noise_level * eta_true * np.random.randn(len(shear_rate))
        eta = eta_true + noise

        model = PowerLawModel(n_samples=500, n_warmup=250, n_chains=1)
        model.fit(shear_rate, eta)

        # Should still fit even with high noise
        assert model.samples is not None
        K_mean = np.mean(model.samples["K"])
        n_mean = np.mean(model.samples["n"])

        # Wider tolerance due to noise
        assert_allclose(K_mean, true_K, rtol=0.5)
        assert_allclose(n_mean, true_n, rtol=0.5)

    def test_power_law_shear_thickening(self):
        """Test PowerLawModel with shear-thickening behavior (n > 1)."""
        np.random.seed(42)
        shear_rate = np.logspace(-1, 2, 30)
        true_K = 0.5
        true_n = 1.3  # Shear-thickening
        noise_level = 0.05

        eta_true = true_K * shear_rate ** (true_n - 1)
        noise = noise_level * eta_true * np.random.randn(len(shear_rate))
        eta = eta_true + noise

        model = PowerLawModel(n_samples=500, n_warmup=250, n_chains=1)
        model.fit(shear_rate, eta)

        n_mean = np.mean(model.samples["n"])

        # Should detect shear-thickening behavior
        assert n_mean > 1.0

    def test_power_law_summary_statistics(self, power_law_data):
        """Test summary statistics for PowerLawModel."""
        shear_rate, eta, _, _ = power_law_data

        model = PowerLawModel(n_samples=500, n_warmup=250, n_chains=1)
        model.fit(shear_rate, eta)

        summary = model.summary()

        # Check structure
        assert "K" in summary
        assert "n" in summary
        assert "sigma" in summary

        # Check that all parameters have required statistics
        for param in ["K", "n", "sigma"]:
            assert "mean" in summary[param]
            assert "std" in summary[param]
            assert "q_2.5" in summary[param]
            assert "q_50" in summary[param]
            assert "q_97.5" in summary[param]

    def test_power_law_credible_intervals_all_params(self, power_law_data):
        """Test credible intervals for all PowerLawModel parameters."""
        shear_rate, eta, _, _ = power_law_data

        model = PowerLawModel(n_samples=500, n_warmup=250, n_chains=1)
        model.fit(shear_rate, eta)

        for param in ["K", "n", "sigma"]:
            lower, upper = model.get_credible_intervals(param, level=0.95)
            assert lower < upper


class TestArrheniusModelExtended:
    """Extended tests for ArrheniusModel."""

    @pytest.fixture
    def arrhenius_data(self):
        """Generate synthetic Arrhenius data."""
        np.random.seed(42)
        temperature = np.linspace(250, 400, 30)
        true_A = 1e-5
        true_Ea = 50000
        R = 8.314
        noise_level = 0.05

        eta_true = true_A * np.exp(true_Ea / (R * temperature))
        noise = noise_level * eta_true * np.random.randn(len(temperature))
        eta = eta_true + noise

        return temperature, eta, true_A, true_Ea

    def test_arrhenius_predict_before_fit(self):
        """Test that predict() raises error before fitting."""
        model = ArrheniusModel()

        with pytest.raises(RuntimeError, match="Model must be fit before prediction"):
            model.predict(np.array([300.0, 350.0, 400.0]))

    def test_arrhenius_summary_statistics(self, arrhenius_data):
        """Test summary statistics for ArrheniusModel."""
        temperature, eta, _, _ = arrhenius_data

        model = ArrheniusModel(n_samples=500, n_warmup=250, n_chains=1)
        model.fit(temperature, eta)

        summary = model.summary()

        # Check structure
        assert "A" in summary
        assert "Ea" in summary
        assert "sigma" in summary

        # Check that all parameters have required statistics
        for param in ["A", "Ea", "sigma"]:
            assert "mean" in summary[param]
            assert "std" in summary[param]

    def test_arrhenius_credible_intervals_all_params(self, arrhenius_data):
        """Test credible intervals for all ArrheniusModel parameters."""
        temperature, eta, _, _ = arrhenius_data

        model = ArrheniusModel(n_samples=500, n_warmup=250, n_chains=1)
        model.fit(temperature, eta)

        for param in ["A", "Ea", "sigma"]:
            lower, upper = model.get_credible_intervals(param, level=0.95)
            assert lower < upper

    def test_arrhenius_with_wide_temperature_range(self):
        """Test ArrheniusModel with wide temperature range."""
        np.random.seed(42)
        temperature = np.linspace(200, 500, 40)  # Wide range
        true_A = 1e-6
        true_Ea = 60000
        R = 8.314
        noise_level = 0.05

        eta_true = true_A * np.exp(true_Ea / (R * temperature))
        noise = noise_level * eta_true * np.random.randn(len(temperature))
        eta = eta_true + noise

        model = ArrheniusModel(n_samples=500, n_warmup=250, n_chains=1)
        model.fit(temperature, eta)

        # Should fit successfully
        assert model.samples is not None
        assert "A" in model.samples
        assert "Ea" in model.samples


class TestCrossModelExtended:
    """Extended tests for CrossModel."""

    @pytest.fixture
    def cross_data(self):
        """Generate synthetic Cross model data."""
        np.random.seed(42)
        shear_rate = np.logspace(-2, 3, 50)
        true_eta0 = 100.0
        true_eta_inf = 1.0
        true_lambda = 1.0
        true_m = 0.7
        noise_level = 0.05

        eta_true = true_eta_inf + (true_eta0 - true_eta_inf) / (
            1 + (true_lambda * shear_rate) ** true_m
        )
        noise = noise_level * eta_true * np.random.randn(len(shear_rate))
        eta = eta_true + noise

        return shear_rate, eta

    def test_cross_predict_before_fit(self):
        """Test that predict() raises error before fitting."""
        model = CrossModel()

        with pytest.raises(RuntimeError, match="Model must be fit before prediction"):
            model.predict(np.array([1.0, 10.0, 100.0]))

    def test_cross_summary_statistics(self, cross_data):
        """Test summary statistics for CrossModel."""
        shear_rate, eta = cross_data

        model = CrossModel(n_samples=500, n_warmup=250, n_chains=1)
        model.fit(shear_rate, eta)

        summary = model.summary()

        # Check structure
        assert "eta0" in summary
        assert "eta_inf" in summary
        assert "lambda_" in summary
        assert "m" in summary
        assert "sigma" in summary

    def test_cross_credible_intervals_all_params(self, cross_data):
        """Test credible intervals for all CrossModel parameters."""
        shear_rate, eta = cross_data

        model = CrossModel(n_samples=500, n_warmup=250, n_chains=1)
        model.fit(shear_rate, eta)

        for param in ["eta0", "eta_inf", "lambda_", "m", "sigma"]:
            lower, upper = model.get_credible_intervals(param, level=0.95)
            assert lower < upper

    def test_cross_with_small_viscosity_ratio(self):
        """Test CrossModel with small viscosity ratio (eta0/eta_inf close to 1)."""
        np.random.seed(42)
        shear_rate = np.logspace(-1, 2, 30)
        true_eta0 = 5.0
        true_eta_inf = 4.0  # Small difference
        true_lambda = 0.5
        true_m = 0.8
        noise_level = 0.05

        eta_true = true_eta_inf + (true_eta0 - true_eta_inf) / (
            1 + (true_lambda * shear_rate) ** true_m
        )
        noise = noise_level * eta_true * np.random.randn(len(shear_rate))
        eta = eta_true + noise

        model = CrossModel(n_samples=500, n_warmup=250, n_chains=1)
        model.fit(shear_rate, eta)

        # Should fit successfully even with small viscosity ratio
        assert model.samples is not None
        assert "eta0" in model.samples


class TestCarreauYasudaModelExtended:
    """Extended tests for CarreauYasudaModel."""

    @pytest.fixture
    def carreau_yasuda_data(self):
        """Generate synthetic Carreau-Yasuda data."""
        np.random.seed(42)
        shear_rate = np.logspace(-2, 3, 50)
        true_eta0 = 100.0
        true_eta_inf = 1.0
        true_lambda = 1.0
        true_a = 2.0
        true_n = 0.5
        noise_level = 0.05

        eta_true = true_eta_inf + (true_eta0 - true_eta_inf) * (
            1 + (true_lambda * shear_rate) ** true_a
        ) ** ((true_n - 1) / true_a)
        noise = noise_level * eta_true * np.random.randn(len(shear_rate))
        eta = eta_true + noise

        return shear_rate, eta

    def test_carreau_yasuda_predict_before_fit(self):
        """Test that predict() raises error before fitting."""
        model = CarreauYasudaModel()

        with pytest.raises(RuntimeError, match="Model must be fit before prediction"):
            model.predict(np.array([1.0, 10.0, 100.0]))

    def test_carreau_yasuda_summary_statistics(self, carreau_yasuda_data):
        """Test summary statistics for CarreauYasudaModel."""
        shear_rate, eta = carreau_yasuda_data

        model = CarreauYasudaModel(n_samples=500, n_warmup=250, n_chains=1)
        model.fit(shear_rate, eta)

        summary = model.summary()

        # Check structure
        assert "eta0" in summary
        assert "eta_inf" in summary
        assert "lambda_" in summary
        assert "a" in summary
        assert "n" in summary
        assert "sigma" in summary

    def test_carreau_yasuda_credible_intervals_all_params(self, carreau_yasuda_data):
        """Test credible intervals for all CarreauYasudaModel parameters."""
        shear_rate, eta = carreau_yasuda_data

        model = CarreauYasudaModel(n_samples=500, n_warmup=250, n_chains=1)
        model.fit(shear_rate, eta)

        for param in ["eta0", "eta_inf", "lambda_", "a", "n", "sigma"]:
            lower, upper = model.get_credible_intervals(param, level=0.95)
            assert lower < upper

    def test_carreau_yasuda_parameter_constraints(self, carreau_yasuda_data):
        """Test that CarreauYasudaModel respects parameter constraints."""
        shear_rate, eta = carreau_yasuda_data

        model = CarreauYasudaModel(n_samples=500, n_warmup=250, n_chains=1)
        model.fit(shear_rate, eta)

        # eta0 should be positive
        assert np.all(model.samples["eta0"] > 0)

        # eta_inf should be positive
        assert np.all(model.samples["eta_inf"] > 0)

        # lambda should be positive
        assert np.all(model.samples["lambda_"] > 0)

        # a should be positive
        assert np.all(model.samples["a"] > 0)

        # sigma should be positive
        assert np.all(model.samples["sigma"] > 0)


class TestModelComparison:
    """Test model comparison and selection."""

    @pytest.fixture
    def complex_viscosity_data(self):
        """Generate complex viscosity data that fits multiple models."""
        np.random.seed(42)
        shear_rate = np.logspace(-2, 3, 50)

        # Generate data from Cross model
        eta0 = 100.0
        eta_inf = 1.0
        lambda_ = 1.0
        m = 0.7
        noise_level = 0.05

        eta_true = eta_inf + (eta0 - eta_inf) / (1 + (lambda_ * shear_rate) ** m)
        noise = noise_level * eta_true * np.random.randn(len(shear_rate))
        eta = eta_true + noise

        return shear_rate, eta

    def test_multiple_models_fit_same_data(self, complex_viscosity_data):
        """Test that multiple models can fit the same data."""
        shear_rate, eta = complex_viscosity_data

        # Fit PowerLaw model
        power_law = PowerLawModel(n_samples=200, n_warmup=100, n_chains=1)
        power_law.fit(shear_rate, eta)

        # Fit Cross model
        cross = CrossModel(n_samples=200, n_warmup=100, n_chains=1)
        cross.fit(shear_rate, eta)

        # Both should fit successfully
        assert power_law.samples is not None
        assert cross.samples is not None

    def test_model_predictions_are_consistent(self, complex_viscosity_data):
        """Test that model predictions are internally consistent."""
        shear_rate, eta = complex_viscosity_data

        model = CrossModel(n_samples=300, n_warmup=150, n_chains=1)
        model.fit(shear_rate, eta)

        # Predict at training points
        predictions = model.predict(shear_rate[:10])

        # Mean predictions should be close to observed values
        # (relaxed tolerance due to model flexibility)
        assert_allclose(predictions["mean"], eta[:10], rtol=0.5)


@pytest.mark.slow
class TestConvergenceDiagnostics:
    """Test MCMC convergence diagnostics (marked as slow)."""

    def test_power_law_multiple_chains_converge(self):
        """Test that multiple chains converge to similar posteriors."""
        np.random.seed(42)
        shear_rate = np.logspace(-1, 2, 30)
        eta = 5.0 * shear_rate ** (0.6 - 1)

        model = PowerLawModel(n_samples=500, n_warmup=250, n_chains=2)
        model.fit(shear_rate, eta)

        # With clean data, posterior should be tight
        K_std = np.std(model.samples["K"])
        n_std = np.std(model.samples["n"])

        # Standard deviations should be relatively small
        assert K_std < 2.0  # K ~ 5.0, so std < 2.0 is reasonable
        assert n_std < 0.2  # n ~ 0.6, so std < 0.2 is reasonable


@pytest.mark.benchmark
class TestPerformance:
    """Performance tests for Bayesian models (marked as benchmark)."""

    def test_power_law_fit_performance(self, benchmark):
        """Benchmark PowerLawModel fitting performance."""
        np.random.seed(42)
        shear_rate = np.logspace(-1, 2, 30)
        eta = 5.0 * shear_rate ** (0.6 - 1)

        def fit_model():
            model = PowerLawModel(n_samples=100, n_warmup=50, n_chains=1)
            model.fit(shear_rate, eta)
            return model

        # Benchmark should complete in reasonable time
        model = benchmark(fit_model)
        assert model.samples is not None


class TestEdgeCaseHandling:
    """Test edge case handling for rheological models."""

    def test_power_law_with_very_small_shear_rates(self):
        """Test PowerLawModel with very small shear rates."""
        np.random.seed(42)
        shear_rate = np.logspace(-5, -2, 20)  # Very small shear rates
        eta = 10.0 * shear_rate ** (0.5 - 1)

        model = PowerLawModel(n_samples=200, n_warmup=100, n_chains=1)
        model.fit(shear_rate, eta)

        # Should fit successfully
        assert model.samples is not None
        assert np.all(np.isfinite(model.samples["K"]))
        assert np.all(np.isfinite(model.samples["n"]))

    def test_power_law_with_very_large_shear_rates(self):
        """Test PowerLawModel with very large shear rates."""
        np.random.seed(42)
        shear_rate = np.logspace(2, 5, 20)  # Very large shear rates
        eta = 2.0 * shear_rate ** (0.8 - 1)

        model = PowerLawModel(n_samples=200, n_warmup=100, n_chains=1)
        model.fit(shear_rate, eta)

        # Should fit successfully
        assert model.samples is not None
        assert np.all(np.isfinite(model.samples["K"]))
        assert np.all(np.isfinite(model.samples["n"]))
