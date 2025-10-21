"""
Tests for built-in rheological models (Task Group 13).

This module tests:
- PowerLawModel fitting and prediction
- ArrheniusModel with temperature data
- CrossModel fitting
- CarreauYasudaModel fitting
- Credible intervals for all models
- Parameter recovery
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


class TestPowerLawModel:
    """Test PowerLawModel: η = K * γ̇^(n-1)"""

    @pytest.fixture
    def power_law_data(self):
        """Generate synthetic power-law viscosity data."""
        np.random.seed(42)
        shear_rate = np.logspace(-2, 3, 50)  # 0.01 to 1000 s^-1
        true_K = 5.0  # Pa·s^n
        true_n = 0.6  # Shear-thinning
        noise_level = 0.05

        # True viscosity with noise
        eta_true = true_K * shear_rate ** (true_n - 1)
        noise = noise_level * eta_true * np.random.randn(len(shear_rate))
        eta = eta_true + noise

        return shear_rate, eta, true_K, true_n

    def test_power_law_fit(self, power_law_data):
        """Test fitting PowerLawModel to synthetic data."""
        shear_rate, eta, _true_K, _true_n = power_law_data

        model = PowerLawModel(n_samples=500, n_warmup=250, n_chains=1)
        model.fit(shear_rate, eta)

        # Check samples were stored
        assert model.samples is not None
        assert "K" in model.samples
        assert "n" in model.samples
        assert "sigma" in model.samples

        # Check sample shapes
        assert model.samples["K"].shape == (500,)
        assert model.samples["n"].shape == (500,)

    def test_power_law_parameter_recovery(self, power_law_data):
        """Test that PowerLawModel recovers true parameters."""
        shear_rate, eta, true_K, true_n = power_law_data

        model = PowerLawModel(n_samples=1000, n_warmup=500, n_chains=2)
        model.fit(shear_rate, eta)

        # Check posterior means close to true values
        K_mean = np.mean(model.samples["K"])
        n_mean = np.mean(model.samples["n"])

        assert_allclose(K_mean, true_K, rtol=0.3)  # Within 30%
        assert_allclose(n_mean, true_n, rtol=0.3)

    def test_power_law_predict(self, power_law_data):
        """Test PowerLawModel prediction with uncertainty."""
        shear_rate, eta, _, _ = power_law_data

        model = PowerLawModel(n_samples=500, n_warmup=250, n_chains=1)
        model.fit(shear_rate, eta)

        # Predict at new points
        shear_rate_new = np.array([0.1, 1.0, 10.0, 100.0])
        result = model.predict(shear_rate_new, credible_interval=0.95)

        # Check result structure
        assert "mean" in result
        assert "lower" in result
        assert "upper" in result
        assert "samples" in result

        # Check shapes
        assert result["mean"].shape == (4,)
        assert result["lower"].shape == (4,)
        assert result["upper"].shape == (4,)

        # Check uncertainty bounds
        for i in range(4):
            assert result["lower"][i] < result["mean"][i] < result["upper"][i]

    def test_power_law_credible_intervals(self, power_law_data):
        """Test credible intervals for PowerLawModel parameters."""
        shear_rate, eta, _, _ = power_law_data

        model = PowerLawModel(n_samples=500, n_warmup=250, n_chains=1)
        model.fit(shear_rate, eta)

        # Check credible intervals
        K_lower, K_upper = model.get_credible_intervals("K", level=0.95)
        n_lower, n_upper = model.get_credible_intervals("n", level=0.95)

        assert K_lower < K_upper
        assert n_lower < n_upper

        # K should be positive
        assert K_lower > 0


class TestArrheniusModel:
    """Test ArrheniusModel: η(T) = A * exp(Ea / (R*T))"""

    @pytest.fixture
    def arrhenius_data(self):
        """Generate synthetic Arrhenius temperature-viscosity data."""
        np.random.seed(42)
        temperature = np.linspace(250, 400, 30)  # K
        true_A = 1e-5  # Pre-exponential factor
        true_Ea = 50000  # J/mol (activation energy)
        R = 8.314  # J/(mol·K)
        noise_level = 0.05

        # True viscosity with noise
        eta_true = true_A * np.exp(true_Ea / (R * temperature))
        noise = noise_level * eta_true * np.random.randn(len(temperature))
        eta = eta_true + noise

        return temperature, eta, true_A, true_Ea

    def test_arrhenius_fit(self, arrhenius_data):
        """Test fitting ArrheniusModel to temperature data."""
        temperature, eta, _, _ = arrhenius_data

        model = ArrheniusModel(n_samples=500, n_warmup=250, n_chains=1)
        model.fit(temperature, eta)

        # Check samples
        assert model.samples is not None
        assert "A" in model.samples
        assert "Ea" in model.samples
        assert "sigma" in model.samples

    def test_arrhenius_parameter_recovery(self, arrhenius_data):
        """Test ArrheniusModel parameter recovery."""
        temperature, eta, true_A, true_Ea = arrhenius_data

        model = ArrheniusModel(n_samples=1000, n_warmup=500, n_chains=2)
        model.fit(temperature, eta)

        # Check posterior means (relaxed tolerance due to exponential sensitivity)
        A_mean = np.mean(model.samples["A"])
        Ea_mean = np.mean(model.samples["Ea"])

        # Check order of magnitude for A (exponential scaling is sensitive)
        assert np.log10(A_mean) == pytest.approx(np.log10(true_A), abs=1.0)

        # Activation energy should be reasonably close
        assert_allclose(Ea_mean, true_Ea, rtol=0.5)

    def test_arrhenius_predict(self, arrhenius_data):
        """Test ArrheniusModel prediction."""
        temperature, eta, _, _ = arrhenius_data

        model = ArrheniusModel(n_samples=500, n_warmup=250, n_chains=1)
        model.fit(temperature, eta)

        # Predict at new temperatures
        temp_new = np.array([260.0, 300.0, 350.0])
        result = model.predict(temp_new, credible_interval=0.95)

        # Check structure
        assert "mean" in result
        assert "lower" in result
        assert "upper" in result

        # Check uncertainty bounds
        for i in range(3):
            assert result["lower"][i] < result["mean"][i] < result["upper"][i]


class TestCrossModel:
    """Test CrossModel: η(γ̇) = η∞ + (η₀ - η∞) / (1 + (λγ̇)^m)"""

    @pytest.fixture
    def cross_data(self):
        """Generate synthetic Cross model data."""
        np.random.seed(42)
        shear_rate = np.logspace(-2, 3, 50)
        true_eta0 = 100.0  # Zero-shear viscosity
        true_eta_inf = 1.0  # Infinite-shear viscosity
        true_lambda = 1.0  # Time constant
        true_m = 0.7  # Power-law exponent
        noise_level = 0.05

        # True viscosity with noise
        eta_true = true_eta_inf + (true_eta0 - true_eta_inf) / (
            1 + (true_lambda * shear_rate) ** true_m
        )
        noise = noise_level * eta_true * np.random.randn(len(shear_rate))
        eta = eta_true + noise

        return shear_rate, eta, true_eta0, true_eta_inf, true_lambda, true_m

    def test_cross_fit(self, cross_data):
        """Test fitting CrossModel."""
        shear_rate, eta, _, _, _, _ = cross_data

        model = CrossModel(n_samples=500, n_warmup=250, n_chains=1)
        model.fit(shear_rate, eta)

        # Check samples
        assert model.samples is not None
        assert "eta0" in model.samples
        assert "eta_inf" in model.samples
        assert "lambda_" in model.samples
        assert "m" in model.samples

    def test_cross_parameter_recovery(self, cross_data):
        """Test CrossModel parameter recovery."""
        shear_rate, eta, true_eta0, _true_eta_inf, _true_lambda, _true_m = cross_data

        model = CrossModel(n_samples=1000, n_warmup=500, n_chains=2)
        model.fit(shear_rate, eta)

        # Check posterior means
        eta0_mean = np.mean(model.samples["eta0"])
        eta_inf_mean = np.mean(model.samples["eta_inf"])

        # Relaxed tolerances for complex model
        # The Cross model has identifiability issues when parameters are correlated
        assert_allclose(eta0_mean, true_eta0, rtol=0.5)
        # Check that eta_inf is in the right order of magnitude
        assert 0.1 < eta_inf_mean < 10.0  # Should be close to 1.0 but allow flexibility

    def test_cross_predict(self, cross_data):
        """Test CrossModel prediction."""
        shear_rate, eta, _, _, _, _ = cross_data

        model = CrossModel(n_samples=500, n_warmup=250, n_chains=1)
        model.fit(shear_rate, eta)

        # Predict
        shear_rate_new = np.array([0.1, 1.0, 10.0])
        result = model.predict(shear_rate_new)

        assert "mean" in result
        assert result["mean"].shape == (3,)


class TestCarreauYasudaModel:
    """Test CarreauYasudaModel: η = η∞ + (η₀ - η∞) * [1 + (λγ̇)^a]^((n-1)/a)"""

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

        # True viscosity
        eta_true = true_eta_inf + (true_eta0 - true_eta_inf) * (
            1 + (true_lambda * shear_rate) ** true_a
        ) ** ((true_n - 1) / true_a)
        noise = noise_level * eta_true * np.random.randn(len(shear_rate))
        eta = eta_true + noise

        return shear_rate, eta

    def test_carreau_yasuda_fit(self, carreau_yasuda_data):
        """Test fitting CarreauYasudaModel."""
        shear_rate, eta = carreau_yasuda_data

        model = CarreauYasudaModel(n_samples=500, n_warmup=250, n_chains=1)
        model.fit(shear_rate, eta)

        # Check samples
        assert model.samples is not None
        assert "eta0" in model.samples
        assert "eta_inf" in model.samples
        assert "lambda_" in model.samples
        assert "a" in model.samples
        assert "n" in model.samples

    def test_carreau_yasuda_predict(self, carreau_yasuda_data):
        """Test CarreauYasudaModel prediction."""
        shear_rate, eta = carreau_yasuda_data

        model = CarreauYasudaModel(n_samples=500, n_warmup=250, n_chains=1)
        model.fit(shear_rate, eta)

        # Predict
        shear_rate_new = np.array([0.1, 1.0, 10.0])
        result = model.predict(shear_rate_new)

        assert "mean" in result
        assert "lower" in result
        assert "upper" in result
        assert result["mean"].shape == (3,)
