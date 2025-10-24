"""
Mocked tests for NLSQ integration focusing on fallback behavior and error handling.

This module tests the NLSQ wrapper with mocked external dependencies to ensure
proper fallback to scipy and error handling.
"""

import builtins
import sys
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from piblin_jax.fitting.nlsq import estimate_initial_parameters, fit_curve


def linear_model(x, a, b):
    """Simple linear model: y = a*x + b"""
    return a * x + b


def quadratic_model(x, a, b, c):
    """Quadratic model: y = a*x^2 + b*x + c"""
    return a * x**2 + b * x + c


def power_law_model(x, a, n):
    """Power law model: y = a*x^n"""
    return a * x**n


class TestNLSQFallback:
    """Test NLSQ unavailable scenarios and scipy fallback."""

    def test_nlsq_unavailable_uses_scipy(self, monkeypatch):
        """Test that scipy fallback is used when NLSQ is unavailable."""
        # Mock NLSQ import to fail
        import_original = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "nlsq":
                raise ImportError("nlsq not available")
            return import_original(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        # Generate test data
        np.random.seed(42)
        x = np.linspace(0, 10, 30)
        y = 2.5 * x + 1.0 + 0.1 * np.random.randn(len(x))

        # Fit should use scipy fallback
        result = fit_curve(linear_model, x, y, p0=np.array([1.0, 0.0]))

        # Should succeed with scipy
        assert result["success"]
        assert result["method"] == "scipy"
        assert result["params"] is not None
        assert len(result["params"]) == 2

        # Should have covariance from scipy
        assert result["covariance"] is not None

        # Should have residuals
        assert result["residuals"] is not None
        assert len(result["residuals"]) == len(y)

    def test_nlsq_available_mock(self, monkeypatch):
        """Test behavior when NLSQ is available (mocked)."""
        # Create mock NLSQ module
        mock_nlsq = MagicMock()

        # Create mock result object
        mock_result = Mock()
        mock_result.params = np.array([2.5, 1.0])
        mock_result.covariance = np.eye(2)
        mock_result.success = True

        mock_nlsq.optimize = Mock(return_value=mock_result)

        # Mock the import
        monkeypatch.setitem(sys.modules, "nlsq", mock_nlsq)

        # Generate test data
        x = np.linspace(0, 10, 30)
        y = 2.5 * x + 1.0

        # Fit should use NLSQ
        result = fit_curve(linear_model, x, y, p0=np.array([1.0, 0.0]))

        # Should have called NLSQ optimize
        assert mock_nlsq.optimize.called

        # Should return NLSQ results
        assert result["method"] == "nlsq"
        assert result["success"]
        np.testing.assert_array_equal(result["params"], np.array([2.5, 1.0]))

    def test_nlsq_missing_covariance_attribute(self, monkeypatch):
        """Test handling when NLSQ result lacks covariance attribute."""
        # Create mock NLSQ with result missing covariance
        mock_nlsq = MagicMock()
        mock_result = Mock()
        mock_result.params = np.array([2.5, 1.0])
        # No covariance attribute
        delattr(mock_result, "covariance")
        mock_result.success = True

        mock_nlsq.optimize = Mock(return_value=mock_result)
        monkeypatch.setitem(sys.modules, "nlsq", mock_nlsq)

        x = np.linspace(0, 10, 30)
        y = 2.5 * x + 1.0

        result = fit_curve(linear_model, x, y, p0=np.array([1.0, 0.0]))

        # Should handle missing covariance gracefully
        assert result["method"] == "nlsq"
        assert result["covariance"] is None

    def test_nlsq_missing_success_attribute(self, monkeypatch):
        """Test handling when NLSQ result lacks success attribute."""
        # Create mock NLSQ with result missing success
        mock_nlsq = MagicMock()
        mock_result = Mock()
        mock_result.params = np.array([2.5, 1.0])
        mock_result.covariance = np.eye(2)
        # No success attribute
        delattr(mock_result, "success")

        mock_nlsq.optimize = Mock(return_value=mock_result)
        monkeypatch.setitem(sys.modules, "nlsq", mock_nlsq)

        x = np.linspace(0, 10, 30)
        y = 2.5 * x + 1.0

        result = fit_curve(linear_model, x, y, p0=np.array([1.0, 0.0]))

        # Should default to True for success
        assert result["method"] == "nlsq"
        assert result["success"] is True

    def test_nlsq_attribute_error_fallback(self, monkeypatch):
        """Test fallback to scipy when NLSQ raises AttributeError."""
        # Mock NLSQ to raise AttributeError
        mock_nlsq = MagicMock()
        mock_nlsq.optimize = Mock(side_effect=AttributeError("Missing attribute"))
        monkeypatch.setitem(sys.modules, "nlsq", mock_nlsq)

        np.random.seed(42)
        x = np.linspace(0, 10, 30)
        y = 2.5 * x + 1.0 + 0.1 * np.random.randn(len(x))

        # Should fall back to scipy
        result = fit_curve(linear_model, x, y, p0=np.array([1.0, 0.0]))

        assert result["method"] == "scipy"
        assert result["success"]


class TestScipyFitFailure:
    """Test scipy curve_fit failure scenarios."""

    def test_scipy_fit_convergence_failure(self, monkeypatch):
        """Test handling when scipy curve_fit fails to converge."""
        # Mock NLSQ as unavailable
        import_original = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "nlsq":
                raise ImportError("nlsq not available")
            return import_original(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        # Mock scipy curve_fit to raise an exception
        with patch("scipy.optimize.curve_fit") as mock_curve_fit:
            mock_curve_fit.side_effect = RuntimeError("Optimal parameters not found")

            x = np.linspace(0, 10, 30)
            y = 2.5 * x + 1.0

            # Should return failure result
            result = fit_curve(linear_model, x, y, p0=np.array([1.0, 0.0]))

            assert result["success"] is False
            assert result["method"] == "scipy"
            np.testing.assert_array_equal(result["params"], np.array([1.0, 0.0]))  # Returns p0
            assert result["covariance"] is None
            assert result["residuals"] is None
            assert "error" in result

    def test_scipy_fit_failure_without_p0(self, monkeypatch):
        """Test handling when scipy fails and no initial parameters provided."""
        # Mock NLSQ as unavailable
        import_original = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "nlsq":
                raise ImportError("nlsq not available")
            return import_original(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        with patch("scipy.optimize.curve_fit") as mock_curve_fit:
            mock_curve_fit.side_effect = RuntimeError("Optimal parameters not found")

            x = np.linspace(0, 10, 30)
            y = 2.5 * x + 1.0

            # No p0 provided
            result = fit_curve(linear_model, x, y)

            assert result["success"] is False
            assert result["params"] is None  # No p0 to return


class TestFitCurveWithWeights:
    """Test weighted fitting scenarios."""

    def test_scipy_with_sigma_and_absolute_sigma(self, monkeypatch):
        """Test that sigma and absolute_sigma are passed to scipy correctly."""
        # Mock NLSQ as unavailable
        import_original = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "nlsq":
                raise ImportError("nlsq not available")
            return import_original(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        with patch("scipy.optimize.curve_fit") as mock_curve_fit:
            # Mock successful fit
            mock_curve_fit.return_value = (np.array([2.5, 1.0]), np.eye(2))

            x = np.linspace(0, 10, 30)
            y = 2.5 * x + 1.0
            sigma = 0.5 * np.ones_like(y)

            result = fit_curve(
                linear_model, x, y, p0=np.array([1.0, 0.0]), sigma=sigma, absolute_sigma=True
            )

            # Verify curve_fit was called with sigma and absolute_sigma
            assert mock_curve_fit.called
            call_kwargs = mock_curve_fit.call_args[1]
            assert "sigma" in call_kwargs
            np.testing.assert_array_equal(call_kwargs["sigma"], sigma)
            assert call_kwargs["absolute_sigma"] is True

    def test_nlsq_with_sigma(self, monkeypatch):
        """Test that sigma is passed to NLSQ correctly."""
        mock_nlsq = MagicMock()
        mock_result = Mock()
        mock_result.params = np.array([2.5, 1.0])
        mock_result.covariance = np.eye(2)
        mock_result.success = True
        mock_nlsq.optimize = Mock(return_value=mock_result)
        monkeypatch.setitem(sys.modules, "nlsq", mock_nlsq)

        x = np.linspace(0, 10, 30)
        y = 2.5 * x + 1.0
        sigma = 0.5 * np.ones_like(y)

        result = fit_curve(linear_model, x, y, p0=np.array([1.0, 0.0]), sigma=sigma)

        # Verify NLSQ was called with sigma
        assert mock_nlsq.optimize.called
        call_kwargs = mock_nlsq.optimize.call_args[1]
        assert "sigma" in call_kwargs
        np.testing.assert_array_equal(call_kwargs["sigma"], sigma)


class TestEstimateInitialParametersEdgeCases:
    """Test edge cases in initial parameter estimation."""

    def test_estimate_with_single_data_point(self):
        """Test estimation with minimal data."""

        def constant(x, a):
            return a * np.ones_like(x)

        x = np.array([1.0])
        y = np.array([5.0])

        p0 = estimate_initial_parameters(constant, x, y)

        # Should still work with single point
        assert len(p0) == 1
        assert np.isfinite(p0[0])

    def test_estimate_linear_with_single_point(self):
        """Test linear estimation edge case with minimal data."""
        x = np.array([1.0])
        y = np.array([5.0])

        p0 = estimate_initial_parameters(linear_model, x, y)

        # Should handle edge case gracefully
        assert len(p0) == 2
        assert all(np.isfinite(p0))

    def test_estimate_with_constant_data(self):
        """Test estimation when all y-values are the same."""
        x = np.linspace(0, 10, 50)
        y = 5.0 * np.ones_like(x)  # All constant

        p0 = estimate_initial_parameters(linear_model, x, y)

        # Should handle constant data
        assert len(p0) == 2
        assert all(np.isfinite(p0))

    def test_estimate_with_bounds_clipping(self):
        """Test that bounds properly clip estimated parameters."""
        x = np.linspace(0, 10, 50)
        y = 100.0 * x + 50.0  # Large values

        # Restrictive bounds
        bounds = (np.array([0.0, 0.0]), np.array([10.0, 10.0]))

        p0 = estimate_initial_parameters(linear_model, x, y, bounds=bounds)

        # All parameters should be clipped to bounds
        assert np.all(p0 >= bounds[0])
        assert np.all(p0 <= bounds[1])

    def test_estimate_four_parameters(self):
        """Test estimation for model with 4+ parameters."""

        def four_param_model(x, a, b, c, d):
            return a * x**3 + b * x**2 + c * x + d

        x = np.linspace(0, 10, 50)
        y = 0.1 * x**3 + 0.5 * x**2 + 1.0 * x - 2.0

        p0 = estimate_initial_parameters(four_param_model, x, y)

        # Should estimate all 4 parameters
        assert len(p0) == 4
        assert all(np.isfinite(p0))

    def test_estimate_with_negative_values(self):
        """Test estimation with negative y-values."""
        x = np.linspace(0, 10, 50)
        y = -2.0 * x - 5.0  # Negative slope and intercept

        p0 = estimate_initial_parameters(linear_model, x, y)

        # Should handle negative values
        assert len(p0) == 2
        assert all(np.isfinite(p0))


class TestResultObjectStructure:
    """Test that result dictionaries have correct structure."""

    def test_nlsq_result_structure(self, monkeypatch):
        """Test that NLSQ result has all expected keys."""
        mock_nlsq = MagicMock()
        mock_result = Mock()
        mock_result.params = np.array([2.5, 1.0])
        mock_result.covariance = np.eye(2)
        mock_result.success = True
        mock_nlsq.optimize = Mock(return_value=mock_result)
        monkeypatch.setitem(sys.modules, "nlsq", mock_nlsq)

        x = np.linspace(0, 10, 30)
        y = 2.5 * x + 1.0

        result = fit_curve(linear_model, x, y, p0=np.array([1.0, 0.0]))

        # Verify all expected keys present
        assert "params" in result
        assert "covariance" in result
        assert "method" in result
        assert "success" in result
        assert "residuals" in result
        assert "result_object" in result

    def test_scipy_success_result_structure(self, monkeypatch):
        """Test that scipy success result has expected keys."""
        # Mock NLSQ unavailable
        import_original = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "nlsq":
                raise ImportError("nlsq not available")
            return import_original(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        np.random.seed(42)
        x = np.linspace(0, 10, 30)
        y = 2.5 * x + 1.0 + 0.1 * np.random.randn(len(x))

        result = fit_curve(linear_model, x, y, p0=np.array([1.0, 0.0]))

        # Verify scipy success structure
        assert "params" in result
        assert "covariance" in result
        assert "method" in result
        assert "success" in result
        assert "residuals" in result
        assert result["method"] == "scipy"
        assert result["success"] is True

    def test_scipy_failure_result_structure(self, monkeypatch):
        """Test that scipy failure result has expected keys including error."""
        import_original = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "nlsq":
                raise ImportError("nlsq not available")
            return import_original(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        with patch("scipy.optimize.curve_fit") as mock_curve_fit:
            mock_curve_fit.side_effect = RuntimeError("Fit failed")

            x = np.linspace(0, 10, 30)
            y = 2.5 * x + 1.0

            result = fit_curve(linear_model, x, y, p0=np.array([1.0, 0.0]))

            # Verify failure result structure
            assert "params" in result
            assert "covariance" in result
            assert "method" in result
            assert "success" in result
            assert "residuals" in result
            assert "error" in result
            assert result["success"] is False
            assert result["method"] == "scipy"
