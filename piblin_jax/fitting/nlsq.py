"""
NLSQ integration for curve fitting.

This module provides a wrapper around the NLSQ library for nonlinear least squares
curve fitting, with automatic fallback to scipy.optimize.curve_fit if NLSQ is not
available.

The NLSQ library (if available) provides enhanced nonlinear least squares fitting
with better convergence properties than scipy for many problems.
"""

from collections.abc import Callable
from typing import Any

import numpy as np


def fit_curve(
    func: Callable[..., np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    p0: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
    absolute_sigma: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Fit a curve using NLSQ or scipy fallback.

    This function attempts to use the NLSQ library for curve fitting. If NLSQ
    is not available, it falls back to scipy.optimize.curve_fit.

    Parameters
    ----------
    func : Callable
        Model function to fit. Should have signature func(x, \\*params).
    x : np.ndarray
        Independent variable data.
    y : np.ndarray
        Dependent variable data.
    p0 : np.ndarray, optional
        Initial parameter guess. If None, will use default initialization.
    sigma : np.ndarray, optional
        Uncertainty/weights for data points. If provided, used in weighted fit.
    absolute_sigma : bool, default=False
        If True, sigma is used in absolute sense and parameter covariance
        matrix is scaled accordingly.
    **kwargs
        Additional keyword arguments passed to fitting function.

    Returns
    -------
    dict
        Dictionary containing:
        - 'params': Fitted parameters
        - 'covariance': Parameter covariance matrix
        - 'method': Method used ('nlsq' or 'scipy')
        - 'success': Whether fit converged
        - 'residuals': Residuals (y - func(x, \\*params))

    Examples
    --------
    >>> import numpy as np
    >>> from piblin_jax.fitting.nlsq import fit_curve
    >>>
    >>> # Define model function
    >>> def linear_model(x, a, b):
    ...     return a * x + b
    >>>
    >>> # Generate data
    >>> x = np.linspace(0, 10, 50)
    >>> y = 2.5 * x + 1.0 + 0.1 * np.random.randn(len(x))
    >>>
    >>> # Fit
    >>> result = fit_curve(linear_model, x, y, p0=[1.0, 0.0])
    >>> print(f"Fitted parameters: {result['params']}")
    >>> print(f"Method used: {result['method']}")

    Notes
    -----
    - NLSQ often provides better convergence than scipy for challenging problems
    - The scipy fallback ensures the function always works
    - For Bayesian inference, consider using BayesianModel instead
    """
    # Try NLSQ first
    try:
        import nlsq

        # Prepare arguments for NLSQ
        nlsq_kwargs = {}
        if p0 is not None:
            nlsq_kwargs["p0"] = p0
        if sigma is not None:
            nlsq_kwargs["sigma"] = sigma

        # Fit using NLSQ
        result = nlsq.optimize(func, x, y, **nlsq_kwargs, **kwargs)

        # Extract results
        params = result.params
        covariance = result.covariance if hasattr(result, "covariance") else None
        success = result.success if hasattr(result, "success") else True

        # Compute residuals
        residuals = y - func(x, *params)

        return {
            "params": params,
            "covariance": covariance,
            "method": "nlsq",
            "success": success,
            "residuals": residuals,
            "result_object": result,
        }

    except (ImportError, AttributeError):
        # Fall back to scipy
        from scipy.optimize import curve_fit

        # Prepare arguments for scipy
        scipy_kwargs: dict[str, Any] = {}
        if sigma is not None:
            scipy_kwargs["sigma"] = sigma
            scipy_kwargs["absolute_sigma"] = absolute_sigma

        # Fit using scipy
        try:
            params, covariance = curve_fit(func, x, y, p0=p0, **scipy_kwargs, **kwargs)
            success = True
        except Exception as e:
            # If fit fails, return failure result
            return {
                "params": p0 if p0 is not None else None,
                "covariance": None,
                "method": "scipy",
                "success": False,
                "residuals": None,
                "error": str(e),
            }

        # Compute residuals
        residuals = y - func(x, *params)

        return {
            "params": params,
            "covariance": covariance,
            "method": "scipy",
            "success": success,
            "residuals": residuals,
        }


def estimate_initial_parameters(
    func: Callable[..., np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """
    Estimate initial parameters for curve fitting.

    Uses simple heuristics to estimate reasonable initial parameter values
    based on the data characteristics.

    Parameters
    ----------
    func : Callable
        Model function (used to determine number of parameters).
    x : np.ndarray
        Independent variable data.
    y : np.ndarray
        Dependent variable data.
    bounds : tuple, optional
        Parameter bounds (lower, upper).

    Returns
    -------
    np.ndarray
        Estimated initial parameters.

    Notes
    -----
    This is a simple heuristic approach. For better results, use domain
    knowledge to provide good initial guesses.
    """
    # Get number of parameters from function signature
    import inspect

    sig = inspect.signature(func)
    n_params = len(sig.parameters) - 1  # Exclude x parameter

    # Simple heuristic: use mean and range of y-values
    y_mean = np.mean(y)
    y_range = np.ptp(y)  # Peak-to-peak

    # Initialize parameters
    if n_params == 1:
        # Single parameter: use mean
        p0 = np.array([y_mean])
    elif n_params == 2:
        # Two parameters: slope and intercept
        slope = (y[-1] - y[0]) / (x[-1] - x[0]) if len(x) > 1 else 1.0
        intercept = y[0] - slope * x[0]
        p0 = np.array([slope, intercept])
    else:
        # Multiple parameters: use simple heuristics
        p0 = np.ones(n_params)
        p0[0] = y_range  # First param: amplitude
        if n_params > 1:
            p0[1] = y_mean  # Second param: offset
        if n_params > 2:
            p0[2] = np.mean(x)  # Third param: center

    # Apply bounds if provided
    if bounds is not None:
        lower, upper = bounds
        p0 = np.clip(p0, lower, upper)

    return p0


__all__ = [
    "estimate_initial_parameters",
    "fit_curve",
]
