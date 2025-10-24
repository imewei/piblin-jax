Curve Fitting
=============

Overview
--------

The ``piblin_jax.fitting`` module provides non-linear least squares (NLSQ) curve fitting
functionality for piblin_jax. It offers a fast, deterministic approach to parameter
estimation when uncertainty quantification is not required.

This module integrates with the JAX-based NLSQ library for high-performance optimization
on CPU/GPU, with automatic fallback to scipy's curve_fit when JAX is unavailable. The
fitting functions are designed to work seamlessly with quantiq's data structures while
providing a simple, familiar interface.

Key characteristics:

- **Performance**: JAX-based optimization with JIT compilation provides significant
  speed improvements over traditional scipy methods, especially for complex models
  or large datasets. GPU acceleration is automatically utilized when available.

- **Automatic Initial Estimates**: The module includes heuristics for automatically
  estimating initial parameter values for common model types, reducing the need for
  manual tuning.

- **Robust Optimization**: The underlying NLSQ implementation uses the Levenberg-Marquardt
  algorithm, which balances the speed of gradient descent with the stability of Gauss-Newton
  methods.

- **Scipy Compatibility**: When JAX is not available, the module automatically falls back
  to scipy.optimize.curve_fit, ensuring broad compatibility across different environments.

When to use fitting vs. Bayesian models:

- **Use fitting** when you need fast parameter estimates, have clean data, and don't need
  uncertainty quantification. Ideal for exploratory analysis, real-time processing, or
  when computational resources are limited.

- **Use Bayesian models** when you need full uncertainty quantification, want to incorporate
  prior knowledge, have noisy/limited data, or need to compare competing models rigorously.

Quick Examples
--------------

Basic Curve Fitting
^^^^^^^^^^^^^^^^^^^

Fit a custom function to data::

    from piblin_jax.fitting import fit_curve
    import numpy as np

    def power_law(x, K, n):
        """Power law model: y = K * x^n"""
        return K * x**n

    # Prepare data
    x_data = np.logspace(-2, 3, 50)
    y_data = np.array([...])  # Your measurements

    # Fit the model
    params, covariance = fit_curve(
        f=power_law,
        xdata=x_data,
        ydata=y_data,
        p0=[1.0, 0.5]  # Initial guess: K=1.0, n=0.5
    )

    K_opt, n_opt = params
    print(f"Fitted parameters: K={K_opt:.3f}, n={n_opt:.3f}")

Automatic Initial Parameter Estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let the module estimate initial parameters for you::

    from piblin_jax.fitting import fit_curve, estimate_initial_parameters

    # Estimate initial parameters automatically
    p0 = estimate_initial_parameters(
        f=power_law,
        xdata=x_data,
        ydata=y_data,
        model_type="power_law"  # Hint for better estimates
    )

    # Fit with estimated parameters
    params, covariance = fit_curve(
        f=power_law,
        xdata=x_data,
        ydata=y_data,
        p0=p0
    )

Working with Bounds and Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Apply parameter bounds for physically meaningful results::

    # Fit with parameter bounds
    params, covariance = fit_curve(
        f=power_law,
        xdata=x_data,
        ydata=y_data,
        p0=[1.0, 0.5],
        bounds=([0.0, 0.0], [np.inf, 2.0])  # K > 0, 0 < n < 2
    )

    # Extract parameter uncertainties from covariance
    param_std = np.sqrt(np.diag(covariance))
    print(f"K = {params[0]:.3f} +/- {param_std[0]:.3f}")
    print(f"n = {params[1]:.3f} +/- {param_std[1]:.3f}")

See Also
--------

- :doc:`bayesian` - Bayesian inference for full uncertainty quantification
- :doc:`data` - Data structures for organizing measurement data
- `scipy.optimize.curve_fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html>`_ - Scipy curve fitting (fallback backend)
- `NLSQ Documentation <https://github.com/Joshuaalbert/jax-nlsq>`_ - JAX-based non-linear least squares

API Reference
-------------

Module Contents
^^^^^^^^^^^^^^^

.. automodule:: piblin_jax.fitting
   :members:
   :undoc-members:
   :show-inheritance:

Non-Linear Least Squares
-------------------------

.. automodule:: piblin_jax.fitting.nlsq
   :members:
   :undoc-members:
   :show-inheritance:
