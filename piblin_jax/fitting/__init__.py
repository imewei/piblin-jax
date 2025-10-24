"""Curve fitting module for piblin-jax.

This module provides non-linear least squares (NLSQ) curve fitting for
rheological and scientific models, with automatic parameter initialization
and robust optimization.

## Overview

The fitting module bridges piblin-jax's Bayesian models with classical NLSQ
optimization, providing:
- **Fast parameter estimation** for model initialization
- **Multiple rheological models** (power-law, Arrhenius, Cross, Carreau-Yasuda)
- **Automatic initial parameter guessing** via heuristics
- **scipy.optimize integration** with robust error handling
- **Parameter uncertainty estimates** via covariance matrix

## When to Use

- **NLSQ fitting**: Quick parameter estimates, no uncertainty quantification
- **Bayesian fitting** (`piblin_jax.bayesian`): Full posterior distributions, uncertainty propagation

**Typical workflow**:
1. Use `fit_curve()` for initial parameter estimates
2. Use Bayesian models for comprehensive uncertainty quantification
3. Or use NLSQ estimates to initialize Bayesian priors

## Main Functions

### fit_curve()

Fit rheological models to experimental data using non-linear least squares::

    from piblin_jax import fit_curve
    import numpy as np

    shear_rate = np.logspace(-1, 2, 30)
    viscosity = 5.0 * shear_rate ** (0.6 - 1)

    result = fit_curve(shear_rate, viscosity, model='power_law')
    print(result['params'])  # {'K': 5.02, 'n': 0.598}

**Supported Models**:
- `'power_law'`: η = K * γ̇^(n-1)
- `'arrhenius'`: η = A * exp(Ea / (R*T))
- `'cross'`: η = η∞ + (η₀ - η∞) / (1 + (λγ̇)^m)
- `'carreau_yasuda'`: η = η∞ + (η₀ - η∞) * [1 + (λγ̇)^a]^((n-1)/a)

### estimate_initial_parameters()

Automatically estimate initial parameter guesses for optimization::

    from piblin_jax import estimate_initial_parameters

    initial = estimate_initial_parameters(
        shear_rate,
        viscosity,
        model='power_law'
    )
    print(initial)  # {'K': 4.8, 'n': 0.55}

**Methods**:
- Power-law: Linear regression on log-log data
- Arrhenius: Linear regression on log(η) vs 1/T
- Cross/Carreau-Yasuda: Plateau detection + heuristics

## Examples

### Basic Fitting

Example::

    import numpy as np
    from piblin_jax import fit_curve

    # Generate data
    shear_rate = np.logspace(-2, 3, 50)
    viscosity = 5.0 * shear_rate ** (0.6 - 1)

    # Fit model
    result = fit_curve(shear_rate, viscosity, model='power_law')

    # Extract results
    params = result['params']
    K, n = params['K'], params['n']
    covariance = result['covariance']
    residuals = result['residuals']

    print(f"K = {K:.3f} ± {np.sqrt(covariance[0,0]):.3f}")
    print(f"n = {n:.3f} ± {np.sqrt(covariance[1,1]):.3f}")

### With Initial Guesses

Example::

    # Provide initial guesses
    initial = {'K': 3.0, 'n': 0.5}
    result = fit_curve(
        shear_rate,
        viscosity,
        model='power_law',
        initial_params=initial
    )

### Multiple Models

Example::

    models = ['power_law', 'cross', 'carreau_yasuda']
    results = {}

    for model in models:
        try:
            results[model] = fit_curve(shear_rate, viscosity, model=model)
            print(f"{model}: RSS = {np.sum(results[model]['residuals']**2):.3e}")
        except Exception as e:
            print(f"{model} failed: {e}")

### NLSQ → Bayesian Workflow

Example::

    from piblin_jax import fit_curve
    from piblin_jax.bayesian.models import PowerLawModel

    # Step 1: Quick NLSQ estimate
    nlsq_result = fit_curve(shear_rate, viscosity, model='power_law')
    print(f"NLSQ estimate: K={nlsq_result['params']['K']:.2f}")

    # Step 2: Bayesian fitting with uncertainty
    model = PowerLawModel(n_samples=2000)
    model.fit(shear_rate, viscosity)

    # Step 3: Compare results
    bayesian_K = np.mean(model.samples['K'])
    K_std = np.std(model.samples['K'])
    print(f"Bayesian estimate: K={bayesian_K:.2f} ± {K_std:.2f}")

## Return Values

`fit_curve()` returns a dictionary with:
- `'params'`: dict[str, float] - Fitted parameter values
- `'covariance'`: np.ndarray - Parameter covariance matrix
- `'residuals'`: np.ndarray - Fitting residuals (y_data - y_fit)
- `'success'`: bool - Optimization convergence status
- `'message'`: str - Optimization status message
- `'nfev'`: int - Number of function evaluations

## Error Handling

Example::

    try:
        result = fit_curve(x, y, model='power_law')
        if not result['success']:
            print(f"Optimization warning: {result['message']}")
            # Still can use params, but check residuals
    except ValueError as e:
        print(f"Invalid input: {e}")
    except RuntimeError as e:
        print(f"Optimization failed: {e}")

## Implementation Details

**Optimization**:
- Backend: `scipy.optimize.curve_fit` with Levenberg-Marquardt algorithm
- Automatic bounds: Parameters constrained to physical ranges
- Robust initialization: Automatic parameter guessing prevents convergence failures

**Performance**:
- Typical fit time: 1-10 ms (CPU)
- No GPU acceleration (classical optimization)
- For GPU speedup, use Bayesian models with JAX backend

## See Also

- `piblin_jax.bayesian.models` - Bayesian fitting with uncertainty quantification
- `piblin_jax.bayesian.models.PowerLawModel` - Bayesian power-law fitting
- `piblin_jax.bayesian.models.ArrheniusModel` - Bayesian Arrhenius fitting
- `scipy.optimize.curve_fit` - Underlying optimization routine

## References

.. [1] More, J.J., Garbow, B.S., and Hillstrom, K.E. (1980).
       "User Guide for MINPACK-1", Argonne National Laboratory.
.. [2] Marquardt, D.W. (1963). "An Algorithm for Least-Squares
       Estimation of Nonlinear Parameters", SIAM Journal on Applied Mathematics.
"""

from .nlsq import estimate_initial_parameters, fit_curve

__all__ = [
    "estimate_initial_parameters",
    "fit_curve",
]
