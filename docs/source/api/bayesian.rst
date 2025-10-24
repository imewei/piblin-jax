Bayesian Models
===============

Overview
--------

The ``piblin_jax.bayesian`` module provides a powerful framework for Bayesian inference
and uncertainty quantification in piblin_jax. Built on NumPyro, it enables probabilistic
modeling with Markov Chain Monte Carlo (MCMC) sampling for robust parameter estimation
and predictive uncertainty.

The Bayesian approach offers several advantages over traditional curve fitting:

- **Full Uncertainty Quantification**: Instead of point estimates, Bayesian methods
  provide complete posterior distributions over parameters. This captures parameter
  correlations and enables principled uncertainty propagation to predictions.

- **Prior Knowledge Integration**: Domain expertise and physical constraints can be
  incorporated through informative priors, improving estimates when data is limited
  or noisy.

- **Model Comparison**: Bayesian model evidence enables rigorous comparison of
  competing models, helping you select the most appropriate model for your data.

- **Predictive Distributions**: Obtain not just predicted values but full predictive
  distributions, capturing both aleatoric (data) and epistemic (parameter) uncertainty.

The module includes pre-built models for common rheological relationships (Power Law,
Cross, Carreau-Yasuda) and thermal activation (Arrhenius), with a flexible base class
for creating custom Bayesian models. All models are JIT-compiled for efficient MCMC
sampling.

Quick Examples
--------------

Fitting a Power Law Model
^^^^^^^^^^^^^^^^^^^^^^^^^^

Fit a rheological power law model with MCMC::

    from piblin_jax.bayesian import PowerLawModel
    import numpy as np

    # Prepare data
    shear_rate = np.logspace(-2, 3, 50)
    viscosity = np.array([...])  # Your measurement data

    # Create and fit model
    model = PowerLawModel()
    model.fit(
        x=shear_rate,
        y=viscosity,
        num_warmup=1000,
        num_samples=2000
    )

    # Get parameter posteriors
    samples = model.get_samples()
    print(f"Consistency index K: {samples['K'].mean():.3f} +/- {samples['K'].std():.3f}")
    print(f"Flow index n: {samples['n'].mean():.3f} +/- {samples['n'].std():.3f}")

Making Predictions with Uncertainty
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Generate predictions with full uncertainty quantification::

    # Predict on new data
    x_new = np.logspace(-2, 3, 100)
    predictions = model.predict(x_new)

    # Extract statistics
    y_mean = predictions['mean']
    y_std = predictions['std']
    y_lower = predictions['quantiles'][0.025]  # 2.5th percentile
    y_upper = predictions['quantiles'][0.975]  # 97.5th percentile

    # Plot with uncertainty bands
    import matplotlib.pyplot as plt
    plt.fill_between(x_new, y_lower, y_upper, alpha=0.3, label='95% CI')
    plt.plot(x_new, y_mean, label='Mean prediction')
    plt.scatter(shear_rate, viscosity, label='Data')
    plt.legend()

Custom Priors and Advanced Usage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Customize priors based on domain knowledge::

    from piblin_jax.bayesian import ArrheniusModel

    # Create model with custom priors
    model = ArrheniusModel()

    # Fit with custom MCMC settings
    model.fit(
        x=temperature,
        y=reaction_rate,
        num_warmup=2000,
        num_samples=5000,
        num_chains=4,  # Parallel chains for convergence diagnostics
        target_accept_prob=0.9  # Higher acceptance for difficult posteriors
    )

    # Access full MCMC diagnostics
    mcmc_info = model.get_mcmc_info()
    print(f"Effective sample size: {mcmc_info['ess']}")
    print(f"R-hat (convergence): {mcmc_info['r_hat']}")

See Also
--------

- :doc:`fitting` - Non-linear least squares fitting (faster, no uncertainty)
- `NumPyro Documentation <https://num.pyro.ai/>`_ - Underlying probabilistic programming framework
- `MCMC Diagnostics Guide <https://num.pyro.ai/en/stable/diagnostics.html>`_ - Understanding MCMC convergence

API Reference
-------------

Module Contents
^^^^^^^^^^^^^^^

The ``piblin_jax.bayesian`` module provides the following classes:

- :class:`BayesianModel` - Base class for all Bayesian models
- :class:`PowerLawModel` - Power-law rheological model
- :class:`CrossModel` - Cross rheological model
- :class:`CarreauYasudaModel` - Carreau-Yasuda rheological model
- :class:`ArrheniusModel` - Arrhenius thermal activation model

Base Classes
------------

.. automodule:: piblin_jax.bayesian.base
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Rheological Models
------------------

Power Law Model
^^^^^^^^^^^^^^^

.. automodule:: piblin_jax.bayesian.models.power_law
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Cross Model
^^^^^^^^^^^

.. automodule:: piblin_jax.bayesian.models.cross
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Carreau-Yasuda Model
^^^^^^^^^^^^^^^^^^^^

.. automodule:: piblin_jax.bayesian.models.carreau_yasuda
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Thermal Models
--------------

Arrhenius Model
^^^^^^^^^^^^^^^

.. automodule:: piblin_jax.bayesian.models.arrhenius
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:
