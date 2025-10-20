Uncertainty Quantification Tutorial
====================================

This tutorial demonstrates how to perform Bayesian uncertainty quantification
in quantiq for rheological and scientific data analysis. You'll learn how to:

- Fit Bayesian models to experimental data
- Obtain parameter estimates with credible intervals
- Generate predictions with uncertainty bands
- Propagate uncertainty through transforms
- Interpret posterior distributions

Why Uncertainty Quantification?
--------------------------------

Traditional least-squares fitting gives point estimates of parameters but
doesn't quantify uncertainty in those estimates. Bayesian methods provide:

**Full posterior distributions**
    Not just a single value, but the entire probability distribution over
    parameter values given your data.

**Credible intervals**
    Rigorous uncertainty ranges for parameters (e.g., "95% probability that
    K is between 4.5 and 5.5").

**Prediction intervals**
    Uncertainty bands for model predictions that account for both parameter
    uncertainty and measurement noise.

**Principled model comparison**
    Bayesian evidence for comparing competing models.

Basic Example: Power-Law Model
-------------------------------

Let's start with a simple shear-thinning fluid following the power-law model:

.. math::

   \eta = K \dot{\gamma}^{n-1}

where :math:`\eta` is viscosity, :math:`\dot{\gamma}` is shear rate,
:math:`K` is consistency index, and :math:`n` is flow behavior index.

Generate Synthetic Data
~~~~~~~~~~~~~~~~~~~~~~~

First, create some synthetic data with noise::

    import numpy as np
    from quantiq.bayesian.models import PowerLawModel

    # True parameters
    K_true = 5.0
    n_true = 0.6

    # Generate data
    np.random.seed(42)
    shear_rate = np.logspace(-1, 2, 20)
    viscosity_true = K_true * shear_rate ** (n_true - 1)
    viscosity = viscosity_true + np.random.normal(0, 0.5, size=len(shear_rate))

Fit Bayesian Model
~~~~~~~~~~~~~~~~~~

Now fit the Bayesian power-law model::

    # Create and fit model
    model = PowerLawModel(n_samples=2000, n_warmup=1000)
    model.fit(shear_rate, viscosity)

    # Check if sampling succeeded
    if model.is_fitted:
        print("Model fitting successful!")
    else:
        print("Warning: Check sampling diagnostics")

The model uses MCMC (Markov Chain Monte Carlo) to sample from the posterior
distribution. ``n_samples=2000`` specifies the number of posterior samples,
and ``n_warmup=1000`` is the burn-in period.

Examine Results
~~~~~~~~~~~~~~~

View a summary of the posterior distribution::

    summary = model.summary()
    print(summary)

Output::

    Parameter Posterior Summary:
    ----------------------------
    K: mean=5.02, std=0.15, 95% CI=[4.73, 5.31]
    n: mean=0.598, std=0.012, 95% CI=[0.575, 0.621]
    sigma: mean=0.51, std=0.09, 95% CI=[0.38, 0.71]

The summary shows:

- **mean**: Posterior mean (point estimate)
- **std**: Posterior standard deviation (uncertainty)
- **95% CI**: 95% credible interval

Extract Specific Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Access posterior samples directly::

    # Get posterior samples
    samples = model.samples
    K_samples = samples['K']
    n_samples = samples['n']

    # Calculate custom statistics
    K_median = np.median(K_samples)
    K_90_interval = np.percentile(K_samples, [5, 95])

    print(f"K median: {K_median:.3f}")
    print(f"K 90% interval: [{K_90_interval[0]:.3f}, {K_90_interval[1]:.3f}]")

Visualize Results
~~~~~~~~~~~~~~~~~

Generate diagnostic plots::

    import matplotlib.pyplot as plt

    # Plot fit with uncertainty
    fig, axes = model.plot_fit(
        shear_rate,
        viscosity,
        show_uncertainty=True,
        uncertainty_level=0.95
    )
    plt.savefig('power_law_fit.png', dpi=300)
    plt.show()

The plot shows:

- Data points (observed viscosity)
- Mean prediction (posterior mean fit)
- 95% credible interval (shaded band)

Make Predictions
~~~~~~~~~~~~~~~~

Predict at new shear rates with uncertainty::

    # New prediction points
    new_shear_rate = np.logspace(-2, 3, 100)

    # Predict with uncertainty
    predictions = model.predict(new_shear_rate, return_uncertainty=True)
    mean_pred = predictions['mean']
    lower_pred = predictions['lower']
    upper_pred = predictions['upper']

    # Plot predictions
    plt.figure(figsize=(10, 6))
    plt.plot(new_shear_rate, mean_pred, 'r-', label='Mean prediction')
    plt.fill_between(new_shear_rate, lower_pred, upper_pred,
                     alpha=0.3, label='95% prediction interval')
    plt.scatter(shear_rate, viscosity, c='k', label='Data')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Shear Rate (1/s)')
    plt.ylabel('Viscosity (Pa·s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

Working with Datasets
---------------------

Quantiq integrates Bayesian uncertainty with the Dataset API::

    from quantiq.data.datasets import OneDimensionalDataset

    # Create dataset
    dataset = OneDimensionalDataset(
        independent_variable_data=shear_rate,
        dependent_variable_data=viscosity,
        conditions={'temperature': 25.0, 'sample': 'A1'}
    )

    # Fit model to dataset
    model = PowerLawModel(n_samples=2000)
    model.fit(dataset.independent_variable_data,
              dataset.dependent_variable_data)

    # Add uncertainty to dataset
    dataset_with_unc = dataset.with_uncertainty(
        model=model,
        n_samples=1000,
        keep_samples=True
    )

    # Check uncertainty status
    print(f"Has uncertainty: {dataset_with_unc.has_uncertainty}")

    # Get credible intervals
    lower, upper = dataset_with_unc.get_credible_intervals(level=0.95)

Propagating Uncertainty Through Transforms
-------------------------------------------

Uncertainty can be propagated through data transformations::

    from quantiq.transform.dataset import GaussianSmoothing

    # Create dataset with uncertainty
    dataset_with_unc = dataset.with_uncertainty(
        model=model,
        n_samples=1000,
        keep_samples=True
    )

    # Apply transform with uncertainty propagation
    smoother = GaussianSmoothing(sigma=2.0)
    smoothed = smoother.apply_to(
        dataset_with_unc,
        propagate_uncertainty=True
    )

    # Uncertainty is now propagated through the transform
    print(f"Smoothed has uncertainty: {smoothed.has_uncertainty}")

Advanced Example: Arrhenius Model
----------------------------------

Temperature-dependent viscosity following Arrhenius equation:

.. math::

   \eta = A \exp\left(\frac{E_a}{RT}\right)

where :math:`A` is pre-exponential factor, :math:`E_a` is activation energy,
:math:`R` is gas constant, and :math:`T` is temperature.

Generate Temperature-Dependent Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    from quantiq.bayesian.models import ArrheniusModel

    # True parameters
    A_true = 1e-3  # Pa·s
    Ea_true = 50000  # J/mol
    R = 8.314  # J/(mol·K)

    # Temperature range (K)
    temperature = np.linspace(273, 373, 15)

    # Generate data with noise
    np.random.seed(42)
    viscosity_true = A_true * np.exp(Ea_true / (R * temperature))
    viscosity = viscosity_true * np.random.lognormal(0, 0.1, size=len(temperature))

Fit Arrhenius Model
~~~~~~~~~~~~~~~~~~~

::

    # Fit model
    model = ArrheniusModel(n_samples=2000)
    model.fit(temperature, viscosity)

    # View results
    print(model.summary())

    # Plot fit
    fig, axes = model.plot_fit(temperature, viscosity, show_uncertainty=True)
    plt.xlabel('Temperature (K)')
    plt.ylabel('Viscosity (Pa·s)')
    plt.show()

Interpreting Results
~~~~~~~~~~~~~~~~~~~~

The posterior samples reveal parameter correlations::

    import matplotlib.pyplot as plt

    samples = model.samples
    A_samples = samples['A']
    Ea_samples = samples['Ea']

    # Plot joint distribution
    plt.figure(figsize=(8, 6))
    plt.scatter(A_samples, Ea_samples / 1000, alpha=0.3, s=1)
    plt.xlabel('A (Pa·s)')
    plt.ylabel('Ea (kJ/mol)')
    plt.title('Joint Posterior Distribution')
    plt.grid(True, alpha=0.3)
    plt.show()

This scatter plot reveals correlation between :math:`A` and :math:`E_a`:
if :math:`A` is higher, :math:`E_a` tends to be higher too.

Model Comparison
----------------

Compare different rheological models using Bayesian evidence::

    from quantiq.bayesian.models import PowerLawModel, CrossModel

    # Fit competing models
    power_law = PowerLawModel(n_samples=2000)
    power_law.fit(shear_rate, viscosity)

    cross = CrossModel(n_samples=2000)
    cross.fit(shear_rate, viscosity)

    # Compare using information criteria
    power_law_aic = power_law.aic()
    cross_aic = cross.aic()

    print(f"Power-law AIC: {power_law_aic:.1f}")
    print(f"Cross AIC: {cross_aic:.1f}")

    if cross_aic < power_law_aic:
        print("Cross model is preferred (lower AIC)")
    else:
        print("Power-law model is preferred (lower AIC)")

Lower AIC indicates better model fit penalized for complexity.

Tips and Best Practices
------------------------

**Number of samples**
    Use at least 1000-2000 samples for reliable uncertainty estimates.
    More samples give smoother distributions but take longer.

**Convergence diagnostics**
    Always check ``model.is_fitted`` and examine trace plots to ensure
    MCMC chains have converged::

        if not model.is_fitted:
            print("Warning: Sampling may not have converged")
            # Increase n_samples or n_warmup

**Prior sensitivity**
    Bayesian results depend on priors. quantiq uses weakly informative
    priors by default. For custom priors, see the API documentation.

**Computational cost**
    Bayesian fitting is ~10-100x slower than NLSQ. Use NLSQ first for
    initial exploration, then Bayesian for final analysis with uncertainty.

**Uncertainty vs confidence**
    Credible intervals (Bayesian) have a different interpretation than
    confidence intervals (frequentist). A 95% credible interval means
    "95% probability the parameter is in this range given the data."

Next Steps
----------

- See :doc:`rheological_models` for detailed model descriptions
- See :doc:`custom_transforms` to create uncertainty-aware transforms
- See :doc:`../user_guide/uncertainty` for complete API reference
- See ``examples/bayesian_rheological_models.py`` for full code examples

References
----------

- Gelman, A., et al. (2013). Bayesian Data Analysis, 3rd Edition.
  Chapman and Hall/CRC.
- McElreath, R. (2020). Statistical Rethinking, 2nd Edition.
  CRC Press.
