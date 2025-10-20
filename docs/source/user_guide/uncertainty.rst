Uncertainty Quantification
==========================

This guide provides a comprehensive overview of uncertainty quantification in quantiq,
covering both the theoretical foundations and practical implementation details.

Overview
--------

quantiq provides comprehensive Bayesian uncertainty quantification through NumPyro,
a probabilistic programming framework built on JAX. This enables:

**Full posterior distributions**
    Not just point estimates, but complete probability distributions over
    parameter values given your data.

**Credible intervals**
    Rigorous uncertainty ranges with clear probabilistic interpretation
    (e.g., "95% probability the parameter is in this range").

**Uncertainty propagation**
    Automatically propagate uncertainty through transforms and computations.

**Model comparison**
    Principled comparison of competing models using Bayesian evidence.

**GPU acceleration**
    Leverages JAX for fast MCMC sampling on GPU hardware.

Why Bayesian Methods?
---------------------

Traditional vs Bayesian Fitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Traditional Least-Squares:**

- Provides point estimates of parameters
- Standard errors assume asymptotic normality
- Difficult to incorporate prior knowledge
- No natural way to compare models

**Bayesian Approach:**

- Provides full posterior distributions
- Credible intervals have direct probabilistic meaning
- Naturally incorporates prior information
- Model comparison via Bayes factors or information criteria
- Uncertainty propagation is straightforward

Probabilistic Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Consider fitting a power-law model to viscosity data:

**Least-squares result:**
    K = 5.0 ± 0.3 (standard error)

    This means: "If we repeated the experiment many times and computed K each time,
    68% of those estimates would fall within ±0.3 of the true value."

**Bayesian result:**
    K = 5.0 [4.5, 5.5] (95% credible interval)

    This means: "Given the data, there's a 95% probability that the true K
    is between 4.5 and 5.5."

The Bayesian interpretation is more intuitive and directly answers the question
"What do we know about this parameter?"

Bayesian Workflow
-----------------

Standard Analysis Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Specify the Model**
   Define the mathematical relationship between parameters and data::

       from quantiq.bayesian.models import PowerLawModel
       model = PowerLawModel(n_samples=2000)

2. **Fit the Model**
   Run MCMC to sample from the posterior distribution::

       model.fit(shear_rate, viscosity)

3. **Check Diagnostics**
   Verify that MCMC chains have converged::

       if model.is_fitted:
           print("Sampling successful")
       else:
           print("Warning: Check convergence diagnostics")

4. **Examine Posterior**
   Analyze parameter distributions::

       print(model.summary())
       samples = model.samples

5. **Make Predictions**
   Generate predictions with uncertainty::

       predictions = model.predict(new_x, return_uncertainty=True)

6. **Compare Models**
   Evaluate competing models::

       aic = model.aic()
       bic = model.bic()

Built-in Models
---------------

Power-Law Model
^^^^^^^^^^^^^^^

**Equation:**

.. math::

   \eta = K \dot{\gamma}^{n-1}

**Parameters:**

- K: consistency index (Pa·s^n)
- n: flow behavior index (dimensionless)
- sigma: observation noise standard deviation

**Use case:** Shear-thinning/thickening fluids

**Example:**

::

    from quantiq.bayesian.models import PowerLawModel
    import numpy as np

    # Data
    shear_rate = np.logspace(-1, 2, 20)
    viscosity = np.array([...])  # Experimental data

    # Fit model
    model = PowerLawModel(n_samples=2000, n_warmup=1000)
    model.fit(shear_rate, viscosity)

    # View results
    print(model.summary())

    # Extract samples
    K_samples = model.samples['K']
    n_samples = model.samples['n']

**Priors:**

- K ~ LogNormal(log(10), 2.0): Weakly informative, centered at 10 Pa·s^n
- n ~ Normal(0.8, 0.5): Weakly informative, centered at shear-thinning
- sigma ~ HalfNormal(1.0): Observation noise

Arrhenius Model
^^^^^^^^^^^^^^^

**Equation:**

.. math::

   \eta(T) = A \exp\left(\frac{E_a}{RT}\right)

**Parameters:**

- A: pre-exponential factor (Pa·s)
- Ea: activation energy (J/mol)
- sigma: observation noise

**Use case:** Temperature-dependent viscosity

**Example:**

::

    from quantiq.bayesian.models import ArrheniusModel

    # Temperature data (K)
    temperature = np.array([273, 298, 323, 348, 373])
    viscosity = np.array([15.2, 8.5, 5.1, 3.2, 2.1])

    # Fit model
    model = ArrheniusModel(n_samples=2000)
    model.fit(temperature, viscosity)

    # Extract activation energy
    Ea_mean = np.mean(model.samples['Ea'])
    print(f"Activation energy: {Ea_mean/1000:.1f} kJ/mol")

**Priors:**

- A ~ LogNormal(log(1e-3), 5.0): Very weak prior
- Ea ~ Normal(50000, 20000): Centered at typical liquid Ea
- sigma ~ HalfNormal(1.0)

Cross Model
^^^^^^^^^^^

**Equation:**

.. math::

   \eta = \eta_\infty + \frac{\eta_0 - \eta_\infty}{1 + (\lambda \dot{\gamma})^m}

**Parameters:**

- η₀: zero-shear viscosity (Pa·s)
- η∞: infinite-shear viscosity (Pa·s)
- λ: relaxation time (s)
- m: rate constant (dimensionless)
- sigma: observation noise

**Use case:** Polymer melts/solutions with zero-shear plateau

**Example:**

::

    from quantiq.bayesian.models import CrossModel

    # Wide shear rate range
    shear_rate = np.logspace(-3, 3, 50)
    viscosity = np.array([...])

    # Fit model
    model = CrossModel(n_samples=2000)
    model.fit(shear_rate, viscosity)

    # Extract plateaus
    eta_0 = np.mean(model.samples['eta_0'])
    eta_inf = np.mean(model.samples['eta_inf'])
    print(f"Shear-thinning ratio: {eta_0/eta_inf:.1f}x")

**Priors:**

- η₀ ~ LogNormal(log(100), 2.0)
- η∞ ~ LogNormal(log(1), 2.0)
- λ ~ LogNormal(log(1), 2.0)
- m ~ Normal(0.7, 0.3): Constrained to (0, ∞)
- sigma ~ HalfNormal(scale based on data)

Carreau-Yasuda Model
^^^^^^^^^^^^^^^^^^^^

**Equation:**

.. math::

   \eta = \eta_\infty + (\eta_0 - \eta_\infty)[1 + (\lambda \dot{\gamma})^a]^{(n-1)/a}

**Parameters:**

- η₀, η∞: viscosity plateaus (Pa·s)
- λ: time constant (s)
- a: transition parameter (dimensionless)
- n: power-law index (dimensionless)
- sigma: observation noise

**Use case:** Complex non-Newtonian behavior with smooth transitions

**Example:**

::

    from quantiq.bayesian.models import CarreauYasudaModel

    model = CarreauYasudaModel(n_samples=2000)
    model.fit(shear_rate, viscosity)

    # Most flexible model, but requires good data
    # Compare with simpler models using AIC

**Priors:**

- η₀ ~ LogNormal(log(1000), 2.0)
- η∞ ~ LogNormal(log(0.1), 2.0)
- λ ~ LogNormal(log(1), 2.0)
- a ~ LogNormal(log(2), 1.0)
- n ~ Normal(0.5, 0.3): Constrained to (0, 1)
- sigma ~ HalfNormal(scale based on data)

MCMC Sampling
-------------

How MCMC Works
~~~~~~~~~~~~~~

Markov Chain Monte Carlo (MCMC) is an algorithm for sampling from probability
distributions that are difficult to sample from directly.

**Key concepts:**

1. **Markov Chain**: Sequence of samples where each depends only on the previous one
2. **Stationary Distribution**: Target distribution (posterior) that chain converges to
3. **Burn-in (warmup)**: Initial samples discarded before convergence
4. **Thinning**: Optional subsampling to reduce autocorrelation

**NumPyro's NUTS sampler:**

quantiq uses the No-U-Turn Sampler (NUTS), a variant of Hamiltonian Monte Carlo:

- Automatically tunes step size and trajectory length
- More efficient than basic MCMC (Metropolis-Hastings)
- Typically requires fewer samples for same accuracy
- Works well in high dimensions

Sampling Parameters
~~~~~~~~~~~~~~~~~~~

**n_samples** (default: 1000)
    Number of posterior samples to draw after warmup.
    More samples = better posterior approximation but slower.

    - 1000-2000: Good for most applications
    - 3000-5000: High-precision uncertainty estimates
    - 10000+: Publication-quality results

**n_warmup** (default: 1000)
    Number of warmup/burn-in samples to discard.
    Used to tune sampler parameters and reach stationary distribution.

    - 500-1000: Usually sufficient
    - 2000+: Complex models or difficult posteriors

**n_chains** (default: 1)
    Number of independent MCMC chains to run.
    Multiple chains help diagnose convergence issues.

    - 1: Fast, but less diagnostic information
    - 4: Standard for convergence diagnostics (R-hat, ESS)

**Example:**

::

    # High-quality fit with convergence diagnostics
    model = PowerLawModel(
        n_samples=2000,
        n_warmup=1000,
        n_chains=4
    )
    model.fit(shear_rate, viscosity)

Convergence Diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~

Always check if sampling succeeded::

    if not model.is_fitted:
        print("Warning: Sampling may not have converged")
        # Increase n_samples or n_warmup
        # Check for model misspecification

**R-hat statistic** (Gelman-Rubin):
    Measures agreement between chains.

    - R-hat ≈ 1.0: Good convergence
    - R-hat > 1.1: Poor convergence, increase warmup

**Effective sample size (ESS):**
    Accounts for autocorrelation in samples.

    - ESS ≈ n_samples: Low autocorrelation (good)
    - ESS << n_samples: High autocorrelation (increase samples)

Posterior Analysis
------------------

Summary Statistics
~~~~~~~~~~~~~~~~~~

The ``summary()`` method provides key statistics::

    summary = model.summary()
    print(summary)

Output::

    Parameter Posterior Summary:
    ----------------------------
    K: mean=5.02, std=0.15, 95% CI=[4.73, 5.31]
    n: mean=0.598, std=0.012, 95% CI=[0.575, 0.621]
    sigma: mean=0.51, std=0.09, 95% CI=[0.38, 0.71]

**Interpreting statistics:**

- **mean**: Posterior mean (Bayesian point estimate)
- **std**: Posterior standard deviation (uncertainty measure)
- **95% CI**: 95% credible interval (central posterior density)

Accessing Samples
~~~~~~~~~~~~~~~~~

Direct access to posterior samples::

    samples = model.samples

    # Extract specific parameter
    K_samples = samples['K']  # Array of shape (n_samples,)
    n_samples = samples['n']

    # Custom statistics
    K_median = np.median(K_samples)
    K_mode = K_samples[np.argmax(np.histogram(K_samples, bins=50)[0])]

    # Quantiles
    K_quantiles = np.percentile(K_samples, [2.5, 50, 97.5])

Credible Intervals
~~~~~~~~~~~~~~~~~~

**Equal-tailed interval (ETI):**
    Default method. 2.5th and 97.5th percentiles for 95% CI::

        lower = np.percentile(K_samples, 2.5)
        upper = np.percentile(K_samples, 97.5)

**Highest density interval (HDI):**
    Shortest interval containing 95% of probability mass.
    Preferred for skewed distributions::

        from quantiq.bayesian.utils import compute_hdi
        lower, upper = compute_hdi(K_samples, credible_mass=0.95)

Parameter Correlations
~~~~~~~~~~~~~~~~~~~~~~

Posterior samples reveal parameter correlations::

    import matplotlib.pyplot as plt

    # Joint distribution
    plt.scatter(samples['K'], samples['n'], alpha=0.3, s=1)
    plt.xlabel('K')
    plt.ylabel('n')
    plt.title('Joint Posterior Distribution')

    # Correlation coefficient
    corr = np.corrcoef(samples['K'], samples['n'])[0, 1]
    print(f"Correlation: {corr:.3f}")

High correlation indicates parameters are not independently identifiable
from the data (common in complex models).

Model Comparison
----------------

Information Criteria
~~~~~~~~~~~~~~~~~~~~

**Akaike Information Criterion (AIC):**

Lower is better. Balances fit quality and model complexity::

    aic = model.aic()

**Bayesian Information Criterion (BIC):**

Similar to AIC but penalizes complexity more heavily::

    bic = model.bic()

**Comparing models:**

::

    from quantiq.bayesian.models import PowerLawModel, CrossModel

    # Fit both models
    power_law = PowerLawModel(n_samples=2000)
    power_law.fit(shear_rate, viscosity)

    cross = CrossModel(n_samples=2000)
    cross.fit(shear_rate, viscosity)

    # Compare
    print(f"Power-law AIC: {power_law.aic():.1f}")
    print(f"Cross AIC: {cross.aic():.1f}")

    delta_aic = abs(power_law.aic() - cross.aic())
    if delta_aic < 2:
        print("Models are essentially equivalent")
    elif delta_aic < 10:
        print("Moderate evidence for preferred model")
    else:
        print("Strong evidence for preferred model")

Bayes Factors
~~~~~~~~~~~~~

More rigorous model comparison using marginal likelihoods.

**Note:** Requires setting ``enable_bayes_factor=True`` during fitting::

    model = PowerLawModel(n_samples=2000)
    model.fit(shear_rate, viscosity, enable_bayes_factor=True)

    # Get log marginal likelihood
    log_ml = model.log_marginal_likelihood

Uncertainty Propagation
-----------------------

Dataset Integration
~~~~~~~~~~~~~~~~~~~

Add uncertainty to datasets::

    from quantiq.data.datasets import OneDimensionalDataset

    # Create dataset
    dataset = OneDimensionalDataset(
        independent_variable_data=shear_rate,
        dependent_variable_data=viscosity
    )

    # Fit Bayesian model
    model = PowerLawModel(n_samples=2000)
    model.fit(shear_rate, viscosity)

    # Add uncertainty to dataset
    dataset_with_unc = dataset.with_uncertainty(
        model=model,
        n_samples=1000,
        keep_samples=True
    )

    # Check status
    print(f"Has uncertainty: {dataset_with_unc.has_uncertainty}")

Transform Propagation
~~~~~~~~~~~~~~~~~~~~~

Propagate uncertainty through transforms::

    from quantiq.transform.dataset import GaussianSmoothing

    # Apply transform with uncertainty propagation
    smoother = GaussianSmoothing(sigma=2.0)
    smoothed = smoother.apply_to(
        dataset_with_unc,
        propagate_uncertainty=True
    )

    # Uncertainty is now propagated
    lower, upper = smoothed.get_credible_intervals(level=0.95)

Monte Carlo Propagation
~~~~~~~~~~~~~~~~~~~~~~~

For custom operations, use Monte Carlo::

    # Function to propagate uncertainty through
    def my_operation(K, n, shear_rate):
        return K * shear_rate ** (n - 1)

    # Sample-based propagation
    results = []
    for i in range(len(samples['K'])):
        K_i = samples['K'][i]
        n_i = samples['n'][i]
        result_i = my_operation(K_i, n_i, new_shear_rate)
        results.append(result_i)

    results = np.array(results)

    # Compute uncertainty
    mean_result = np.mean(results, axis=0)
    lower_result = np.percentile(results, 2.5, axis=0)
    upper_result = np.percentile(results, 97.5, axis=0)

Best Practices
--------------

Model Selection
~~~~~~~~~~~~~~~

1. **Start simple**: Begin with power-law or Arrhenius
2. **Check residuals**: Look for systematic patterns
3. **Add complexity**: Move to Cross or Carreau-Yasuda if needed
4. **Compare formally**: Use AIC/BIC to justify complexity
5. **Physical meaning**: Prefer models with interpretable parameters

Prior Selection
~~~~~~~~~~~~~~~

**Weakly informative priors** (default in quantiq):

- Constrain parameters to physically reasonable ranges
- Don't dominate the data
- Help with numerical stability

**Custom priors:**

For domain expertise, modify priors::

    # Example: Strong prior on power-law index
    # (requires model subclassing, see API docs)

Computational Efficiency
~~~~~~~~~~~~~~~~~~~~~~~~

**Start small:**
    Test with n_samples=500, n_warmup=500 first

**Increase gradually:**
    Double samples until results stabilize

**Use GPU:**
    Install JAX with GPU support for 10-100x speedup::

        pip install "jax[cuda12]"  # NVIDIA
        pip install "jax[metal]"   # Apple Silicon

**Parallel chains:**
    Use n_chains=4 on multi-core CPU

Common Issues
~~~~~~~~~~~~~

**Poor convergence (R-hat > 1.1):**

- Increase n_warmup
- Try different initial values
- Simplify the model

**Low ESS:**

- Increase n_samples
- Check for high parameter correlation
- Consider reparameterization

**Unrealistic posteriors:**

- Check data scaling (avoid extreme values)
- Verify model is appropriate for data
- Inspect prior sensitivity

**Slow sampling:**

- Reduce n_samples for initial exploration
- Use GPU acceleration
- Consider simpler model

Advanced Topics
---------------

Custom Models
~~~~~~~~~~~~~

Subclass ``BayesianModel`` to create custom models.
See API documentation for details.

Hierarchical Models
~~~~~~~~~~~~~~~~~~~

Model variation across groups (e.g., multiple experiments).
Requires custom NumPyro model definition.

Model Averaging
~~~~~~~~~~~~~~~

Combine predictions from multiple models weighted by evidence::

    # Fit multiple models
    models = [power_law, cross, carreau_yasuda]
    weights = compute_model_weights(models)  # Based on AIC

    # Weighted average predictions
    predictions = sum(w * m.predict(x) for w, m in zip(weights, models))

References
----------

**Bayesian Statistics:**

.. [1] Gelman, A., et al. (2013). Bayesian Data Analysis, 3rd Edition.
       Chapman and Hall/CRC.
.. [2] McElreath, R. (2020). Statistical Rethinking, 2nd Edition.
       CRC Press.

**MCMC Methods:**

.. [3] Hoffman, M.D., & Gelman, A. (2014). "The No-U-Turn Sampler:
       Adaptively Setting Path Lengths in Hamiltonian Monte Carlo."
       Journal of Machine Learning Research, 15, 1593-1623.

**NumPyro:**

.. [4] Phan, D., et al. (2019). "Composable Effects for Flexible and
       Accelerated Probabilistic Programming in NumPyro."
       arXiv:1912.11554.

Next Steps
----------

- See :doc:`../tutorials/uncertainty_quantification` for step-by-step examples
- See :doc:`../tutorials/rheological_models` for model-specific guidance
- See API reference for detailed method documentation
- See ``examples/bayesian_*.py`` for complete working examples
