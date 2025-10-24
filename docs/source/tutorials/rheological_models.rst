Rheological Models Tutorial
============================

This tutorial covers quantiq's built-in rheological models for analyzing
fluid behavior. Learn how to:

- Choose the appropriate model for your data
- Fit models using both NLSQ and Bayesian methods
- Interpret model parameters
- Compare competing models
- Extract physical insights

Overview of Rheological Models
-------------------------------

Quantiq provides four main rheological models:

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Model
     - Use Case
     - Key Parameters
   * - **Power-Law**
     - Shear-thinning/thickening fluids
     - K (consistency), n (flow index)
   * - **Arrhenius**
     - Temperature-dependent viscosity
     - A (pre-exponential), Ea (activation energy)
   * - **Cross**
     - Zero-shear plateau behavior
     - η₀, η∞, λ (time constant), m (exponent)
   * - **Carreau-Yasuda**
     - Complex non-Newtonian behavior
     - η₀, η∞, λ, a (transition), n (power index)

Power-Law Model
---------------

Mathematical Form
~~~~~~~~~~~~~~~~~

The power-law model describes shear-thinning (n < 1) or shear-thickening (n > 1)
behavior:

.. math::

   \eta = K \dot{\gamma}^{n-1}

where:

- :math:`\eta` = viscosity (Pa·s)
- :math:`\dot{\gamma}` = shear rate (1/s)
- :math:`K` = consistency index (Pa·s^n)
- :math:`n` = flow behavior index (dimensionless)

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**n = 1**: Newtonian fluid (constant viscosity)

**n < 1**: Shear-thinning (pseudoplastic)
    - Common in polymer solutions, paints, ketchup
    - Viscosity decreases with increasing shear rate

**n > 1**: Shear-thickening (dilatant)
    - Common in concentrated suspensions, cornstarch-water
    - Viscosity increases with increasing shear rate

**K**: Consistency index
    - Higher K = more viscous
    - Units depend on n: Pa·s^n

NLSQ Fitting Example
~~~~~~~~~~~~~~~~~~~~

Quick parameter estimation without uncertainty::

    import numpy as np
    from piblin_jax.fitting import fit_curve

    # Experimental data
    shear_rate = np.array([0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0])
    viscosity = np.array([52.3, 25.1, 18.2, 8.5, 6.2, 2.9, 2.1])

    # Fit power-law model
    result = fit_curve(shear_rate, viscosity, model='power_law')

    # Extract parameters
    K = result['params']['K']
    n = result['params']['n']
    covariance = result['covariance']

    print(f"K = {K:.3f} Pa·s^{n:.3f}")
    print(f"n = {n:.3f}")
    print(f"Fit quality: RSS = {np.sum(result['residuals']**2):.2e}")

Bayesian Fitting Example
~~~~~~~~~~~~~~~~~~~~~~~~~

Full uncertainty quantification::

    from piblin_jax.bayesian.models import PowerLawModel

    # Fit model
    model = PowerLawModel(n_samples=2000, n_warmup=1000)
    model.fit(shear_rate, viscosity)

    # View summary
    print(model.summary())

    # Plot with uncertainty bands
    import matplotlib.pyplot as plt
    model.plot_fit(shear_rate, viscosity, show_uncertainty=True)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Shear Rate (1/s)')
    plt.ylabel('Viscosity (Pa·s)')
    plt.title('Power-Law Fit with 95% Credible Interval')
    plt.show()

Practical Example: Polymer Solution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analyze a polymer melt::

    # Experimental data from capillary rheometry
    shear_rate = np.logspace(-2, 3, 30)
    viscosity_measured = 1000 * shear_rate ** (-0.35)
    viscosity = viscosity_measured * np.random.lognormal(0, 0.05, size=len(shear_rate))

    # Bayesian fit
    model = PowerLawModel(n_samples=2000)
    model.fit(shear_rate, viscosity)

    # Extract results
    samples = model.samples
    K_mean = np.mean(samples['K'])
    n_mean = np.mean(samples['n'])

    print(f"Consistency index K = {K_mean:.1f} Pa·s^n")
    print(f"Flow index n = {n_mean:.3f}")

    if n_mean < 1:
        print("Material is shear-thinning (pseudoplastic)")
    elif n_mean > 1:
        print("Material is shear-thickening (dilatant)")
    else:
        print("Material is Newtonian")

Arrhenius Model
---------------

Mathematical Form
~~~~~~~~~~~~~~~~~

Temperature dependence of viscosity:

.. math::

   \eta(T) = A \exp\left(\frac{E_a}{RT}\right)

where:

- :math:`A` = pre-exponential factor (Pa·s)
- :math:`E_a` = activation energy (J/mol)
- :math:`R` = gas constant = 8.314 J/(mol·K)
- :math:`T` = absolute temperature (K)

Physical Interpretation
~~~~~~~~~~~~~~~~~~~~~~~

**Ea (Activation Energy)**
    - Energy barrier for molecular flow
    - Higher Ea = more temperature-sensitive
    - Typical range: 20-100 kJ/mol for liquids

**A (Pre-exponential Factor)**
    - Viscosity at infinite temperature (theoretical)
    - Related to molecular structure and size

Temperature Sensitivity
~~~~~~~~~~~~~~~~~~~~~~~

Calculate viscosity change with temperature::

    from piblin_jax.bayesian.models import ArrheniusModel

    # Temperature range (K)
    temperature = np.array([273, 298, 323, 348, 373])
    viscosity = np.array([15.2, 8.5, 5.1, 3.2, 2.1])  # Pa·s

    # Fit model
    model = ArrheniusModel(n_samples=2000)
    model.fit(temperature, viscosity)

    # Extract activation energy
    samples = model.samples
    Ea = np.mean(samples['Ea'])  # J/mol
    Ea_std = np.std(samples['Ea'])

    print(f"Activation energy: {Ea/1000:.1f} ± {Ea_std/1000:.1f} kJ/mol")

    # Predict at new temperature
    T_new = 310  # K (37°C, body temperature)
    pred = model.predict(np.array([T_new]), return_uncertainty=True)
    print(f"Predicted viscosity at {T_new}K: {pred['mean'][0]:.2f} Pa·s")
    print(f"95% CI: [{pred['lower'][0]:.2f}, {pred['upper'][0]:.2f}]")

Practical Application: Cooking Oil
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analyze temperature-dependent viscosity of cooking oil::

    # Experimental data
    T_celsius = np.array([20, 40, 60, 80, 100])
    T_kelvin = T_celsius + 273.15
    viscosity = np.array([58.5, 35.2, 22.8, 16.1, 11.9])  # mPa·s

    # Fit Arrhenius model
    model = ArrheniusModel(n_samples=2000)
    model.fit(T_kelvin, viscosity)

    # Plot Arrhenius plot (ln(η) vs 1/T)
    R = 8.314
    samples = model.samples

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Standard plot
    model.plot_fit(T_kelvin, viscosity, show_uncertainty=True, ax=ax1)
    ax1.set_xlabel('Temperature (K)')
    ax1.set_ylabel('Viscosity (mPa·s)')

    # Arrhenius plot
    ax2.scatter(1000/T_kelvin, np.log(viscosity), c='k', s=50, label='Data')
    T_range = np.linspace(T_kelvin.min(), T_kelvin.max(), 100)
    for i in range(0, len(samples['A']), 100):
        A_i = samples['A'][i]
        Ea_i = samples['Ea'][i]
        eta_i = A_i * np.exp(Ea_i / (R * T_range))
        ax2.plot(1000/T_range, np.log(eta_i), 'r-', alpha=0.05)
    ax2.set_xlabel('1000/T (1/K)')
    ax2.set_ylabel('ln(η)')
    ax2.set_title('Arrhenius Plot')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

Cross Model
-----------

Mathematical Form
~~~~~~~~~~~~~~~~~

Describes fluids with zero-shear and infinite-shear plateaus:

.. math::

   \eta = \eta_\infty + \frac{\eta_0 - \eta_\infty}{1 + (\lambda \dot{\gamma})^m}

where:

- :math:`\eta_0` = zero-shear viscosity (Pa·s)
- :math:`\eta_\infty` = infinite-shear viscosity (Pa·s)
- :math:`\lambda` = relaxation time (s)
- :math:`m` = rate constant (dimensionless)

When to Use Cross Model
~~~~~~~~~~~~~~~~~~~~~~~

**Ideal for:**
    - Polymer melts and solutions
    - Materials with clear zero-shear plateau
    - Wide shear rate range data

**Advantages over power-law:**
    - Captures both Newtonian and non-Newtonian regions
    - More accurate extrapolation
    - Physical meaning for all parameters

Fitting Example
~~~~~~~~~~~~~~~

::

    from piblin_jax.bayesian.models import CrossModel

    # Wide shear rate range
    shear_rate = np.logspace(-3, 3, 50)

    # Generate Cross-model data
    eta_0 = 100.0
    eta_inf = 1.0
    lambda_ = 1.0
    m = 0.7
    viscosity_true = eta_inf + (eta_0 - eta_inf) / (1 + (lambda_ * shear_rate)**m)
    viscosity = viscosity_true * np.random.lognormal(0, 0.05, size=len(shear_rate))

    # Fit Cross model
    model = CrossModel(n_samples=2000)
    model.fit(shear_rate, viscosity)

    # View summary
    print(model.summary())

    # Plot with regions labeled
    fig, ax = plt.subplots(figsize=(10, 6))
    model.plot_fit(shear_rate, viscosity, show_uncertainty=True, ax=ax)

    # Add region labels
    ax.axvline(1/lambda_, color='gray', linestyle='--', alpha=0.5)
    ax.text(0.01, eta_0*1.1, 'Zero-shear\nplateau', fontsize=10)
    ax.text(100, eta_inf*1.5, 'Infinite-shear\nplateau', fontsize=10)
    ax.text(1/lambda_, (eta_0+eta_inf)/2, 'Transition\nregion',
            fontsize=10, ha='center')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Shear Rate (1/s)')
    ax.set_ylabel('Viscosity (Pa·s)')
    ax.set_title('Cross Model Fit')
    ax.grid(True, alpha=0.3)
    plt.show()

Extract Physical Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    samples = model.samples

    eta_0_mean = np.mean(samples['eta_0'])
    eta_inf_mean = np.mean(samples['eta_inf'])
    lambda_mean = np.mean(samples['lambda'])

    print(f"Zero-shear viscosity: {eta_0_mean:.2f} Pa·s")
    print(f"Infinite-shear viscosity: {eta_inf_mean:.2f} Pa·s")
    print(f"Relaxation time: {lambda_mean:.3f} s")
    print(f"Critical shear rate: {1/lambda_mean:.3f} 1/s")

    # Viscosity ratio
    ratio = eta_0_mean / eta_inf_mean
    print(f"Shear-thinning ratio: {ratio:.1f}x")

Carreau-Yasuda Model
--------------------

Mathematical Form
~~~~~~~~~~~~~~~~~

Most general model for complex non-Newtonian behavior:

.. math::

   \eta = \eta_\infty + (\eta_0 - \eta_\infty)[1 + (\lambda \dot{\gamma})^a]^{(n-1)/a}

where:

- :math:`\eta_0` = zero-shear viscosity
- :math:`\eta_\infty` = infinite-shear viscosity
- :math:`\lambda` = time constant
- :math:`a` = transition parameter
- :math:`n` = power-law index in shear-thinning region

Advantages
~~~~~~~~~~

**Most flexible model:**
    - Captures gradual transitions (via parameter a)
    - Reduces to Cross model when a → ∞
    - Reduces to power-law at high shear rates

**Best for:**
    - Complex polymer systems
    - Materials with smooth transitions
    - Precise fitting across wide shear range

Fitting Example
~~~~~~~~~~~~~~~

::

    from piblin_jax.bayesian.models import CarreauYasudaModel

    # Generate complex rheological data
    shear_rate = np.logspace(-2, 3, 60)
    eta_0 = 1000.0
    eta_inf = 0.5
    lambda_ = 2.0
    a = 2.0
    n = 0.4

    viscosity_true = eta_inf + (eta_0 - eta_inf) * \
                     (1 + (lambda_ * shear_rate)**a)**((n-1)/a)
    viscosity = viscosity_true * np.random.lognormal(0, 0.03, size=len(shear_rate))

    # Fit Carreau-Yasuda model
    model = CarreauYasudaModel(n_samples=2000)
    model.fit(shear_rate, viscosity)

    # Compare with Cross model
    from piblin_jax.bayesian.models import CrossModel
    cross_model = CrossModel(n_samples=2000)
    cross_model.fit(shear_rate, viscosity)

    # Model comparison
    cy_aic = model.aic()
    cross_aic = cross_model.aic()

    print("Model Comparison:")
    print(f"Carreau-Yasuda AIC: {cy_aic:.1f}")
    print(f"Cross AIC: {cross_aic:.1f}")

    if cy_aic < cross_aic:
        print("Carreau-Yasuda provides better fit")
    else:
        print("Cross model is sufficient")

Model Selection Guide
---------------------

Decision Tree
~~~~~~~~~~~~~

Follow this guide to choose the right model:

1. **Temperature dependence?**
   - Yes → **Arrhenius Model**
   - No → Continue to 2

2. **Zero-shear plateau visible?**
   - Yes → Continue to 3
   - No → **Power-Law Model**

3. **Smooth or abrupt transition?**
   - Smooth, gradual → **Carreau-Yasuda Model**
   - Sharp, well-defined → **Cross Model**
   - Simple analysis → **Cross Model**

Comparing Multiple Models
~~~~~~~~~~~~~~~~~~~~~~~~~~

Fit all models and compare::

    from piblin_jax.bayesian.models import (
        PowerLawModel, CrossModel, CarreauYasudaModel
    )

    models = {
        'Power-Law': PowerLawModel(n_samples=2000),
        'Cross': CrossModel(n_samples=2000),
        'Carreau-Yasuda': CarreauYasudaModel(n_samples=2000)
    }

    results = {}
    for name, model in models.items():
        print(f"Fitting {name}...")
        model.fit(shear_rate, viscosity)
        results[name] = {
            'aic': model.aic(),
            'bic': model.bic(),
            'model': model
        }

    # Print comparison table
    print("\nModel Comparison:")
    print(f"{'Model':<20} {'AIC':<10} {'BIC':<10}")
    print("-" * 40)
    for name, res in results.items():
        print(f"{name:<20} {res['aic']:<10.1f} {res['bic']:<10.1f}")

    # Best model by AIC
    best = min(results.items(), key=lambda x: x[1]['aic'])
    print(f"\nBest model: {best[0]} (lowest AIC)")

Practical Tips
--------------

Data Quality Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Shear rate range:**
    - Power-law: 1-2 decades minimum
    - Cross/Carreau-Yasuda: 3+ decades to capture plateaus
    - Arrhenius: At least 4-5 temperature points

**Number of points:**
    - Minimum: 10-15 points
    - Recommended: 20-30 points
    - More points = better parameter uncertainty estimates

**Noise level:**
    - NLSQ handles ~10% noise well
    - Bayesian methods robust to 20%+ noise
    - High noise → use more samples (n_samples=3000+)

Common Pitfalls
~~~~~~~~~~~~~~~

**Extrapolation:**
    - Never extrapolate beyond measured shear rate range
    - Power-law particularly unreliable outside data range
    - Use Cross/Carreau-Yasuda for safer extrapolation

**Parameter correlation:**
    - λ and m often correlated in Cross model
    - Check joint posterior distributions
    - High correlation → may need more data

**Overfitting:**
    - Carreau-Yasuda has 5 parameters
    - May overfit sparse data
    - Use simpler models when possible (Occam's razor)

Next Steps
----------

- See :doc:`uncertainty_quantification` for detailed Bayesian fitting
- See ``examples/bayesian_rheological_models.py`` for complete examples
- See API reference for model parameter details
- See :doc:`../user_guide/concepts` for theoretical background

References
----------

- Cross, M.M. (1965). "Rheology of non-Newtonian fluids: A new flow
  equation for pseudoplastic systems." Journal of Colloid Science,
  20(5), 417-437.
- Carreau, P.J. (1972). "Rheological equations from molecular network
  theories." Transactions of the Society of Rheology, 16(1), 99-127.
- Yasuda, K., et al. (1981). "Shear flow properties of concentrated
  solutions of linear and star branched polystyrenes."
  Rheologica Acta, 20(2), 163-178.
- Bird, R.B., Armstrong, R.C., Hassager, O. (1987). Dynamics of
  Polymeric Liquids, Volume 1: Fluid Mechanics, 2nd Edition.
  Wiley-Interscience.
