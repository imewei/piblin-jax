Uncertainty Quantification Tutorial
====================================

Learn Bayesian parameter estimation with quantiq.

Example: Power-Law Fitting
---------------------------

::

    import numpy as np
    from quantiq.bayesian import PowerLawModel

    # Data
    shear_rate = np.array([0.1, 1.0, 10.0, 100.0])
    viscosity = np.array([50.0, 15.8, 5.0, 1.58])

    # Fit model
    model = PowerLawModel(n_samples=2000)
    model.fit(shear_rate, viscosity)

    # Results with uncertainty
    summary = model.summary()
    print(summary)

    # Predictions with uncertainty bands
    model.plot_fit(shear_rate, viscosity, show_uncertainty=True)

See :doc:`../user_guide/uncertainty` for complete guide.
