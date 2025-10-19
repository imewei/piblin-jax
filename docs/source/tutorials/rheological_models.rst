Rheological Models Tutorial
============================

Explore built-in rheological models.

Available Models
----------------

1. **PowerLawModel**: Shear-thinning/thickening
2. **ArrheniusModel**: Temperature dependence
3. **CrossModel**: Zero-shear plateau
4. **CarreauYasudaModel**: Complex behavior

See ``examples/bayesian_rheological_models.py`` for complete examples.

Quick Example
-------------

::

    from quantiq.bayesian import PowerLawModel

    model = PowerLawModel()
    model.fit(shear_rate, viscosity)
    model.plot_fit(shear_rate, viscosity, show_uncertainty=True)

See :doc:`uncertainty_quantification` for detailed tutorial.
