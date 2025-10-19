Uncertainty Quantification
==========================

quantiq provides comprehensive Bayesian uncertainty quantification through NumPyro.

Overview
--------

Bayesian methods provide:

- Full posterior distributions for parameters
- Credible intervals
- Uncertainty propagation
- Model comparison

Built-in Models
---------------

Power-Law Model
^^^^^^^^^^^^^^^

::

    from quantiq.bayesian import PowerLawModel

    model = PowerLawModel(n_samples=2000)
    model.fit(shear_rate, viscosity)
    print(model.summary())

Other Models
^^^^^^^^^^^^

- **ArrheniusModel**: Temperature-dependent viscosity
- **CrossModel**: Flow curves with plateaus
- **CarreauYasudaModel**: Complex rheological behavior

See :doc:`../tutorials/uncertainty_quantification` for detailed examples.
