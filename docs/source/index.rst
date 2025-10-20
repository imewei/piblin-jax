quantiq Documentation
=====================

**quantiq** is a modern JAX-powered framework for measurement data science,
providing a complete reimplementation of piblin with dramatic performance
improvements and advanced uncertainty quantification.

.. image:: https://img.shields.io/badge/python-3.12+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
   :target: https://github.com/astral-sh/ruff
   :alt: Ruff

Key Features
------------

* **JAX-Powered Performance**: 5-10x CPU speedup, 50-100x GPU acceleration
* **Bayesian Uncertainty Quantification**: NumPyro integration for rigorous uncertainty propagation
* **100% piblin Compatibility**: Drop-in replacement with ``import quantiq as piblin``
* **Modern Python 3.12+**: Type-safe with modern syntax and functional programming
* **Non-Linear Fitting**: NLSQ integration for advanced curve fitting
* **Automatic GPU Acceleration**: Transparent device placement without configuration

Performance Targets
-------------------

* **CPU**: 5-10x speedup over piblin baseline
* **GPU**: 50-100x speedup for large datasets
* **Memory**: Efficient batch processing with lazy evaluation
* **Coverage**: >95% test coverage

Quick Start
-----------

Installation::

    pip install quantiq

Basic usage::

    import quantiq

    # Read data
    data = quantiq.read_file('experiment.csv')

    # Create a transform pipeline
    from quantiq.transform import Pipeline, Interpolate1D, Smoothing

    pipeline = Pipeline([
        Interpolate1D(new_x=new_points),
        Smoothing(window_size=5)
    ])

    # Apply transformations
    result = pipeline.apply_to(data)

    # Visualize
    result.visualize()

piblin Compatibility
--------------------

For seamless migration from piblin::

    import quantiq as piblin

    # All your existing piblin code works!
    data = piblin.read_file('experiment.csv')
    # ... rest of your piblin code ...

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/installation
   user_guide/quickstart
   user_guide/concepts
   user_guide/uncertainty
   user_guide/performance
   user_guide/migration

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index
   tutorials/basic_workflow
   tutorials/uncertainty_quantification
   tutorials/custom_transforms
   tutorials/rheological_models

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
   api/data
   api/transform
   api/bayesian
   api/fitting
   api/dataio
   api/backend

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
