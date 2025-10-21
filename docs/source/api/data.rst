Data Structures
===============

Overview
--------

The ``quantiq.data`` module provides the core data structures for representing
measurement data in quantiq. It implements a hierarchical system for organizing
experimental data, from individual measurements to complex experimental campaigns
with multiple conditions and replicates.

The module is built around three key concepts:

- **Immutability**: All data structures are immutable by design, ensuring data integrity
  and enabling safe sharing across transforms and analyses. Once created, datasets cannot
  be modified; instead, transformations return new datasets.

- **Type Safety**: Each dataset type (0D, 1D, 2D, 3D) has specific guarantees about its
  structure. This type system enables compile-time validation and better IDE support,
  while maintaining flexibility through metadata.

- **Hierarchical Organization**: Data is organized in a natural hierarchy:
  Dataset → Measurement → MeasurementSet → Experiment → ExperimentSet. This mirrors
  typical experimental workflows where you collect replicate measurements under various
  conditions.

The module also provides comprehensive metadata support with validation, merging utilities,
and Region of Interest (ROI) definitions for selective data processing.

Quick Examples
--------------

Creating a 1D Dataset
^^^^^^^^^^^^^^^^^^^^^

The most common use case is creating a 1D dataset from arrays::

    from quantiq.data.datasets import OneDimensionalDataset
    import numpy as np

    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    dataset = OneDimensionalDataset(x=x, y=y, name="Sine Wave")

Working with Metadata
^^^^^^^^^^^^^^^^^^^^^

Datasets support rich metadata for tracking experimental conditions::

    dataset = OneDimensionalDataset(
        x=x, y=y,
        name="Viscosity vs Shear Rate",
        metadata={
            "temperature": 25.0,
            "sample_id": "ABC123",
            "operator": "Jane Doe",
            "timestamp": "2024-01-15T10:30:00"
        }
    )

    # Access metadata
    temp = dataset.metadata["temperature"]

Building Collections
^^^^^^^^^^^^^^^^^^^^

Organize multiple measurements into collections::

    from quantiq.data.collections import Measurement, MeasurementSet

    # Create a measurement with multiple datasets
    measurement = Measurement(
        datasets=[dataset1, dataset2, dataset3],
        metadata={"replicate": 1, "temperature": 25.0}
    )

    # Group measurements into a set
    measurement_set = MeasurementSet(
        measurements=[meas1, meas2, meas3],
        metadata={"experiment_id": "EXP001"}
    )

See Also
--------

- :doc:`transform` - Data transformation API
- :doc:`dataio` - Reading and writing data files
- `JAX Array API <https://jax.readthedocs.io/en/latest/jax.numpy.html>`_ - Underlying array operations

API Reference
-------------

Module Contents
^^^^^^^^^^^^^^^

.. automodule:: quantiq.data

Datasets
--------

Base Dataset
^^^^^^^^^^^^

.. automodule:: quantiq.data.datasets.base
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Zero-Dimensional Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: quantiq.data.datasets.zero_dimensional
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

One-Dimensional Dataset
^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: quantiq.data.datasets.one_dimensional
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Two-Dimensional Dataset
^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: quantiq.data.datasets.two_dimensional
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Three-Dimensional Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: quantiq.data.datasets.three_dimensional
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Composite Dataset
^^^^^^^^^^^^^^^^^

.. automodule:: quantiq.data.datasets.composite
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Distribution Dataset
^^^^^^^^^^^^^^^^^^^^

.. automodule:: quantiq.data.datasets.distribution
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Histogram Dataset
^^^^^^^^^^^^^^^^^

.. automodule:: quantiq.data.datasets.histogram
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Collections
-----------

Measurement
^^^^^^^^^^^

.. automodule:: quantiq.data.collections.measurement
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

MeasurementSet
^^^^^^^^^^^^^^

.. automodule:: quantiq.data.collections.measurement_set
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

ConsistentMeasurementSet
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: quantiq.data.collections.consistent_measurement_set
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

TabularMeasurementSet
^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: quantiq.data.collections.tabular_measurement_set
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

TidyMeasurementSet
^^^^^^^^^^^^^^^^^^

.. automodule:: quantiq.data.collections.tidy_measurement_set
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Experiment
^^^^^^^^^^

.. automodule:: quantiq.data.collections.experiment
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

ExperimentSet
^^^^^^^^^^^^^

.. automodule:: quantiq.data.collections.experiment_set
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Utilities
---------

Metadata
^^^^^^^^

.. automodule:: quantiq.data.metadata
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Region of Interest (ROI)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: quantiq.data.roi
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:
