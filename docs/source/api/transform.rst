Transformations
===============

Overview
--------

The ``quantiq.transform`` module provides a comprehensive framework for data processing
and manipulation in quantiq. Built on functional programming principles, the transform
system enables composable, reusable, and efficient data pipelines with JAX optimization
support.

The transform architecture is designed around several key principles:

- **Hierarchy-Aware**: Transforms operate at different levels of the data hierarchy
  (Dataset, Measurement, MeasurementSet, etc.). Each transform level knows how to
  process its corresponding data structure while preserving metadata and relationships.

- **Composability**: Transforms can be chained together using the Pipeline pattern,
  allowing complex operations to be built from simple, well-tested components. Pipelines
  themselves are transforms, enabling recursive composition.

- **JAX Optimization**: The framework supports lazy evaluation and JIT compilation for
  high-performance numerical processing. Transform pipelines can be compiled once and
  executed efficiently on GPU/TPU devices.

- **Extensibility**: Custom transforms can be created by subclassing base transform
  classes or by wrapping arbitrary functions using LambdaTransform. The framework
  also supports dynamic transforms that adapt parameters based on input data.

The module includes built-in transforms for common operations like smoothing, interpolation,
baseline correction, normalization, and calculus operations, as well as collection-level
transforms for filtering, splitting, and merging datasets.

Quick Examples
--------------

Basic Transform Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^

Chain multiple transforms together for sequential processing::

    from quantiq.transform import Pipeline
    from quantiq.transform.dataset import SmoothingTransform, NormalizeTransform

    # Create a pipeline
    pipeline = Pipeline([
        SmoothingTransform(window_length=5, polyorder=2),
        NormalizeTransform(method="minmax")
    ])

    # Apply to dataset
    processed_dataset = pipeline.apply(dataset)

Lazy Evaluation for Performance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use lazy pipelines for JIT compilation and deferred execution::

    from quantiq.transform import LazyPipeline
    from quantiq.transform.dataset import InterpolateTransform, BaselineTransform

    # Create lazy pipeline
    lazy_pipeline = LazyPipeline([
        InterpolateTransform(num_points=1000),
        BaselineTransform(method="polynomial", degree=2)
    ])

    # Apply lazily (can be JIT compiled)
    result = lazy_pipeline.apply(dataset)
    processed = result.compute()  # Trigger computation

Custom Lambda Transforms
^^^^^^^^^^^^^^^^^^^^^^^^

Create custom transforms from functions::

    from quantiq.transform import LambdaTransform
    import numpy as np

    def custom_processing(dataset):
        # Custom logic
        new_y = np.log10(dataset.y)
        return dataset.replace(y=new_y)

    # Wrap as transform
    log_transform = LambdaTransform(custom_processing)

    # Use in pipeline
    pipeline = Pipeline([
        log_transform,
        NormalizeTransform()
    ])

See Also
--------

- :doc:`data` - Data structures that transforms operate on
- :doc:`backend` - JAX/NumPy backend for array operations
- `JAX Transformations <https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html>`_ - JAX JIT compilation

API Reference
-------------

Module Contents
^^^^^^^^^^^^^^^

.. automodule:: quantiq.transform
   :members:
   :undoc-members:
   :show-inheritance:

Base Transform Classes
----------------------

.. automodule:: quantiq.transform.base
   :members:
   :undoc-members:
   :show-inheritance:

Pipeline
--------

.. automodule:: quantiq.transform.pipeline
   :members:
   :undoc-members:
   :show-inheritance:

Region-Based Transforms
-----------------------

.. automodule:: quantiq.transform.region
   :members:
   :undoc-members:
   :show-inheritance:

Lambda and Dynamic Transforms
------------------------------

.. automodule:: quantiq.transform.lambda_transform
   :members:
   :undoc-members:
   :show-inheritance:

Dataset Transforms
------------------

Overview
^^^^^^^^

.. automodule:: quantiq.transform.dataset
   :members:
   :undoc-members:
   :show-inheritance:

Smoothing Transforms
^^^^^^^^^^^^^^^^^^^^

.. automodule:: quantiq.transform.dataset.smoothing
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Interpolation Transforms
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: quantiq.transform.dataset.interpolate
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Normalization Transforms
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: quantiq.transform.dataset.normalization
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Baseline Correction Transforms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: quantiq.transform.dataset.baseline
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Calculus Transforms
^^^^^^^^^^^^^^^^^^^^

.. automodule:: quantiq.transform.dataset.calculus
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Measurement Transforms
----------------------

Overview
^^^^^^^^

.. automodule:: quantiq.transform.measurement
   :members:
   :undoc-members:
   :show-inheritance:

Filtering Transforms
^^^^^^^^^^^^^^^^^^^^

.. automodule:: quantiq.transform.measurement.filter
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:
