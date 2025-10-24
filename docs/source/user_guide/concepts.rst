Core Concepts
=============

This guide explains the fundamental concepts and architecture of piblin_jax.

Architecture Overview
---------------------

piblin-jax is built on a layered architecture designed for performance, composability, and ease of use:

Layered Architecture
^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    ┌───────────────────────────────────────────────────────────────────┐
    │                      Application Layer                            │
    │  User Scripts, Notebooks, Custom Analysis, Rheology Applications  │
    └───────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌───────────────────────────────────────────────────────────────────┐
    │                        User API Layer                             │
    ├───────────────────────────────────────────────────────────────────┤
    │  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐            │
    │  │   Data       │  │ Collections │  │   File I/O   │            │
    │  │ Structures   │  │ (Measurement│  │  (Readers/   │            │
    │  │ (Datasets)   │  │      Set)   │  │   Writers)   │            │
    │  └──────────────┘  └─────────────┘  └──────────────┘            │
    └───────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌───────────────────────────────────────────────────────────────────┐
    │                      Core Processing Layer                        │
    ├───────────────────────────────────────────────────────────────────┤
    │  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐            │
    │  │  Transform   │  │   Fitting   │  │   Bayesian   │            │
    │  │   Pipeline   │  │   (NLSQ)    │  │  (NumPyro)   │            │
    │  │   System     │  │             │  │   Models     │            │
    │  └──────────────┘  └─────────────┘  └──────────────┘            │
    └───────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌───────────────────────────────────────────────────────────────────┐
    │                    Backend Abstraction Layer                      │
    ├───────────────────────────────────────────────────────────────────┤
    │  Array Operations  │  Math Functions  │  Backend Detection      │
    │  (JAX/NumPy)       │  (exp, sin, ...)  │  (jax/numpy)           │
    └───────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌───────────────────────────────────────────────────────────────────┐
    │                       Foundation Layer                            │
    ├───────────────────────────────────────────────────────────────────┤
    │  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐            │
    │  │     JAX      │  │   NumPyro   │  │    NumPy     │            │
    │  │ (JIT, GPU,   │  │   (MCMC,    │  │ (Fallback)   │            │
    │  │  Auto-diff)  │  │  Bayesian)  │  │              │            │
    │  └──────────────┘  └─────────────┘  └──────────────┘            │
    └───────────────────────────────────────────────────────────────────┘

Module Organization
^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    piblin_jax/
    ├── data/                    # Data structures
    │   ├── datasets/            # Dataset types (0D, 1D, 2D, 3D)
    │   │   ├── base.py          # BaseDataset abstract class
    │   │   ├── zero_dimensional.py
    │   │   ├── one_dimensional.py
    │   │   ├── two_dimensional.py
    │   │   └── three_dimensional.py
    │   ├── collections/         # Hierarchical organization
    │   │   ├── measurement.py   # Single measurement
    │   │   ├── measurement_set.py
    │   │   └── experiment.py
    │   └── metadata/            # Metadata handling
    │       ├── core.py          # Metadata validation
    │       └── roi.py           # Region of Interest
    │
    ├── transform/               # Data transformation
    │   ├── pipeline.py          # Pipeline composition
    │   ├── base.py              # Transform abstract class
    │   ├── dataset/             # Dataset transforms
    │   │   ├── smoothing.py     # Gaussian smoothing
    │   │   ├── interpolation.py # 1D interpolation
    │   │   └── normalization.py # Data normalization
    │   ├── region/              # Region transforms
    │   │   └── selection.py     # ROI selection
    │   └── lambda_transform.py  # Custom transforms
    │
    ├── bayesian/                # Bayesian inference
    │   ├── base.py              # BayesianModel class
    │   └── rheology/            # Rheological models
    │       ├── power_law.py     # Power law model
    │       ├── arrhenius.py     # Arrhenius model
    │       ├── cross.py         # Cross model
    │       └── carreau_yasuda.py
    │
    ├── fitting/                 # NLSQ curve fitting
    │   ├── curve_fit.py         # Main fitting interface
    │   └── parameter_estimation.py
    │
    ├── dataio/                  # File I/O
    │   ├── readers/             # Data readers
    │   │   ├── csv_reader.py
    │   │   └── registry.py      # Reader registry
    │   └── writers/             # Data writers
    │       ├── csv_writer.py
    │       └── registry.py
    │
    └── backend/                 # Backend abstraction
        ├── detection.py         # Detect JAX/NumPy
        ├── array.py             # Array interface
        └── math.py              # Math functions

Data Flow
^^^^^^^^^

Typical workflow through the layers:

.. code-block:: text

    [CSV File] ──────────────► dataio.readers
                                     │
                                     ▼
                              OneDimensionalDataset
                                     │
                                     ▼
                              Transform Pipeline
                              ┌─────────────────┐
                              │  Smoothing      │
                              │       ↓         │
                              │  Interpolation  │
                              │       ↓         │
                              │  Normalization  │
                              └─────────────────┘
                                     │
                                     ▼
                              Processed Dataset ──► Fitting/Bayesian
                                                         │
                                                         ▼
                                                    Parameters +
                                                   Uncertainties
                                                         │
                                                         ▼
                                              [Results/Visualization]

Each transform in the pipeline:

1. **Receives**: Immutable dataset
2. **Applies**: JAX-optimized operations via backend layer
3. **Returns**: New immutable dataset (original unchanged)
4. **Metadata**: Automatically tracks transformation history

Key Design Principles
---------------------

1. **Immutability**: Data structures are immutable by default
2. **Composability**: Transforms can be composed into pipelines
3. **Type Safety**: Comprehensive type hints throughout
4. **Performance**: JAX-powered automatic optimization
5. **Compatibility**: 100% backward compatible with piblin

Data Structures
---------------

Datasets
^^^^^^^^

Datasets are the core data containers in piblin_jax. They are immutable and
type-specific:

**Zero-Dimensional Dataset**
  Scalar values with uncertainty::

    from piblin_jax.data import ZeroDimensionalDataset
    temperature = ZeroDimensionalDataset(value=25.0, uncertainty=0.5)

**One-Dimensional Dataset**
  Arrays of (x, y) data::

    from piblin_jax.data import OneDimensionalDataset
    dataset = OneDimensionalDataset(x=x_values, y=y_values)

**Two-Dimensional Dataset**
  Gridded data (x, y, z)::

    from piblin_jax.data import TwoDimensionalDataset
    surface = TwoDimensionalDataset(x=x, y=y, z=z_grid)

**Three-Dimensional Dataset**
  Volumetric data::

    from piblin_jax.data import ThreeDimensionalDataset
    volume = ThreeDimensionalDataset(x=x, y=y, z=z, data=data_3d)

**Composite Datasets**
  Multiple related datasets::

    from piblin_jax.data import CompositeDataset
    composite = CompositeDataset(datasets={'temp': temp_ds, 'pressure': pressure_ds})

Collections
^^^^^^^^^^^

Collections organize multiple datasets hierarchically:

**Measurement**
  Related datasets from a single experimental run::

    from piblin_jax.data.collections import Measurement
    measurement = Measurement(name='Trial 1')
    measurement.add_dataset('temperature', temp_dataset)

**MeasurementSet**
  Multiple related measurements::

    from piblin_jax.data.collections import MeasurementSet
    measurement_set = MeasurementSet(name='Daily Experiments')

**Experiment**
  Hierarchical organization of measurements::

    from piblin_jax.data.collections import Experiment
    experiment = Experiment(name='Rheology Study')

Metadata System
^^^^^^^^^^^^^^^

All data structures support rich metadata::

    dataset = OneDimensionalDataset(
        x=x, y=y,
        metadata={
            'sample_id': 'ABC123',
            'temperature': 25.0,
            'operator': 'Alice',
            'timestamp': '2025-10-19T12:00:00'
        }
    )

Transforms
----------

Transform Types
^^^^^^^^^^^^^^^

**Dataset Transforms**
  Operate on individual datasets:

  - ``GaussianSmoothing``: Smooth noisy data
  - ``Interpolate1D``: Interpolate to new x-values
  - ``Normalization``: Normalize data
  - ``Derivative``: Numerical differentiation
  - ``Integral``: Numerical integration

**Region Transforms**
  Select or modify regions:

  - ``SelectRegion``: Extract data within bounds
  - ``RemoveRegion``: Remove data within bounds

**Measurement Transforms**
  Operate on measurements:

  - ``Filter``: Filter measurements by criteria

**Lambda Transforms**
  Custom transformations::

    from piblin_jax.transform import LambdaTransform
    custom = LambdaTransform(func=lambda ds: modify(ds))

Transform Pipeline
^^^^^^^^^^^^^^^^^^

Compose transforms into reusable pipelines::

    from piblin_jax.transform import Pipeline

    pipeline = Pipeline([
        GaussianSmoothing(sigma=2.0),
        Interpolate1D(new_x=new_points),
        Normalization(method='minmax')
    ])

    result = pipeline.apply_to(dataset)

Pipelines are:

- **Reusable**: Apply to multiple datasets
- **Composable**: Nest pipelines within pipelines
- **Serializable**: Save and load pipeline configurations
- **Optimized**: JAX automatically optimizes execution

Backend Abstraction
-------------------

piblin-jax abstracts numerical operations through a backend layer:

.. code-block:: python

    from piblin_jax.backend import get_backend, array, exp, sin

    # Get current backend
    backend = get_backend()  # 'jax' or 'numpy'

    # Backend-agnostic operations
    x = array([1.0, 2.0, 3.0])
    y = exp(sin(x))

This allows:

- **Transparent GPU acceleration** when JAX is available
- **Fallback to NumPy** for compatibility
- **Consistent API** regardless of backend

Bayesian Inference
------------------

piblin-jax integrates NumPyro for Bayesian parameter estimation:

Model Structure
^^^^^^^^^^^^^^^

All Bayesian models inherit from ``BayesianModel``::

    from piblin_jax.bayesian import BayesianModel

    class MyModel(BayesianModel):
        def model(self, x, y=None):
            # Define priors
            param1 = numpyro.sample('param1', dist.Normal(0, 1))

            # Define likelihood
            y_pred = param1 * x
            numpyro.sample('obs', dist.Normal(y_pred, 0.1), obs=y)

Built-in Models
^^^^^^^^^^^^^^^

- **PowerLawModel**: :math:`\\eta = K \\dot{\\gamma}^{n-1}`
- **ArrheniusModel**: :math:`\\eta = A \\exp(E_a / RT)`
- **CrossModel**: Flow curves with plateaus
- **CarreauYasudaModel**: Complex rheological behavior

See :doc:`uncertainty` for details.

Uncertainty Propagation
^^^^^^^^^^^^^^^^^^^^^^^

Uncertainties propagate through transforms automatically when using
Bayesian models.

piblin Compatibility
--------------------

piblin-jax maintains 100% API compatibility with piblin:

Compatibility Layer
^^^^^^^^^^^^^^^^^^^

::

    import piblin_jax as piblin

    # All piblin code works unchanged
    data = piblin.read_file('data.csv')
    # ... existing piblin workflow ...

This allows gradual migration and A/B testing of performance.

Performance Optimization
------------------------

JAX Integration
^^^^^^^^^^^^^^^

piblin-jax leverages JAX for:

- **JIT Compilation**: Automatic optimization
- **Vectorization**: SIMD operations
- **GPU Acceleration**: Transparent GPU usage
- **Auto-differentiation**: For Bayesian inference

Lazy Evaluation
^^^^^^^^^^^^^^^

Operations are lazy when possible, deferring computation until needed.

Batching
^^^^^^^^

Process multiple datasets efficiently using collections.

Type System
-----------

piblin-jax is fully typed with comprehensive type hints::

    from typing import Optional
    from piblin_jax.data import OneDimensionalDataset

    def process_data(
        dataset: OneDimensionalDataset,
        sigma: float = 1.0,
        normalize: bool = True
    ) -> OneDimensionalDataset:
        ...

This enables:

- **IDE autocomplete**
- **Static type checking** with mypy
- **Better documentation**
- **Fewer runtime errors**

Best Practices
--------------

1. **Use Pipelines**: Compose transforms for reusability
2. **Leverage Collections**: Organize related datasets
3. **Add Metadata**: Document your data
4. **Type Annotations**: Use type hints in custom code
5. **Immutability**: Don't modify data in-place
6. **GPU Wisely**: Use GPU for large datasets (>10k points)

Example: Complete Workflow
---------------------------

::

    import piblin_jax
    from piblin_jax.transform import Pipeline, GaussianSmoothing, Normalization
    from piblin_jax.data.collections import MeasurementSet

    # Load data
    datasets = [piblin_jax.read_file(f'sample_{i}.csv') for i in range(10)]

    # Create measurement set
    ms = MeasurementSet.from_datasets(datasets)

    # Define pipeline
    pipeline = Pipeline([
        GaussianSmoothing(sigma=2.0),
        Normalization(method='minmax')
    ])

    # Process all datasets
    processed = ms.apply_transform(pipeline)

    # Bayesian analysis
    from piblin_jax.bayesian import PowerLawModel
    model = PowerLawModel()

    for measurement in processed.measurements:
        ds = measurement.get_dataset('flow_curve')
        model.fit(ds.x, ds.y)
        print(model.summary())

Next Steps
----------

- **Hands-on Tutorial**: :doc:`../tutorials/basic_workflow`
- **Uncertainty Quantification**: :doc:`uncertainty`
- **Performance Tips**: :doc:`performance`
- **API Reference**: :doc:`../api/index`
