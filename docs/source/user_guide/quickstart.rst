Quick Start Guide
=================

This guide will help you get started with quantiq in just a few minutes.

Installation Reminder
---------------------

If you haven't installed quantiq yet, see the :doc:`installation` guide.

Quick install::

    pip install quantiq

Your First quantiq Program
---------------------------

Let's create a simple program that loads data, applies transformations,
and visualizes the results.

Loading Data
^^^^^^^^^^^^

quantiq provides several ways to create and load data::

    import quantiq
    import numpy as np

    # Create a 1D dataset from arrays
    from quantiq.data import OneDimensionalDataset

    x = np.linspace(0, 10, 100)
    y = np.sin(x) + 0.1 * np.random.randn(100)  # Noisy sine wave

    dataset = OneDimensionalDataset(x=x, y=y)

Alternatively, load from a file::

    # Load from CSV file
    dataset = quantiq.read_file('experiment.csv')

Applying Transformations
^^^^^^^^^^^^^^^^^^^^^^^^^

quantiq provides a rich set of transformations for data processing::

    from quantiq.transform import GaussianSmoothing, Normalization

    # Smooth the data
    smoother = GaussianSmoothing(sigma=2.0)
    smoothed = smoother.apply_to(dataset)

    # Normalize the smoothed data
    normalizer = Normalization(method='minmax')
    normalized = normalizer.apply_to(smoothed)

Building a Pipeline
^^^^^^^^^^^^^^^^^^^

For multiple transformations, use a pipeline::

    from quantiq.transform import Pipeline, GaussianSmoothing, Normalization

    pipeline = Pipeline([
        GaussianSmoothing(sigma=2.0),
        Normalization(method='minmax')
    ])

    result = pipeline.apply_to(dataset)

Pipelines are reusable and can be applied to multiple datasets::

    dataset1 = quantiq.read_file('experiment1.csv')
    dataset2 = quantiq.read_file('experiment2.csv')

    result1 = pipeline.apply_to(dataset1)
    result2 = pipeline.apply_to(dataset2)

Visualization
^^^^^^^^^^^^^

quantiq datasets have built-in visualization capabilities::

    import matplotlib.pyplot as plt

    # Simple visualization
    dataset.visualize()
    plt.show()

    # More control
    fig, ax = plt.subplots()
    dataset.visualize(ax=ax, label='Original')
    smoothed.visualize(ax=ax, label='Smoothed')
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()

Complete Example
^^^^^^^^^^^^^^^^

Here's a complete example putting it all together::

    import quantiq
    import numpy as np
    import matplotlib.pyplot as plt
    from quantiq.data import OneDimensionalDataset
    from quantiq.transform import Pipeline, GaussianSmoothing, Normalization

    # Generate noisy data
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + 0.1 * np.random.randn(100)

    # Create dataset
    dataset = OneDimensionalDataset(x=x, y=y)

    # Create and apply pipeline
    pipeline = Pipeline([
        GaussianSmoothing(sigma=1.5),
        Normalization(method='minmax')
    ])
    result = pipeline.apply_to(dataset)

    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    dataset.visualize(ax=ax1, label='Original')
    ax1.set_title('Original Data')
    ax1.legend()

    result.visualize(ax=ax2, label='Processed', color='red')
    ax2.set_title('Smoothed & Normalized')
    ax2.legend()

    plt.tight_layout()
    plt.show()

Working with Collections
-------------------------

quantiq provides hierarchical data structures for organizing multiple datasets.

Measurements
^^^^^^^^^^^^

A Measurement contains multiple related datasets::

    from quantiq.data.collections import Measurement

    # Create measurement from datasets
    measurement = Measurement(name='Trial 1')
    measurement.add_dataset('temperature', temp_dataset)
    measurement.add_dataset('pressure', pressure_dataset)
    measurement.add_dataset('flow_rate', flow_dataset)

    # Access datasets
    temp = measurement.get_dataset('temperature')

Measurement Sets
^^^^^^^^^^^^^^^^

Group related measurements together::

    from quantiq.data.collections import MeasurementSet

    measurement_set = MeasurementSet(name='Daily Experiments')
    measurement_set.add_measurement(measurement1)
    measurement_set.add_measurement(measurement2)
    measurement_set.add_measurement(measurement3)

    # Apply transformations to all measurements
    pipeline = Pipeline([GaussianSmoothing(sigma=2.0)])
    processed_set = measurement_set.apply_transform(pipeline)

Bayesian Parameter Estimation
------------------------------

quantiq includes built-in Bayesian models for parameter estimation with
uncertainty quantification.

Basic Fitting
^^^^^^^^^^^^^

Fit a power-law model to rheological data::

    from quantiq.bayesian import PowerLawModel
    import numpy as np

    # Experimental data
    shear_rate = np.array([0.1, 1.0, 10.0, 100.0])
    viscosity = np.array([50.0, 15.8, 5.0, 1.58])

    # Create and fit model
    model = PowerLawModel(n_samples=2000, n_warmup=1000)
    model.fit(shear_rate, viscosity)

    # Get parameter estimates
    summary = model.summary()
    print(summary)

The model will print parameter estimates with credible intervals::

    Parameter Estimates:
    K (consistency):    5.02 [4.89, 5.15] (95% CI)
    n (flow index):     0.60 [0.58, 0.62] (95% CI)

Making Predictions
^^^^^^^^^^^^^^^^^^

Use the fitted model to make predictions with uncertainty::

    # Predict at new points
    new_shear_rate = np.logspace(-1, 2, 50)
    predictions = model.predict(new_shear_rate)

    # Visualize fit with uncertainty bands
    model.plot_fit(shear_rate, viscosity, show_uncertainty=True)

Available Models
^^^^^^^^^^^^^^^^

quantiq provides several built-in rheological models:

- **PowerLawModel**: :math:`\\eta = K \\dot{\\gamma}^{n-1}`
- **ArrheniusModel**: :math:`\\eta = A \\exp(E_a / RT)`
- **CrossModel**: Flow curves with plateaus
- **CarreauYasudaModel**: Complex flow behavior

See :doc:`../tutorials/rheological_models` for detailed examples.

piblin Compatibility
--------------------

Migrating from piblin? quantiq is 100% backward compatible::

    # Just change your import
    import quantiq as piblin

    # All your existing piblin code works!
    data = piblin.read_file('experiment.csv')
    # ... rest of your piblin code ...

This allows you to migrate gradually and take advantage of quantiq's
performance improvements without rewriting your code.

Performance Tips
----------------

JAX Backend
^^^^^^^^^^^

quantiq automatically uses JAX for numerical operations, providing
significant speedups::

    from quantiq.backend import get_backend

    # Check which backend is being used
    print(f"Backend: {get_backend()}")  # 'jax' (default)

GPU Acceleration
^^^^^^^^^^^^^^^^

For large datasets on Linux with CUDA 12+, quantiq can leverage GPU acceleration::

    # Install GPU support (Linux only, CUDA 12+)
    # pip install quantiq[gpu-cuda]

    # JAX automatically uses GPU when available
    # No code changes needed!

**Platform Constraints:**

- GPU acceleration requires Linux with CUDA 12+
- macOS and Windows users benefit from 5-10x CPU speedup via JAX
- Linux with NVIDIA GPU provides 50-100x speedup

Batch Processing
^^^^^^^^^^^^^^^^

Process multiple datasets efficiently::

    # Instead of a loop
    results = []
    for dataset in datasets:
        result = pipeline.apply_to(dataset)
        results.append(result)

    # Use measurement sets for batch processing
    measurement_set = MeasurementSet.from_datasets(datasets)
    processed_set = measurement_set.apply_transform(pipeline)

Next Steps
----------

Now that you've learned the basics, explore:

- :doc:`concepts` - Core concepts and architecture
- :doc:`../tutorials/basic_workflow` - Complete workflow examples
- :doc:`../tutorials/uncertainty_quantification` - Bayesian inference
- :doc:`../api/data` - Data structures API reference
- :doc:`../api/transform` - Transformations API reference

For specific topics:

- **Performance Optimization**: :doc:`performance`
- **Migrating from piblin**: :doc:`migration`
- **Uncertainty Quantification**: :doc:`uncertainty`
- **Custom Transforms**: :doc:`../tutorials/custom_transforms`

Getting Help
------------

If you encounter issues or have questions:

- Check the :doc:`../api/index` for detailed API documentation
- Browse :doc:`../tutorials/index` for more examples
- Search `GitHub Issues <https://github.com/quantiq/quantiq/issues>`_
- Ask on `GitHub Discussions <https://github.com/quantiq/quantiq/discussions>`_
