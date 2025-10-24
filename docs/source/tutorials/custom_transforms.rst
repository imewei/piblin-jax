Custom Transforms Tutorial
===========================

This tutorial shows you how to create custom data transformations in piblin-jax
that integrate seamlessly with the transform pipeline system. You'll learn:

- How to subclass the base transform classes
- How to implement the ``_apply`` method
- How to add JIT compilation for performance
- How to handle uncertainty propagation
- How to compose transforms into pipelines

Transform Hierarchy
-------------------

Quantiq provides transforms at different hierarchy levels:

**DatasetTransform**
    Operates on individual datasets (1D, 2D, 3D, etc.).
    Most common for signal processing and data manipulation.

**MeasurementTransform**
    Operates on Measurement objects containing multiple datasets.
    Useful for operations across related datasets.

**MeasurementSetTransform**
    Operates on collections of measurements.
    Useful for normalization across replicates.

**ExperimentTransform**
    Operates on entire experiments.
    Useful for global corrections or calibrations.

Basic Custom Transform
----------------------

Create a Simple Scaling Transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's create a transform that scales data by a factor::

    from piblin_jax.transform.base import DatasetTransform
    from piblin_jax.data.datasets import OneDimensionalDataset

    class ScaleTransform(DatasetTransform):
        """Scale dependent variable by a constant factor."""

        def __init__(self, factor: float):
            """Initialize transform.

            Parameters
            ----------
            factor : float
                Scaling factor to apply.
            """
            super().__init__()
            self.factor = factor

        def _apply(self, dataset: OneDimensionalDataset) -> OneDimensionalDataset:
            """Apply scaling transformation.

            Parameters
            ----------
            dataset : OneDimensionalDataset
                Input dataset.

            Returns
            -------
            OneDimensionalDataset
                Scaled dataset.
            """
            # Access internal data arrays
            scaled_y = dataset._dependent_variable_data * self.factor

            # Modify in-place
            dataset._dependent_variable_data = scaled_y

            return dataset

Use the Transform
~~~~~~~~~~~~~~~~~

::

    import numpy as np
    from piblin_jax.data.datasets import OneDimensionalDataset

    # Create dataset
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    dataset = OneDimensionalDataset(
        independent_variable_data=x,
        dependent_variable_data=y
    )

    # Apply transform
    transform = ScaleTransform(factor=2.0)
    scaled = transform.apply_to(dataset, make_copy=True)

    # Check result
    print(f"Original max: {dataset.dependent_variable_data.max():.3f}")
    print(f"Scaled max: {scaled.dependent_variable_data.max():.3f}")

Note: ``make_copy=True`` ensures the original dataset is unchanged.

JIT-Compiled Transforms
-----------------------

For performance-critical operations, use JAX JIT compilation::

    from piblin_jax.transform.base import DatasetTransform
    from piblin_jax.backend import jnp
    from piblin_jax.backend.operations import jit

    class FastNormalize(DatasetTransform):
        """Fast Z-score normalization with JIT compilation."""

        def __init__(self):
            super().__init__()

        @staticmethod
        @jit
        def _compute_normalized(y):
            """JIT-compiled normalization computation."""
            mean = jnp.mean(y)
            std = jnp.std(y)
            return (y - mean) / (std + 1e-10)

        def _apply(self, dataset: OneDimensionalDataset) -> OneDimensionalDataset:
            """Apply normalization."""
            y_internal = dataset._dependent_variable_data
            normalized = self._compute_normalized(y_internal)
            dataset._dependent_variable_data = normalized
            return dataset

The ``@jit`` decorator compiles the function with JAX, providing 3-100x speedups
for array operations. The first call is slow (compilation), but subsequent calls
are very fast.

Advanced Transform with Parameters
-----------------------------------

Moving Average Filter
~~~~~~~~~~~~~~~~~~~~~

Create a configurable moving average filter::

    from piblin_jax.transform.base import DatasetTransform
    from piblin_jax.backend import jnp
    from piblin_jax.backend.operations import jit
    import numpy as np

    class MovingAverageFilter(DatasetTransform):
        """Apply moving average filter to smooth data."""

        def __init__(self, window_size: int = 5, mode: str = 'same'):
            """Initialize filter.

            Parameters
            ----------
            window_size : int, default=5
                Size of the moving average window (must be odd).
            mode : str, default='same'
                Padding mode: 'same', 'valid', or 'full'.
            """
            super().__init__()
            if window_size % 2 == 0:
                raise ValueError("window_size must be odd")
            self.window_size = window_size
            self.mode = mode

        @staticmethod
        @jit
        def _compute_moving_average(y, window):
            """JIT-compiled convolution for moving average."""
            return jnp.convolve(y, window, mode='same')

        def _apply(self, dataset: OneDimensionalDataset) -> OneDimensionalDataset:
            """Apply moving average filter."""
            # Create uniform window
            window = jnp.ones(self.window_size) / self.window_size

            # Apply filter
            y_internal = dataset._dependent_variable_data
            smoothed = self._compute_moving_average(y_internal, window)

            # Handle edges based on mode
            if self.mode == 'valid':
                # Trim edges
                half = self.window_size // 2
                smoothed = smoothed[half:-half]
                x_internal = dataset._independent_variable_data[half:-half]
                dataset._independent_variable_data = x_internal

            dataset._dependent_variable_data = smoothed
            return dataset

Example usage::

    # Apply moving average
    smoother = MovingAverageFilter(window_size=7, mode='same')
    smoothed = smoother.apply_to(dataset, make_copy=True)

    # Plot comparison
    import matplotlib.pyplot as plt
    plt.plot(dataset.independent_variable_data,
             dataset.dependent_variable_data,
             'b-', alpha=0.5, label='Original')
    plt.plot(smoothed.independent_variable_data,
             smoothed.dependent_variable_data,
             'r-', linewidth=2, label='Smoothed')
    plt.legend()
    plt.show()

Transform Pipelines
-------------------

Combine Multiple Transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Chain transforms together using Pipeline::

    from piblin_jax.transform import Pipeline
    from piblin_jax.transform.dataset import (
        Derivative,
        GaussianSmoothing,
        Normalize
    )

    # Create pipeline
    pipeline = Pipeline([
        GaussianSmoothing(sigma=2.0),    # Step 1: Smooth
        Derivative(order=1),              # Step 2: Differentiate
        Normalize(method='minmax')        # Step 3: Normalize
    ])

    # Apply entire pipeline
    result = pipeline.apply_to(dataset, make_copy=True)

The pipeline applies each transform in sequence, automatically handling
copying and data flow.

Conditional Pipeline
~~~~~~~~~~~~~~~~~~~~

Add logic to pipeline execution::

    class ConditionalPipeline:
        """Pipeline with conditional transform application."""

        def __init__(self, transforms, conditions):
            """Initialize conditional pipeline.

            Parameters
            ----------
            transforms : list
                List of transform objects.
            conditions : list of callable
                List of condition functions (dataset -> bool).
            """
            self.transforms = transforms
            self.conditions = conditions

        def apply_to(self, dataset, make_copy=True):
            """Apply pipeline conditionally."""
            if make_copy:
                from copy import deepcopy
                result = deepcopy(dataset)
            else:
                result = dataset

            for transform, condition in zip(self.transforms, self.conditions):
                if condition(result):
                    result = transform.apply_to(result, make_copy=False)

            return result

Example::

    # Define conditions
    def needs_smoothing(dataset):
        """Check if data is noisy."""
        y = dataset.dependent_variable_data
        noise_level = np.std(np.diff(y))
        return noise_level > 0.1

    def needs_normalization(dataset):
        """Check if data needs normalization."""
        y = dataset.dependent_variable_data
        return y.max() - y.min() > 10

    # Create conditional pipeline
    pipeline = ConditionalPipeline(
        transforms=[
            GaussianSmoothing(sigma=2.0),
            Normalize(method='minmax')
        ],
        conditions=[needs_smoothing, needs_normalization]
    )

    result = pipeline.apply_to(dataset)

Multi-Level Transforms
----------------------

Measurement-Level Transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Operate across multiple datasets in a measurement::

    from piblin_jax.transform.base import MeasurementTransform
    from piblin_jax.data.collections import Measurement

    class CrossDatasetNormalize(MeasurementTransform):
        """Normalize all datasets to same scale."""

        def __init__(self):
            super().__init__()

        def _apply(self, measurement: Measurement) -> Measurement:
            """Normalize all datasets together."""
            # Find global min/max across all datasets
            global_min = float('inf')
            global_max = float('-inf')

            for dataset in measurement.datasets:
                if hasattr(dataset, 'dependent_variable_data'):
                    y = dataset.dependent_variable_data
                    global_min = min(global_min, y.min())
                    global_max = max(global_max, y.max())

            # Normalize each dataset
            for dataset in measurement.datasets:
                if hasattr(dataset, 'dependent_variable_data'):
                    y = dataset._dependent_variable_data
                    normalized = (y - global_min) / (global_max - global_min)
                    dataset._dependent_variable_data = normalized

            return measurement

Uncertainty-Aware Transforms
-----------------------------

Propagate Uncertainty
~~~~~~~~~~~~~~~~~~~~~

Transforms can propagate uncertainty through operations::

    class LogTransform(DatasetTransform):
        """Take logarithm of dependent variable."""

        def __init__(self, base: float = 10.0):
            super().__init__()
            self.base = base

        @staticmethod
        @jit
        def _compute_log(y, base):
            """JIT-compiled logarithm."""
            return jnp.log(y) / jnp.log(base)

        def _apply(self, dataset: OneDimensionalDataset) -> OneDimensionalDataset:
            """Apply logarithm transform."""
            y_internal = dataset._dependent_variable_data
            log_y = self._compute_log(y_internal, self.base)
            dataset._dependent_variable_data = log_y
            return dataset

Apply with uncertainty propagation::

    # Create dataset with uncertainty
    dataset_with_unc = dataset.with_uncertainty(
        model=bayesian_model,
        n_samples=1000,
        keep_samples=True
    )

    # Apply transform with uncertainty propagation
    transform = LogTransform(base=10.0)
    result = transform.apply_to(
        dataset_with_unc,
        propagate_uncertainty=True
    )

    # Uncertainty is now propagated through the log transform
    print(f"Result has uncertainty: {result.has_uncertainty}")

Best Practices
--------------

**Immutability**
    Use ``make_copy=True`` (default) to preserve original data. Only use
    ``make_copy=False`` if memory is critical.

**JIT compilation**
    Add ``@jit`` decorator to computational methods for 3-100x speedups.
    First call is slow (compilation), subsequent calls are fast.

**Type hints**
    Use type hints for dataset parameters to improve code clarity::

        def _apply(self, dataset: OneDimensionalDataset) -> OneDimensionalDataset:
            ...

**Error handling**
    Validate inputs in ``__init__`` and raise clear exceptions::

        if window_size < 1:
            raise ValueError("window_size must be >= 1")

**Documentation**
    Provide clear docstrings with Parameters, Returns, and Examples sections.

**Backend agnostic**
    Use ``jnp`` from ``piblin_jax.backend`` instead of direct NumPy/JAX imports
    to ensure compatibility with both backends.

Real-World Example: Baseline Correction
----------------------------------------

Complete Transform Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    from piblin_jax.transform.base import DatasetTransform
    from piblin_jax.backend import jnp
    from piblin_jax.backend.operations import jit
    from scipy.signal import savgol_filter
    import numpy as np

    class BaselineCorrection(DatasetTransform):
        """Remove baseline drift using polynomial fitting."""

        def __init__(self, method: str = 'polynomial', degree: int = 2):
            """Initialize baseline correction.

            Parameters
            ----------
            method : str, default='polynomial'
                Method: 'polynomial', 'linear', or 'savgol'.
            degree : int, default=2
                Polynomial degree (for polynomial method).
            """
            super().__init__()
            self.method = method
            self.degree = degree

        def _apply(self, dataset: OneDimensionalDataset) -> OneDimensionalDataset:
            """Apply baseline correction."""
            x = dataset._independent_variable_data
            y = dataset._dependent_variable_data

            if self.method == 'polynomial':
                # Fit polynomial to data
                coeffs = np.polyfit(x, y, self.degree)
                baseline = np.polyval(coeffs, x)

            elif self.method == 'linear':
                # Simple linear baseline
                slope = (y[-1] - y[0]) / (x[-1] - x[0])
                baseline = y[0] + slope * (x - x[0])

            elif self.method == 'savgol':
                # Savitzky-Golay filter baseline
                window = min(51, len(y) // 4 * 2 + 1)  # Ensure odd
                baseline = savgol_filter(y, window, polyorder=2)

            else:
                raise ValueError(f"Unknown method: {self.method}")

            # Subtract baseline
            corrected = jnp.array(y - baseline)
            dataset._dependent_variable_data = corrected

            return dataset

Usage::

    # Apply baseline correction
    corrector = BaselineCorrection(method='polynomial', degree=2)
    corrected = corrector.apply_to(dataset, make_copy=True)

    # Visualize correction
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(dataset.independent_variable_data,
             dataset.dependent_variable_data, 'b-')
    ax1.set_title('Original Data with Baseline Drift')
    ax1.grid(True, alpha=0.3)

    ax2.plot(corrected.independent_variable_data,
             corrected.dependent_variable_data, 'r-')
    ax2.set_title('Baseline-Corrected Data')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

Next Steps
----------

- See :doc:`../user_guide/concepts` for transform architecture details
- See :doc:`uncertainty_quantification` for uncertainty-aware transforms
- See ``piblin_jax/transform/dataset/`` for built-in transform implementations
- See API docs for complete transform class reference

Tips
----

**Debugging transforms**
    Test your transform on simple synthetic data before applying to real data.

**Performance profiling**
    Use ``%%timeit`` in Jupyter to measure transform performance::

        %%timeit
        transform.apply_to(dataset, make_copy=True)

**Chaining transforms**
    Prefer Pipeline over manual chaining for clarity and error handling.

**Metadata preservation**
    Transforms automatically preserve dataset metadata (conditions, details).
