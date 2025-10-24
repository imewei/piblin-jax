Advanced Pipeline Composition
==============================

This tutorial covers advanced patterns for composing and reusing transform pipelines
in piblin_jax. You'll learn how to build complex data processing workflows that are
modular, maintainable, and efficient.

.. contents:: Table of Contents
   :local:
   :depth: 2

Prerequisites
-------------

This tutorial assumes you're familiar with:

- :doc:`basic_workflow` - Basic piblin-jax usage
- :doc:`custom_transforms` - Creating custom transforms
- Pipeline basics from the :doc:`../user_guide/concepts`

Overview
--------

Complex data analysis often requires sophisticated processing pipelines with:

- **Conditional logic** - Apply different transforms based on data properties
- **Parallel processing** - Split data, process independently, merge results
- **Dynamic parameters** - Adjust transform settings based on intermediate results
- **Error handling** - Gracefully handle edge cases and failures
- **Reusable components** - Build libraries of specialized pipelines

This tutorial demonstrates these patterns with practical examples.

Conditional Pipelines
---------------------

Apply Different Transforms Based on Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes you need to apply different processing based on data characteristics:

.. code-block:: python

    from piblin_jax.data.datasets import OneDimensionalDataset
    from piblin_jax.transform.pipeline import Pipeline
    from piblin_jax.transform.dataset.smoothing import GaussianSmoothing
    from piblin_jax.transform.dataset.normalization import MinMaxNormalization
    import numpy as np

    def create_adaptive_pipeline(dataset):
        """
        Create a pipeline that adapts to data noise level.

        High noise → aggressive smoothing
        Low noise → minimal smoothing
        """
        # Estimate noise level
        diff = np.diff(dataset.y)
        noise_std = np.std(diff)

        if noise_std > 0.5:
            # High noise: aggressive smoothing
            return Pipeline([
                GaussianSmoothing(sigma=5.0),
                MinMaxNormalization()
            ])
        else:
            # Low noise: light smoothing
            return Pipeline([
                GaussianSmoothing(sigma=1.0),
                MinMaxNormalization()
            ])

    # Usage
    noisy_data = OneDimensionalDataset(x, noisy_y)
    pipeline = create_adaptive_pipeline(noisy_data)
    result = pipeline.apply(noisy_data)

Branching Logic with Factory Pattern
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use factory functions to select pipelines based on conditions:

.. code-block:: python

    from typing import Literal

    def pipeline_factory(
        data_type: Literal["rheology", "spectroscopy", "timeseries"]
    ) -> Pipeline:
        """Factory function for domain-specific pipelines."""

        if data_type == "rheology":
            # Rheological data: log-scale, smooth, normalize
            from piblin_jax.transform.dataset import LogTransform
            return Pipeline([
                LogTransform(base=10),
                GaussianSmoothing(sigma=2.0),
                MinMaxNormalization()
            ])

        elif data_type == "spectroscopy":
            # Spectroscopy: baseline correction, normalization
            from piblin_jax.transform.dataset.baseline import BaselineCorrection
            return Pipeline([
                BaselineCorrection(method="polynomial", degree=2),
                MinMaxNormalization()
            ])

        else:  # timeseries
            # Time series: smoothing, derivative
            from piblin_jax.transform.dataset.calculus import Derivative
            return Pipeline([
                GaussianSmoothing(sigma=3.0),
                Derivative(order=1)
            ])

    # Usage
    rheology_pipeline = pipeline_factory("rheology")
    result = rheology_pipeline.apply(dataset)

Parallel Pipeline Patterns
---------------------------

Split-Process-Merge Pattern
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Process different aspects of data in parallel, then combine results:

.. code-block:: python

    from piblin_jax.data.datasets import CompositeDataset

    def parallel_analysis(dataset):
        """
        Analyze data using parallel pipelines, combine results.

        Pipeline 1: Smooth for trend analysis
        Pipeline 2: Derivative for feature detection
        Pipeline 3: FFT for frequency analysis
        """
        # Define parallel pipelines
        trend_pipeline = Pipeline([
            GaussianSmoothing(sigma=10.0),
            MinMaxNormalization()
        ])

        feature_pipeline = Pipeline([
            GaussianSmoothing(sigma=2.0),
            Derivative(order=1)
        ])

        # Apply pipelines in parallel (conceptually)
        trend_data = trend_pipeline.apply(dataset)
        feature_data = feature_pipeline.apply(dataset)

        # Combine results (implementation depends on use case)
        return {
            "trend": trend_data,
            "features": feature_data,
            "original": dataset
        }

Multi-Stage Pipelines
^^^^^^^^^^^^^^^^^^^^^^

Build pipelines with distinct processing stages:

.. code-block:: python

    class MultiStagePipeline:
        """
        Pipeline with distinct preprocessing, analysis, and postprocessing stages.

        Each stage can be modified independently for flexibility.
        """

        def __init__(self):
            # Stage 1: Preprocessing (cleaning, outlier removal)
            self.preprocessing = Pipeline([
                OutlierRemoval(threshold=3.0),
                GaussianSmoothing(sigma=2.0)
            ])

            # Stage 2: Analysis (feature extraction, transforms)
            self.analysis = Pipeline([
                Derivative(order=1),
                MinMaxNormalization()
            ])

            # Stage 3: Postprocessing (final cleanup)
            self.postprocessing = Pipeline([
                GaussianSmoothing(sigma=1.0)
            ])

        def apply(self, dataset):
            """Apply all stages sequentially."""
            stage1 = self.preprocessing.apply(dataset)
            stage2 = self.analysis.apply(stage1)
            stage3 = self.postprocessing.apply(stage2)
            return stage3

        def apply_up_to_stage(self, dataset, stage: int):
            """Apply only up to specified stage (1, 2, or 3)."""
            if stage >= 1:
                result = self.preprocessing.apply(dataset)
            if stage >= 2:
                result = self.analysis.apply(result)
            if stage >= 3:
                result = self.postprocessing.apply(result)
            return result

Dynamic Pipeline Configuration
-------------------------------

Adjust Parameters Based on Intermediate Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def iterative_smoothing(dataset, target_noise_level=0.01, max_iterations=10):
        """
        Iteratively smooth until noise level reaches target.

        This demonstrates dynamic parameter adjustment based on
        intermediate results.
        """
        current = dataset
        iteration = 0

        while iteration < max_iterations:
            # Measure current noise
            noise = np.std(np.diff(current.y))

            if noise <= target_noise_level:
                print(f"Target noise level reached in {iteration} iterations")
                break

            # Adjust smoothing strength based on current noise
            sigma = max(1.0, noise * 2.0)

            # Apply smoothing
            pipeline = Pipeline([GaussianSmoothing(sigma=sigma)])
            current = pipeline.apply(current)

            iteration += 1

        return current

Data-Driven Transform Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def optimize_smoothing_window(dataset, quality_metric):
        """
        Find optimal smoothing window size by testing multiple options.

        Parameters
        ----------
        dataset : OneDimensionalDataset
            Input data
        quality_metric : callable
            Function that scores quality (higher is better)

        Returns
        -------
        OneDimensionalDataset
            Best smoothed version
        """
        window_sizes = [3, 5, 7, 11, 15, 21]
        results = []

        for window in window_sizes:
            pipeline = Pipeline([GaussianSmoothing(sigma=window/3.0)])
            smoothed = pipeline.apply(dataset)
            score = quality_metric(smoothed)
            results.append((score, smoothed, window))

        # Return best result
        best_score, best_result, best_window = max(results, key=lambda x: x[0])
        print(f"Best window size: {best_window} (score: {best_score:.3f})")
        return best_result

Pipeline Reusability Patterns
------------------------------

Building Pipeline Libraries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create reusable pipeline components for your domain:

.. code-block:: python

    class RheologyPipelines:
        """Library of standard pipelines for rheological data analysis."""

        @staticmethod
        def flow_curve_processing():
            """Standard flow curve processing pipeline."""
            return Pipeline([
                OutlierRemoval(threshold=3.0),
                LogTransform(base=10),
                GaussianSmoothing(sigma=2.0)
            ])

        @staticmethod
        def oscillatory_analysis():
            """Pipeline for oscillatory rheology data."""
            return Pipeline([
                BaselineCorrection(method="linear"),
                MinMaxNormalization(),
                GaussianSmoothing(sigma=1.5)
            ])

        @staticmethod
        def temperature_sweep():
            """Pipeline for temperature sweep experiments."""
            return Pipeline([
                GaussianSmoothing(sigma=3.0),
                Derivative(order=1),  # Find transitions
                MinMaxNormalization()
            ])

    # Usage
    flow_data = read_rheology_file("flow_curve.csv")
    pipeline = RheologyPipelines.flow_curve_processing()
    processed = pipeline.apply(flow_data)

Composable Pipeline Builders
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use builder pattern for flexible pipeline construction:

.. code-block:: python

    class PipelineBuilder:
        """Fluent interface for building pipelines."""

        def __init__(self):
            self.transforms = []

        def smooth(self, sigma=2.0):
            """Add smoothing step."""
            self.transforms.append(GaussianSmoothing(sigma=sigma))
            return self

        def normalize(self, method="minmax"):
            """Add normalization step."""
            if method == "minmax":
                self.transforms.append(MinMaxNormalization())
            else:
                self.transforms.append(ZScoreNormalization())
            return self

        def differentiate(self, order=1):
            """Add derivative step."""
            self.transforms.append(Derivative(order=order))
            return self

        def build(self):
            """Construct the pipeline."""
            return Pipeline(self.transforms)

    # Usage with fluent interface
    pipeline = (PipelineBuilder()
                .smooth(sigma=3.0)
                .normalize(method="minmax")
                .differentiate(order=1)
                .build())

Error Handling in Pipelines
----------------------------

Robust Pipeline Execution
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from typing import Optional

    def robust_pipeline_apply(
        pipeline: Pipeline,
        dataset: OneDimensionalDataset,
        fallback_pipeline: Optional[Pipeline] = None
    ) -> OneDimensionalDataset:
        """
        Apply pipeline with error handling and fallback.

        Parameters
        ----------
        pipeline : Pipeline
            Primary pipeline to try
        dataset : OneDimensionalDataset
            Input data
        fallback_pipeline : Pipeline, optional
            Fallback pipeline if primary fails

        Returns
        -------
        OneDimensionalDataset
            Processed dataset
        """
        try:
            return pipeline.apply(dataset)
        except Exception as e:
            print(f"Primary pipeline failed: {e}")

            if fallback_pipeline is not None:
                print("Trying fallback pipeline...")
                return fallback_pipeline.apply(dataset)
            else:
                print("No fallback available, returning original data")
                return dataset

Validation and Debugging
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    class ValidatedPipeline:
        """Pipeline wrapper with validation and debugging."""

        def __init__(self, pipeline: Pipeline, validators=None):
            self.pipeline = pipeline
            self.validators = validators or []
            self.debug = False

        def enable_debug(self):
            """Enable debug output."""
            self.debug = True

        def apply(self, dataset):
            """Apply pipeline with validation."""
            current = dataset

            for i, transform in enumerate(self.pipeline.transforms):
                if self.debug:
                    print(f"\nStep {i+1}: {transform.__class__.__name__}")
                    print(f"  Input shape: {current.x.shape}")

                # Apply transform
                current = transform.apply(current)

                if self.debug:
                    print(f"  Output shape: {current.x.shape}")
                    print(f"  Output range: [{current.y.min():.2f}, {current.y.max():.2f}]")

                # Run validators
                for validator in self.validators:
                    is_valid, message = validator(current)
                    if not is_valid:
                        raise ValueError(f"Validation failed: {message}")

            return current

    # Example validator
    def no_nans_validator(dataset):
        """Check for NaN values."""
        has_nans = np.any(np.isnan(dataset.y))
        if has_nans:
            return False, "Dataset contains NaN values"
        return True, "OK"

    # Usage
    pipeline = Pipeline([...])
    validated = ValidatedPipeline(pipeline, validators=[no_nans_validator])
    validated.enable_debug()
    result = validated.apply(dataset)

Best Practices
--------------

1. **Keep Pipelines Focused**

   - Each pipeline should have a clear, single purpose
   - Use composition to build complex workflows from simple pipelines
   - Avoid monolithic pipelines with too many steps

2. **Use Factory Functions**

   - Centralize pipeline creation logic
   - Make it easy to create standard pipelines consistently
   - Enable runtime pipeline selection

3. **Document Pipeline Behavior**

   - Add docstrings explaining what each pipeline does
   - Document expected input/output characteristics
   - Include usage examples

4. **Test Pipelines**

   - Create unit tests for custom pipelines
   - Test edge cases (empty data, NaNs, extreme values)
   - Validate intermediate results

5. **Enable Debugging**

   - Add logging or print statements for complex pipelines
   - Validate outputs at each step
   - Use try-except for graceful error handling

Complete Example
----------------

Here's a complete example bringing together multiple concepts:

.. code-block:: python

    from typing import Literal
    import numpy as np
    from piblin_jax.data.datasets import OneDimensionalDataset
    from piblin_jax.transform.pipeline import Pipeline
    from piblin_jax.transform.dataset import (
        GaussianSmoothing,
        MinMaxNormalization,
        Derivative,
        BaselineCorrection
    )

    class AdaptiveDataProcessor:
        """
        Adaptive data processor with automatic pipeline selection.

        Features:
        - Automatic data type detection
        - Noise-adaptive smoothing
        - Error handling with fallback
        - Debug mode for troubleshooting
        """

        def __init__(self, debug=False):
            self.debug = debug

        def detect_data_type(self, dataset) -> Literal["smooth", "noisy", "baseline"]:
            """Detect data characteristics."""
            # Compute noise estimate
            diff = np.diff(dataset.y)
            noise_std = np.std(diff)

            # Detect baseline drift
            trend = np.polyfit(dataset.x, dataset.y, 1)
            has_baseline = abs(trend[0]) > 0.01

            if noise_std > 0.5:
                return "noisy"
            elif has_baseline:
                return "baseline"
            else:
                return "smooth"

        def create_pipeline(self, data_type: str, noise_level: float) -> Pipeline:
            """Create appropriate pipeline for data type."""
            if data_type == "noisy":
                # Aggressive smoothing
                sigma = min(10.0, noise_level * 5.0)
                return Pipeline([
                    GaussianSmoothing(sigma=sigma),
                    MinMaxNormalization()
                ])

            elif data_type == "baseline":
                # Baseline correction
                return Pipeline([
                    BaselineCorrection(method="polynomial", degree=2),
                    GaussianSmoothing(sigma=2.0),
                    MinMaxNormalization()
                ])

            else:  # smooth
                # Light processing
                return Pipeline([
                    GaussianSmoothing(sigma=1.0),
                    MinMaxNormalization()
                ])

        def process(self, dataset: OneDimensionalDataset) -> OneDimensionalDataset:
            """Process dataset with adaptive pipeline."""
            # Detect data characteristics
            data_type = self.detect_data_type(dataset)
            noise_level = np.std(np.diff(dataset.y))

            if self.debug:
                print(f"Detected data type: {data_type}")
                print(f"Noise level: {noise_level:.3f}")

            # Create and apply appropriate pipeline
            pipeline = self.create_pipeline(data_type, noise_level)

            if self.debug:
                print(f"Pipeline: {[t.__class__.__name__ for t in pipeline.transforms]}")

            try:
                result = pipeline.apply(dataset)
                if self.debug:
                    print("✓ Processing successful")
                return result

            except Exception as e:
                print(f"Error in processing: {e}")
                print("Falling back to minimal processing...")

                fallback = Pipeline([GaussianSmoothing(sigma=1.0)])
                return fallback.apply(dataset)

    # Usage
    processor = AdaptiveDataProcessor(debug=True)
    result = processor.process(my_dataset)

Next Steps
----------

- Explore :doc:`gpu_acceleration` for performance optimization
- Learn about :doc:`uncertainty_quantification` for Bayesian workflows
- See :doc:`../api/transform` for all available transforms

.. seealso::

   - :doc:`basic_workflow` - Introduction to quantiq
   - :doc:`custom_transforms` - Creating custom transforms
   - :doc:`rheological_models` - Domain-specific examples
