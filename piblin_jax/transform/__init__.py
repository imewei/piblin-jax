"""
Transform system for piblin-jax.

This module provides the transform framework for data processing:
- Base transform classes for each hierarchy level
- Pipeline composition for sequential transforms
- Lazy evaluation for JAX optimization
- JIT compilation support
- Region-based transforms for selective processing
- Lambda transforms for user-defined functions
- Dynamic transforms for data-driven parameters
- Core dataset transforms (interpolation, smoothing, normalization, etc.)
- Collection-level transforms (filtering, splitting, merging)

Hierarchy:
- Transform: Abstract base class
- DatasetTransform: Operates on Dataset objects
- MeasurementTransform: Operates on Measurement objects
- MeasurementSetTransform: Operates on MeasurementSet objects
- ExperimentTransform: Operates on Experiment objects
- ExperimentSetTransform: Operates on ExperimentSet objects

Pipeline:
- Pipeline: Sequential composition of transforms
- LazyPipeline: Pipeline with lazy evaluation

Region-Based:
- RegionTransform: Base class for region-based transforms
- RegionMultiplyTransform: Example region-based transform

Lambda and Dynamic:
- LambdaTransform: Wrap arbitrary functions as transforms
- DynamicTransform: Base class for data-driven transforms
- AutoScaleTransform: Automatic data scaling
- AutoBaselineTransform: Automatic baseline correction

Dataset Transforms:

- dataset: Module containing core dataset-level transforms
  (Interpolation, smoothing, baseline correction, normalization, calculus)

Measurement Transforms:

- measurement: Module containing collection-level transforms
  (FilterDatasets, FilterMeasurements, SplitByRegion, MergeReplicates)
"""

# Import measurement transform submodule
# Import dataset transform submodule
from . import dataset, measurement
from .base import (
    DatasetTransform,
    ExperimentSetTransform,
    ExperimentTransform,
    MeasurementSetTransform,
    MeasurementTransform,
    Transform,
    jit_transform,
)
from .lambda_transform import (
    AutoBaselineTransform,
    AutoScaleTransform,
    DynamicTransform,
    LambdaTransform,
)
from .pipeline import LazyPipeline, LazyResult, Pipeline
from .region import RegionMultiplyTransform, RegionTransform

__all__ = [
    "AutoBaselineTransform",
    "AutoScaleTransform",
    "DatasetTransform",
    "DynamicTransform",
    "ExperimentSetTransform",
    "ExperimentTransform",
    # Lambda and Dynamic
    "LambdaTransform",
    "LazyPipeline",
    "LazyResult",
    "MeasurementSetTransform",
    "MeasurementTransform",
    # Pipeline
    "Pipeline",
    "RegionMultiplyTransform",
    # Region-based
    "RegionTransform",
    # Base transforms
    "Transform",
    # Dataset transforms
    "dataset",
    # Utilities
    "jit_transform",
    # Measurement transforms
    "measurement",
]
