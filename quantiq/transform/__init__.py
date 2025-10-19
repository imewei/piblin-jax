"""
Transform system for quantiq.

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

from .base import (
    Transform,
    DatasetTransform,
    MeasurementTransform,
    MeasurementSetTransform,
    ExperimentTransform,
    ExperimentSetTransform,
    jit_transform,
)
from .pipeline import Pipeline, LazyPipeline, LazyResult
from .region import RegionTransform, RegionMultiplyTransform
from .lambda_transform import (
    LambdaTransform,
    DynamicTransform,
    AutoScaleTransform,
    AutoBaselineTransform,
)
# Import dataset transform submodule
from . import dataset
# Import measurement transform submodule
from . import measurement

__all__ = [
    # Base transforms
    "Transform",
    "DatasetTransform",
    "MeasurementTransform",
    "MeasurementSetTransform",
    "ExperimentTransform",
    "ExperimentSetTransform",
    # Pipeline
    "Pipeline",
    "LazyPipeline",
    "LazyResult",
    # Region-based
    "RegionTransform",
    "RegionMultiplyTransform",
    # Lambda and Dynamic
    "LambdaTransform",
    "DynamicTransform",
    "AutoScaleTransform",
    "AutoBaselineTransform",
    # Dataset transforms
    "dataset",
    # Measurement transforms
    "measurement",
    # Utilities
    "jit_transform",
]
