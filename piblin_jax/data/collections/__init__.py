"""
Collection classes for piblin-jax.

This module provides the hierarchical data model for organizing experimental data:

Hierarchy:
ExperimentSet -> Experiment -> MeasurementSet -> Measurement -> Dataset

Collection Types:
- Measurement: Container for multiple Dataset objects
- MeasurementSet: Base class for measurement collections
- ConsistentMeasurementSet: Measurements with same dataset structure
- TidyMeasurementSet: Measurements with comparable conditions
- TabularMeasurementSet: Measurements in tabular format
- Experiment: Container for multiple MeasurementSet objects
- ExperimentSet: Top-level container for multiple Experiment objects
"""

from .consistent_measurement_set import ConsistentMeasurementSet
from .experiment import Experiment
from .experiment_set import ExperimentSet
from .measurement import Measurement
from .measurement_set import MeasurementSet
from .tabular_measurement_set import TabularMeasurementSet
from .tidy_measurement_set import TidyMeasurementSet

__all__ = [
    "ConsistentMeasurementSet",
    "Experiment",
    "ExperimentSet",
    "Measurement",
    "MeasurementSet",
    "TabularMeasurementSet",
    "TidyMeasurementSet",
]
