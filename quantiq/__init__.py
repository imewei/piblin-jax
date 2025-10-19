"""
quantiq - Modern JAX-Powered Framework for Measurement Data Science

A complete reimplementation of piblin with JAX performance enhancements,
NumPyro Bayesian uncertainty quantification, and modern Python 3.12+ features.

Target Performance:
- 5-10x CPU speedup over piblin
- 50-100x GPU speedup for large datasets

Key Features:
- JAX-based JIT compilation and automatic GPU acceleration
- NumPyro Bayesian uncertainty quantification
- NLSQ non-linear fitting integration
- 100% piblin API behavioral compatibility
- Functional programming paradigm
- Lazy evaluation with computation graph optimization

Usage:
    import quantiq
    # or for piblin compatibility:
    import quantiq as piblin
"""

__version__ = "0.1.0"

# Import submodules
from . import bayesian
from . import data
from . import dataio
from . import transform
from . import backend

# Core dataset classes
from .data.datasets import (
    OneDimensionalDataset,
    TwoDimensionalDataset,
    ThreeDimensionalDataset,
    ZeroDimensionalDataset,
    OneDimensionalCompositeDataset,
    Histogram,
    Distribution,
)

# Collection classes
from .data.collections import (
    Measurement,
    MeasurementSet,
    Experiment,
    ExperimentSet,
    ConsistentMeasurementSet,
    TabularMeasurementSet,
    TidyMeasurementSet,
)

# Transform classes
from .transform.pipeline import Pipeline
from .transform.lambda_transform import LambdaTransform
from .transform.base import Transform, DatasetTransform

# Bayesian classes
from .bayesian.base import BayesianModel
from .bayesian.models import (
    PowerLawModel,
    ArrheniusModel,
    CrossModel,
    CarreauYasudaModel,
)

# Data I/O
from .dataio import read_files, read_directory, read_directories
from .dataio.readers.csv import GenericCSVReader

# Fitting
from .fitting import fit_curve, estimate_initial_parameters

__all__ = [
    "__version__",
    # Submodules
    "bayesian",
    "data",
    "dataio",
    "transform",
    "backend",
    # Dataset classes
    "OneDimensionalDataset",
    "TwoDimensionalDataset",
    "ThreeDimensionalDataset",
    "ZeroDimensionalDataset",
    "OneDimensionalCompositeDataset",
    "Histogram",
    "Distribution",
    # Collection classes
    "Measurement",
    "MeasurementSet",
    "Experiment",
    "ExperimentSet",
    "ConsistentMeasurementSet",
    "TabularMeasurementSet",
    "TidyMeasurementSet",
    # Transform classes
    "Pipeline",
    "LambdaTransform",
    "Transform",
    "DatasetTransform",
    # Bayesian classes
    "BayesianModel",
    "PowerLawModel",
    "ArrheniusModel",
    "CrossModel",
    "CarreauYasudaModel",
    # Data I/O
    "read_files",
    "read_directory",
    "read_directories",
    "GenericCSVReader",
    # Fitting
    "fit_curve",
    "estimate_initial_parameters",
]
