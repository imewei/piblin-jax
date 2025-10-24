"""
piblin-jax - Modern JAX-Powered Framework for Measurement Data Science

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
    import piblin_jax
    # or for piblin compatibility:
    import piblin_jax as piblin
"""

__version__ = "0.0.3"

# Import core submodules (JAX-independent)
from . import backend, data, dataio, transform

# Conditionally import JAX-dependent modules
# The bayesian module requires JAX/NumPyro which may not be available
# when testing with numpy backend or in environments without JAX
if backend.is_jax_available():
    from . import bayesian

    # Bayesian classes
    from .bayesian.base import BayesianModel
    from .bayesian.models import (
        ArrheniusModel,
        CarreauYasudaModel,
        CrossModel,
        PowerLawModel,
    )
else:
    # Provide None placeholders when JAX is unavailable
    # This allows the package to be imported even without JAX installed
    bayesian = None  # type: ignore
    BayesianModel = None  # type: ignore
    ArrheniusModel = None  # type: ignore
    CarreauYasudaModel = None  # type: ignore
    CrossModel = None  # type: ignore
    PowerLawModel = None  # type: ignore

# Collection classes
from .data.collections import (
    ConsistentMeasurementSet,
    Experiment,
    ExperimentSet,
    Measurement,
    MeasurementSet,
    TabularMeasurementSet,
    TidyMeasurementSet,
)

# Core dataset classes
from .data.datasets import (
    Distribution,
    Histogram,
    OneDimensionalCompositeDataset,
    OneDimensionalDataset,
    ThreeDimensionalDataset,
    TwoDimensionalDataset,
    ZeroDimensionalDataset,
)

# Data I/O
from .dataio import read_directories, read_directory, read_files
from .dataio.readers.csv import GenericCSVReader

# Fitting
from .fitting import estimate_initial_parameters, fit_curve
from .transform.base import DatasetTransform, Transform
from .transform.lambda_transform import LambdaTransform

# Transform classes
from .transform.pipeline import Pipeline

__all__ = [
    "ArrheniusModel",
    # Bayesian classes
    "BayesianModel",
    "CarreauYasudaModel",
    "ConsistentMeasurementSet",
    "CrossModel",
    "DatasetTransform",
    "Distribution",
    "Experiment",
    "ExperimentSet",
    "GenericCSVReader",
    "Histogram",
    "LambdaTransform",
    # Collection classes
    "Measurement",
    "MeasurementSet",
    "OneDimensionalCompositeDataset",
    # Dataset classes
    "OneDimensionalDataset",
    # Transform classes
    "Pipeline",
    "PowerLawModel",
    "TabularMeasurementSet",
    "ThreeDimensionalDataset",
    "TidyMeasurementSet",
    "Transform",
    "TwoDimensionalDataset",
    "ZeroDimensionalDataset",
    "__version__",
    "backend",
    # Submodules
    "bayesian",
    "data",
    "dataio",
    "estimate_initial_parameters",
    # Fitting
    "fit_curve",
    "read_directories",
    "read_directory",
    # Data I/O
    "read_files",
    "transform",
]
