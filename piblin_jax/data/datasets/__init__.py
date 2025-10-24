"""
Dataset classes for piblin-jax.

This module provides all dataset types for storing experimental and computed data:
- Dataset: Abstract base class
- ZeroDimensionalDataset: Single scalar values
- OneDimensionalDataset: 1D paired data (most common)
- TwoDimensionalDataset: 2D data with two independent variables
- ThreeDimensionalDataset: 3D volumetric data
- Histogram: Binned data with variable-width bins
- Distribution: Continuous probability density functions
- OneDimensionalCompositeDataset: Multi-channel data with shared axis
"""

from .base import Dataset
from .composite import OneDimensionalCompositeDataset
from .distribution import Distribution
from .histogram import Histogram
from .one_dimensional import OneDimensionalDataset
from .three_dimensional import ThreeDimensionalDataset
from .two_dimensional import TwoDimensionalDataset
from .zero_dimensional import ZeroDimensionalDataset

__all__ = [
    "Dataset",
    "Distribution",
    "Histogram",
    "OneDimensionalCompositeDataset",
    "OneDimensionalDataset",
    "ThreeDimensionalDataset",
    "TwoDimensionalDataset",
    "ZeroDimensionalDataset",
]
