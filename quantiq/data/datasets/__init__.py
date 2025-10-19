"""
Dataset classes for quantiq.

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
from .zero_dimensional import ZeroDimensionalDataset
from .one_dimensional import OneDimensionalDataset
from .two_dimensional import TwoDimensionalDataset
from .three_dimensional import ThreeDimensionalDataset
from .histogram import Histogram
from .distribution import Distribution
from .composite import OneDimensionalCompositeDataset

__all__ = [
    "Dataset",
    "ZeroDimensionalDataset",
    "OneDimensionalDataset",
    "TwoDimensionalDataset",
    "ThreeDimensionalDataset",
    "Histogram",
    "Distribution",
    "OneDimensionalCompositeDataset",
]
