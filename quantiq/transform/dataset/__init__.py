"""
Core dataset-level transforms for quantiq.

This module provides fundamental transforms for processing 1D datasets:
- Interpolation: Resample to new x-values
- Smoothing: Reduce noise (moving average, Gaussian)
- Baseline correction: Remove systematic offsets and drifts
- Normalization: Scale data to standard ranges
- Calculus: Derivatives and integration

All transforms are JAX-compatible with JIT compilation support
and graceful fallback to NumPy when JAX is unavailable.
"""

from .interpolate import Interpolate1D
from .smoothing import MovingAverageSmooth, GaussianSmooth
from .baseline import PolynomialBaseline, AsymmetricLeastSquaresBaseline
from .normalization import (
    MinMaxNormalize,
    ZScoreNormalize,
    RobustNormalize,
    MaxNormalize,
)
from .calculus import (
    Derivative,
    CumulativeIntegral,
    DefiniteIntegral,
)

__all__ = [
    # Interpolation
    "Interpolate1D",
    # Smoothing
    "MovingAverageSmooth",
    "GaussianSmooth",
    # Baseline correction
    "PolynomialBaseline",
    "AsymmetricLeastSquaresBaseline",
    # Normalization
    "MinMaxNormalize",
    "ZScoreNormalize",
    "RobustNormalize",
    "MaxNormalize",
    # Calculus
    "Derivative",
    "CumulativeIntegral",
    "DefiniteIntegral",
]
