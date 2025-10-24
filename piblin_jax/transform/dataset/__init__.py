"""
Core dataset-level transforms for piblin-jax.

This module provides fundamental transforms for processing 1D datasets:
- Interpolation: Resample to new x-values
- Smoothing: Reduce noise (moving average, Gaussian)
- Baseline correction: Remove systematic offsets and drifts
- Normalization: Scale data to standard ranges
- Calculus: Derivatives and integration

All transforms are JAX-compatible with JIT compilation support
and graceful fallback to NumPy when JAX is unavailable.
"""

from .baseline import AsymmetricLeastSquaresBaseline, PolynomialBaseline
from .calculus import (
    CumulativeIntegral,
    DefiniteIntegral,
    Derivative,
)
from .interpolate import Interpolate1D
from .normalization import (
    MaxNormalize,
    MinMaxNormalize,
    RobustNormalize,
    ZScoreNormalize,
)
from .smoothing import GaussianSmooth, MovingAverageSmooth

__all__ = [
    "AsymmetricLeastSquaresBaseline",
    "CumulativeIntegral",
    "DefiniteIntegral",
    # Calculus
    "Derivative",
    "GaussianSmooth",
    # Interpolation
    "Interpolate1D",
    "MaxNormalize",
    # Normalization
    "MinMaxNormalize",
    # Smoothing
    "MovingAverageSmooth",
    # Baseline correction
    "PolynomialBaseline",
    "RobustNormalize",
    "ZScoreNormalize",
]
