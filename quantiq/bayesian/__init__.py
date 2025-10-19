"""
Bayesian inference module for quantiq.

This module provides Bayesian modeling capabilities using NumPyro,
including MCMC sampling and uncertainty quantification.
"""

from .base import BayesianModel
from .models import (
    PowerLawModel,
    ArrheniusModel,
    CrossModel,
    CarreauYasudaModel,
)

__all__ = [
    "BayesianModel",
    "PowerLawModel",
    "ArrheniusModel",
    "CrossModel",
    "CarreauYasudaModel",
]
