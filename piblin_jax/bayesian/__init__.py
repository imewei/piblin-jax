"""
Bayesian inference module for quantiq.

This module provides Bayesian modeling capabilities using NumPyro,
including MCMC sampling and uncertainty quantification.
"""

from .base import BayesianModel
from .models import (
    ArrheniusModel,
    CarreauYasudaModel,
    CrossModel,
    PowerLawModel,
)

__all__ = [
    "ArrheniusModel",
    "BayesianModel",
    "CarreauYasudaModel",
    "CrossModel",
    "PowerLawModel",
]
