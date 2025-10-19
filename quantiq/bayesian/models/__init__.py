"""
Built-in rheological models for Bayesian inference.

This module provides ready-to-use rheological models that inherit from
BayesianModel and implement common constitutive equations for viscosity.
"""

from .power_law import PowerLawModel
from .arrhenius import ArrheniusModel
from .cross import CrossModel
from .carreau_yasuda import CarreauYasudaModel

__all__ = [
    "PowerLawModel",
    "ArrheniusModel",
    "CrossModel",
    "CarreauYasudaModel",
]
