"""
Built-in rheological models for Bayesian inference.

This module provides ready-to-use rheological models that inherit from
BayesianModel and implement common constitutive equations for viscosity.
"""

from .arrhenius import ArrheniusModel
from .carreau_yasuda import CarreauYasudaModel
from .cross import CrossModel
from .power_law import PowerLawModel

__all__ = [
    "ArrheniusModel",
    "CarreauYasudaModel",
    "CrossModel",
    "PowerLawModel",
]
