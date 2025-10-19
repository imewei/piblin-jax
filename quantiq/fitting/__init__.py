"""
Fitting module for quantiq.

Provides curve fitting functionality with NLSQ integration and scipy fallback.
"""

from .nlsq import fit_curve, estimate_initial_parameters

__all__ = [
    'fit_curve',
    'estimate_initial_parameters',
]
