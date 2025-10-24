"""
Interpolation transforms for 1D datasets.

This module provides interpolation transforms that resample datasets
to new x-values using various interpolation methods.
"""

from typing import Any

import numpy as np

from piblin_jax.backend import BACKEND, jnp
from piblin_jax.data.datasets import OneDimensionalDataset
from piblin_jax.transform.base import DatasetTransform


class Interpolate1D(DatasetTransform):
    """
    Interpolate 1D dataset to new x-values.

    This transform resamples a 1D dataset to a new set of independent
    variable values using linear interpolation. It supports both JAX
    and NumPy backends with automatic fallback.

    Parameters
    ----------
    new_x : array-like
        New independent variable values for interpolation.
    method : str, default='linear'
        Interpolation method. Currently only 'linear' is fully supported.
        Future versions may support 'cubic', 'spline', etc.

    Attributes
    ----------
    new_x : ndarray
        Target x-values for interpolation.
    method : str
        Interpolation method name.

    Examples
    --------
    >>> import numpy as np
    >>> from piblin_jax.data.datasets import OneDimensionalDataset
    >>> from piblin_jax.transform.dataset import Interpolate1D
    >>>
    >>> # Create original dataset
    >>> x = np.array([0, 1, 2, 3, 4])
    >>> y = np.array([0, 1, 4, 9, 16])  # y = x^2
    >>> dataset = OneDimensionalDataset(
    ...     independent_variable_data=x,
    ...     dependent_variable_data=y
    ... )
    >>>
    >>> # Interpolate to finer grid
    >>> new_x = np.linspace(0, 4, 17)  # 17 points from 0 to 4
    >>> interp = Interpolate1D(new_x, method='linear')
    >>> result = interp.apply_to(dataset)
    >>>
    >>> # Result has new x-values with interpolated y-values
    >>> result.independent_variable_data.shape
    (17,)

    Notes
    -----
    - Linear interpolation is used for both JAX and NumPy backends
    - For JAX backend, uses jnp.interp (compiled with JIT)
    - For NumPy backend, uses np.interp
    - Extrapolation uses constant values (edge values)
    - Metadata (conditions, details) is preserved from original dataset
    """

    def __init__(self, new_x: Any, method: str = "linear"):
        """
        Initialize interpolation transform.

        Parameters
        ----------
        new_x : array-like
            New independent variable values.
        method : str, default='linear'
            Interpolation method ('linear' supported).
        """
        super().__init__()
        self.new_x = jnp.asarray(new_x)
        self.method = method

        if method not in ["linear"]:
            raise ValueError(
                f"Interpolation method '{method}' not supported. "
                "Currently only 'linear' is implemented."
            )

    def _apply(self, dataset: OneDimensionalDataset) -> OneDimensionalDataset:  # type: ignore[override]
        """
        Apply interpolation to dataset.

        Parameters
        ----------
        dataset : OneDimensionalDataset
            Input dataset to interpolate.

        Returns
        -------
        OneDimensionalDataset
            New dataset with interpolated values.

        Notes
        -----
        Creates a new dataset instance with interpolated data while
        preserving metadata from the original dataset.
        """
        # Get original data
        x = dataset.independent_variable_data
        y = dataset.dependent_variable_data

        # Convert to backend arrays
        x = jnp.asarray(x)
        y = jnp.asarray(y)

        # Perform interpolation
        if BACKEND == "jax":
            # JAX interpolation (JIT-compiled)
            new_y = jnp.interp(self.new_x, x, y)
        else:
            # NumPy fallback
            new_y = np.interp(self.new_x, x, y)

        # Create new dataset with interpolated data
        # Preserve metadata from original dataset
        return OneDimensionalDataset(
            independent_variable_data=self.new_x,
            dependent_variable_data=new_y,
            conditions=dataset.conditions,
            details=dataset.details,
        )


__all__ = ["Interpolate1D"]
