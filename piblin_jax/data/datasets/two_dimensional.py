"""
Two-dimensional dataset with two independent variables and 2D dependent data.

Used for data that varies with two parameters, such as time-temperature maps,
spatial imaging data, or parameter sweeps.
"""

from typing import Any

import numpy as np

from piblin_jax.backend import jnp, to_numpy

from .base import Dataset


class TwoDimensionalDataset(Dataset):
    """
    Two-dimensional dataset with two independent variables and a 2D dependent array.

    This dataset type represents data that varies with two independent parameters:
    - Time-temperature maps (kinetics studies)
    - Spatial imaging data (microscopy, spectroscopy maps)
    - Parameter sweep experiments
    - Contour plots and heatmaps

    Parameters
    ----------
    independent_variable_data_1 : array_like
        1D array of first independent variable (e.g., temperature, x-coordinate).
    independent_variable_data_2 : array_like
        1D array of second independent variable (e.g., time, y-coordinate).
    dependent_variable_data : array_like
        2D array of dependent variable values with shape (len(var1), len(var2)).
    conditions : dict[str, Any] | None, optional
        Experimental conditions.
    details : dict[str, Any] | None, optional
        Additional metadata.

    Attributes
    ----------
    independent_variable_data_1 : np.ndarray
        First independent variable as NumPy array.
    independent_variable_data_2 : np.ndarray
        Second independent variable as NumPy array.
    dependent_variable_data : np.ndarray
        2D dependent variable as NumPy array.
    conditions : dict[str, Any]
        Experimental conditions.
    details : dict[str, Any]
        Additional metadata.

    Raises
    ------
    ValueError
        If dimension compatibility is violated.

    Examples
    --------
    >>> import numpy as np
    >>> from piblin_jax.data.datasets import TwoDimensionalDataset
    >>> # Temperature-time kinetics map
    >>> temperature = np.linspace(20, 100, 50)  # 50 temperatures
    >>> time = np.linspace(0, 3600, 100)        # 100 time points
    >>> # Reaction extent at each (temp, time) combination
    >>> extent = np.random.rand(50, 100)
    >>> dataset = TwoDimensionalDataset(
    ...     independent_variable_data_1=temperature,
    ...     independent_variable_data_2=time,
    ...     dependent_variable_data=extent,
    ...     conditions={"catalyst": "Pd/C", "solvent": "ethanol"},
    ...     details={"experiment_id": "KIN-2025-042"}
    ... )
    >>> dataset.independent_variable_data_1.shape
    (50,)
    >>> dataset.dependent_variable_data.shape
    (50, 100)

    >>> # Spectroscopy imaging (spatial map)
    >>> x_coords = np.linspace(0, 10, 64)
    >>> y_coords = np.linspace(0, 10, 64)
    >>> intensity_map = np.random.rand(64, 64)
    >>> image = TwoDimensionalDataset(
    ...     independent_variable_data_1=x_coords,
    ...     independent_variable_data_2=y_coords,
    ...     dependent_variable_data=intensity_map,
    ...     conditions={"wavelength": 532, "power": 10},
    ...     details={"units": "microns", "resolution": "0.156 um/pixel"}
    ... )

    Notes
    -----
    The dependent variable must have shape (len(var1), len(var2)). Arrays are
    stored internally as backend arrays and converted to NumPy when accessed.
    """

    def __init__(
        self,
        independent_variable_data_1: Any,
        independent_variable_data_2: Any,
        dependent_variable_data: Any,
        conditions: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
    ):
        """
        Initialize two-dimensional dataset.

        Parameters
        ----------
        independent_variable_data_1 : array_like
            1D array of first independent variable.
        independent_variable_data_2 : array_like
            1D array of second independent variable.
        dependent_variable_data : array_like
            2D array of dependent variable.
        conditions : dict[str, Any] | None, optional
            Experimental conditions.
        details : dict[str, Any] | None, optional
            Additional metadata.

        Raises
        ------
        ValueError
            If dimension compatibility is violated.
        """
        super().__init__(conditions=conditions, details=details)

        # Convert to backend arrays
        self._independent_variable_data_1 = jnp.asarray(independent_variable_data_1)
        self._independent_variable_data_2 = jnp.asarray(independent_variable_data_2)
        self._dependent_variable_data = jnp.asarray(dependent_variable_data)

        # Validation: check dimension compatibility
        expected_shape = (
            self._independent_variable_data_1.shape[0],
            self._independent_variable_data_2.shape[0],
        )

        if self._dependent_variable_data.shape != expected_shape:
            raise ValueError(
                f"Dependent variable dimension mismatch. "
                f"Expected shape {expected_shape} based on independent variables "
                f"(len={self._independent_variable_data_1.shape[0]}, "
                f"len={self._independent_variable_data_2.shape[0]}), "
                f"but got {self._dependent_variable_data.shape}"
            )

    @property
    def independent_variable_data_1(self) -> np.ndarray:
        """
        Get first independent variable as NumPy array.

        Returns
        -------
        np.ndarray
            1D NumPy array of first independent variable values.

        Examples
        --------
        >>> dataset.independent_variable_data_1
        array([20., 21.63, 23.27, ..., 98.37, 100.])
        """
        return to_numpy(self._independent_variable_data_1)

    @property
    def independent_variable_data_2(self) -> np.ndarray:
        """
        Get second independent variable as NumPy array.

        Returns
        -------
        np.ndarray
            1D NumPy array of second independent variable values.

        Examples
        --------
        >>> dataset.independent_variable_data_2
        array([0., 36.36, 72.73, ..., 3527.27, 3563.64, 3600.])
        """
        return to_numpy(self._independent_variable_data_2)

    @property
    def dependent_variable_data(self) -> np.ndarray:
        """
        Get dependent variable data as NumPy array.

        Returns
        -------
        np.ndarray
            2D NumPy array of dependent variable values with shape
            (len(var1), len(var2)).

        Examples
        --------
        >>> dataset.dependent_variable_data.shape
        (50, 100)
        >>> dataset.dependent_variable_data[0, 0]  # Value at first (temp, time)
        0.234
        """
        return to_numpy(self._dependent_variable_data)
