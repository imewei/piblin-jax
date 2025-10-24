"""
Three-dimensional dataset with three independent variables and 3D dependent data.

Used for volumetric data, 3D imaging, or data varying with three parameters.
"""

from typing import Any

import numpy as np

from piblin_jax.backend import jnp, to_numpy

from .base import Dataset


class ThreeDimensionalDataset(Dataset):
    """
    Three-dimensional dataset with three independent variables and a 3D dependent array.

    This dataset type represents volumetric or 3D data:
    - 3D microscopy/imaging (confocal, CT, MRI)
    - Volumetric spectroscopy
    - Three-parameter experiments (e.g., temperature, pressure, time)
    - Computational fluid dynamics results
    - Molecular dynamics trajectories

    Parameters
    ----------
    independent_variable_data_1 : array_like
        1D array of first independent variable (e.g., x-coordinate, temperature).
    independent_variable_data_2 : array_like
        1D array of second independent variable (e.g., y-coordinate, pressure).
    independent_variable_data_3 : array_like
        1D array of third independent variable (e.g., z-coordinate, time).
    dependent_variable_data : array_like
        3D array of dependent variable values with shape
        (len(var1), len(var2), len(var3)).
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
    independent_variable_data_3 : np.ndarray
        Third independent variable as NumPy array.
    dependent_variable_data : np.ndarray
        3D dependent variable as NumPy array.
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
    >>> from piblin_jax.data.datasets import ThreeDimensionalDataset
    >>> # 3D confocal microscopy data
    >>> x = np.linspace(0, 100, 64)  # microns
    >>> y = np.linspace(0, 100, 64)  # microns
    >>> z = np.linspace(0, 50, 32)   # microns (z-stack)
    >>> intensity = np.random.rand(64, 64, 32)
    >>> volume = ThreeDimensionalDataset(
    ...     independent_variable_data_1=x,
    ...     independent_variable_data_2=y,
    ...     independent_variable_data_3=z,
    ...     dependent_variable_data=intensity,
    ...     conditions={"wavelength": 488, "objective": "40x"},
    ...     details={"voxel_size": "1.56 x 1.56 x 1.56 um"}
    ... )
    >>> volume.dependent_variable_data.shape
    (64, 64, 32)

    >>> # Three-parameter experiment (T, P, t)
    >>> temperatures = np.array([25, 50, 75, 100])
    >>> pressures = np.array([1, 5, 10, 15, 20])
    >>> times = np.array([0, 60, 120, 180])
    >>> conversion = np.random.rand(4, 5, 4)
    >>> experiment = ThreeDimensionalDataset(
    ...     independent_variable_data_1=temperatures,
    ...     independent_variable_data_2=pressures,
    ...     independent_variable_data_3=times,
    ...     dependent_variable_data=conversion,
    ...     conditions={"catalyst": "Pt/Al2O3", "reactant": "H2 + CO"},
    ...     details={"experiment": "Fischer-Tropsch"}
    ... )

    Notes
    -----
    The dependent variable must have shape (len(var1), len(var2), len(var3)).
    Arrays are stored internally as backend arrays and converted to NumPy
    when accessed.
    """

    def __init__(
        self,
        independent_variable_data_1: Any,
        independent_variable_data_2: Any,
        independent_variable_data_3: Any,
        dependent_variable_data: Any,
        conditions: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
    ):
        """
        Initialize three-dimensional dataset.

        Parameters
        ----------
        independent_variable_data_1 : array_like
            1D array of first independent variable.
        independent_variable_data_2 : array_like
            1D array of second independent variable.
        independent_variable_data_3 : array_like
            1D array of third independent variable.
        dependent_variable_data : array_like
            3D array of dependent variable.
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
        self._independent_variable_data_3 = jnp.asarray(independent_variable_data_3)
        self._dependent_variable_data = jnp.asarray(dependent_variable_data)

        # Validation: check dimension compatibility
        expected_shape = (
            self._independent_variable_data_1.shape[0],
            self._independent_variable_data_2.shape[0],
            self._independent_variable_data_3.shape[0],
        )

        if self._dependent_variable_data.shape != expected_shape:
            raise ValueError(
                f"Dependent variable dimension mismatch. "
                f"Expected shape {expected_shape} based on independent variables "
                f"(len={self._independent_variable_data_1.shape[0]}, "
                f"len={self._independent_variable_data_2.shape[0]}, "
                f"len={self._independent_variable_data_3.shape[0]}), "
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
        array([0., 1.59, 3.17, ..., 98.41, 100.])
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
        array([0., 1.59, 3.17, ..., 98.41, 100.])
        """
        return to_numpy(self._independent_variable_data_2)

    @property
    def independent_variable_data_3(self) -> np.ndarray:
        """
        Get third independent variable as NumPy array.

        Returns
        -------
        np.ndarray
            1D NumPy array of third independent variable values.

        Examples
        --------
        >>> dataset.independent_variable_data_3
        array([0., 1.61, 3.23, ..., 48.39, 50.])
        """
        return to_numpy(self._independent_variable_data_3)

    @property
    def dependent_variable_data(self) -> np.ndarray:
        """
        Get dependent variable data as NumPy array.

        Returns
        -------
        np.ndarray
            3D NumPy array of dependent variable values with shape
            (len(var1), len(var2), len(var3)).

        Examples
        --------
        >>> dataset.dependent_variable_data.shape
        (64, 64, 32)
        >>> dataset.dependent_variable_data[0, 0, 0]  # Value at first point
        0.456
        """
        return to_numpy(self._dependent_variable_data)
