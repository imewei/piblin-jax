"""
Composite one-dimensional dataset with multiple dependent variables.

Used for multi-channel instrument data where multiple signals share the
same independent variable (e.g., time, wavelength).
"""

from typing import Any

import numpy as np

from piblin_jax.backend import jnp, to_numpy

from .base import Dataset


class OneDimensionalCompositeDataset(Dataset):
    """
    Composite 1D dataset with shared independent variable and multiple dependents.

    This dataset type represents multi-channel or multi-detector data where
    multiple signals share the same independent variable:
    - Multi-detector chromatography (UV, fluorescence, conductivity)
    - Multi-channel spectroscopy
    - Multi-sensor time series
    - Parallel measurements with shared axis

    Parameters
    ----------
    independent_variable_data : array_like
        1D array of independent variable (time, wavelength, etc.) shared by
        all channels.
    dependent_variable_data_list : list of array_like
        List of 1D arrays, each representing a different channel/detector.
        All must have the same length as independent_variable_data.
    conditions : dict[str, Any] | None, optional
        Experimental conditions.
    details : dict[str, Any] | None, optional
        Additional metadata.

    Attributes
    ----------
    independent_variable_data : np.ndarray
        Shared independent variable as NumPy array.
    dependent_variable_data_list : list of np.ndarray
        List of dependent variables as NumPy arrays.
    conditions : dict[str, Any]
        Experimental conditions.
    details : dict[str, Any]
        Additional metadata.

    Raises
    ------
    ValueError
        If dependent_variable_data_list is empty, or if any channel has
        different length than independent_variable_data.

    Examples
    --------
    >>> import numpy as np
    >>> from piblin_jax.data.datasets import OneDimensionalCompositeDataset
    >>> # Multi-detector HPLC data
    >>> time = np.linspace(0, 20, 2000)  # minutes
    >>> uv_254 = np.sin(time) + 0.1 * np.random.randn(2000)
    >>> uv_280 = np.cos(time) + 0.1 * np.random.randn(2000)
    >>> fluorescence = np.sin(2 * time) + 0.05 * np.random.randn(2000)
    >>> hplc = OneDimensionalCompositeDataset(
    ...     independent_variable_data=time,
    ...     dependent_variable_data_list=[uv_254, uv_280, fluorescence],
    ...     conditions={"mobile_phase": "ACN/H2O 60:40", "flow_rate": 1.0},
    ...     details={
    ...         "channels": ["UV 254nm", "UV 280nm", "Fluorescence"],
    ...         "instrument": "HPLC-1"
    ...     }
    ... )
    >>> hplc.independent_variable_data.shape
    (2000,)
    >>> len(hplc.dependent_variable_data_list)
    3
    >>> hplc.dependent_variable_data_list[0].shape
    (2000,)

    >>> # Multi-channel oscilloscope data
    >>> t = np.linspace(0, 1, 10000)
    >>> ch1 = np.sin(2 * np.pi * 5 * t)
    >>> ch2 = np.sin(2 * np.pi * 10 * t)
    >>> ch3 = np.sin(2 * np.pi * 15 * t)
    >>> ch4 = np.sin(2 * np.pi * 20 * t)
    >>> scope_data = OneDimensionalCompositeDataset(
    ...     independent_variable_data=t,
    ...     dependent_variable_data_list=[ch1, ch2, ch3, ch4],
    ...     conditions={"sampling_rate": 10000},
    ...     details={"instrument": "oscilloscope", "channels": 4}
    ... )

    Notes
    -----
    This dataset type is useful when multiple measurements are made simultaneously
    along the same independent axis. Each channel is stored as a separate NumPy
    array in the list, allowing different processing or analysis on each channel
    while maintaining their shared relationship through the common independent
    variable.

    The internal storage uses backend arrays (JAX when available) and converts
    to NumPy at the property boundaries.
    """

    def __init__(
        self,
        independent_variable_data: Any,
        dependent_variable_data_list: list[Any],
        conditions: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
    ):
        """
        Initialize composite one-dimensional dataset.

        Parameters
        ----------
        independent_variable_data : array_like
            1D array of shared independent variable.
        dependent_variable_data_list : list of array_like
            List of 1D arrays for each channel.
        conditions : dict[str, Any] | None, optional
            Experimental conditions.
        details : dict[str, Any] | None, optional
            Additional metadata.

        Raises
        ------
        ValueError
            If list is empty or if any channel length doesn't match
            independent variable.
        """
        super().__init__(conditions=conditions, details=details)

        # Validation: must have at least one dependent variable
        if not dependent_variable_data_list or len(dependent_variable_data_list) == 0:
            raise ValueError(
                "OneDimensionalCompositeDataset requires at least one dependent variable. "
                "Got empty list."
            )

        # Convert independent variable to backend array
        self._independent_variable_data = jnp.asarray(independent_variable_data)
        expected_length = self._independent_variable_data.shape[0]

        # Convert all dependent variables to backend arrays and validate
        self._dependent_variable_data_list = []
        for i, dep_data in enumerate(dependent_variable_data_list):
            dep_array = jnp.asarray(dep_data)

            # Validation: each channel must match independent variable length
            if dep_array.shape[0] != expected_length:
                raise ValueError(
                    f"All dependent variables must have same length as independent variable. "
                    f"Independent variable has length {expected_length}, but "
                    f"dependent variable at index {i} has length {dep_array.shape[0]}"
                )

            self._dependent_variable_data_list.append(dep_array)

    @property
    def independent_variable_data(self) -> np.ndarray:
        """
        Get shared independent variable as NumPy array.

        Returns
        -------
        np.ndarray
            1D NumPy array of independent variable shared by all channels.

        Examples
        --------
        >>> dataset.independent_variable_data
        array([0., 0.01, 0.02, ..., 19.98, 19.99, 20.])
        """
        return to_numpy(self._independent_variable_data)

    @property
    def dependent_variable_data_list(self) -> list[np.ndarray]:
        """
        Get list of dependent variables as NumPy arrays.

        Returns
        -------
        list of np.ndarray
            List of 1D NumPy arrays, one for each channel/detector.

        Examples
        --------
        >>> len(dataset.dependent_variable_data_list)
        3
        >>> dataset.dependent_variable_data_list[0]  # First channel
        array([0.123, 0.145, ..., 0.234])
        >>> dataset.dependent_variable_data_list[1]  # Second channel
        array([0.456, 0.478, ..., 0.567])
        >>> # Process each channel
        >>> for i, channel in enumerate(dataset.dependent_variable_data_list):
        ...     print(f"Channel {i}: max = {channel.max():.3f}")
        Channel 0: max = 1.234
        Channel 1: max = 1.567
        Channel 2: max = 0.987
        """
        # Convert all backend arrays to NumPy
        return [to_numpy(dep) for dep in self._dependent_variable_data_list]
