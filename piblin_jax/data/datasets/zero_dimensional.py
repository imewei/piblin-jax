"""
Zero-dimensional dataset for scalar values.

Used for single values like steady-state measurements, summary statistics,
or aggregated results.
"""

from typing import Any

from piblin_jax.backend import jnp, to_numpy

from .base import Dataset


class ZeroDimensionalDataset(Dataset):
    """
    Zero-dimensional dataset containing a single scalar value.

    This dataset type represents a single measured or calculated value,
    such as a steady-state measurement, summary statistic, or aggregated result.

    Parameters
    ----------
    value : float
        The scalar value to store.
    conditions : dict[str, Any] | None, optional
        Experimental conditions associated with this measurement.
    details : dict[str, Any] | None, optional
        Additional context and metadata.

    Attributes
    ----------
    value : float
        The scalar value (converted to Python float).
    conditions : dict[str, Any]
        Experimental conditions.
    details : dict[str, Any]
        Additional metadata.

    Examples
    --------
    >>> from piblin_jax.data.datasets import ZeroDimensionalDataset
    >>> # Steady-state temperature measurement
    >>> temp = ZeroDimensionalDataset(
    ...     value=98.6,
    ...     conditions={"location": "oral", "patient_id": "12345"},
    ...     details={"units": "fahrenheit", "instrument": "thermometer"}
    ... )
    >>> temp.value
    98.6

    >>> # Summary statistic
    >>> mean_concentration = ZeroDimensionalDataset(
    ...     value=2.5e-3,
    ...     conditions={"sample": "batch_42"},
    ...     details={"units": "mol/L", "statistic": "mean"}
    ... )
    >>> mean_concentration.value
    0.0025

    Notes
    -----
    The value is stored internally as a backend array (JAX or NumPy scalar)
    and converted to a Python float when accessed through the `value` property.
    This ensures compatibility with both JAX transformations and standard
    Python numeric operations.
    """

    def __init__(
        self,
        value: float,
        conditions: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
    ):
        """
        Initialize zero-dimensional dataset with a scalar value.

        Parameters
        ----------
        value : float
            The scalar value to store.
        conditions : dict[str, Any] | None, optional
            Experimental conditions.
        details : dict[str, Any] | None, optional
            Additional metadata.
        """
        super().__init__(conditions=conditions, details=details)

        # Store as backend scalar
        self._value = jnp.asarray(value)

    @property
    def value(self) -> float:
        """
        Get the scalar value as a Python float.

        :no-index:

        Returns
        -------
        float
            The stored scalar value.

        Examples
        --------
        >>> dataset = ZeroDimensionalDataset(value=42.5)
        >>> dataset.value
        42.5
        >>> type(dataset.value)
        <class 'float'> or <class 'numpy.floating'>
        """
        # Convert to NumPy then to Python scalar
        return float(to_numpy(self._value))
