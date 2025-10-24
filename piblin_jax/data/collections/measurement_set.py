"""
MeasurementSet base class for piblin-jax.

Container for multiple Measurement objects representing a series of related measurements.
"""

from collections.abc import Iterator
from typing import Any

from .measurement import Measurement


class MeasurementSet:
    """
    Base class for collections of Measurement objects.

    A MeasurementSet represents a series of related measurements, such as:
    - Time series measurements
    - Replicate measurements
    - Parameter sweep measurements
    - Multi-sample measurements

    This is the base class. Specialized variants include:
    - ConsistentMeasurementSet: All measurements have same structure
    - TidyMeasurementSet: All measurements share comparable conditions
    - TabularMeasurementSet: Measurements arranged in tabular format

    Parameters
    ----------
    measurements : list[Measurement]
        List of Measurement objects in this set.
    conditions : dict[str, Any] | None, optional
        Experimental conditions for the entire measurement series
        (e.g., sample, experimental setup, date).
    details : dict[str, Any] | None, optional
        Additional context for this measurement series
        (e.g., series description, experimental notes).

    Attributes
    ----------
    measurements : tuple[Measurement, ...]
        Immutable tuple of measurements in this set.
    conditions : dict[str, Any]
        Experimental conditions for this measurement series.
    details : dict[str, Any]
        Additional metadata for this measurement series.

    Notes
    -----
    The measurements are stored as a tuple to ensure immutability, which is
    required for JAX transformations. Individual measurements can be accessed
    by indexing or iteration.

    Examples
    --------
    >>> import numpy as np
    >>> from piblin_jax.data.datasets import OneDimensionalDataset
    >>> from piblin_jax.data.collections import Measurement, MeasurementSet
    >>>
    >>> # Create replicate measurements
    >>> x = np.linspace(0, 10, 100)
    >>> measurements = []
    >>> for i in range(3):
    ...     y = np.sin(x) + np.random.normal(0, 0.1, len(x))
    ...     ds = OneDimensionalDataset(x, y)
    ...     m = Measurement(
    ...         datasets=[ds],
    ...         conditions={"replicate": i+1}
    ...     )
    ...     measurements.append(m)
    >>>
    >>> # Create measurement set
    >>> ms = MeasurementSet(
    ...     measurements=measurements,
    ...     conditions={"sample": "S1", "date": "2025-10-18"},
    ...     details={"notes": "Replicate measurements with noise"}
    ... )
    >>>
    >>> # Access measurements
    >>> len(ms)
    3
    >>> first_measurement = ms[0]
    >>> for m in ms:
    ...     print(m.conditions["replicate"])
    1
    2
    3
    """

    def __init__(
        self,
        measurements: list[Measurement],
        conditions: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
    ):
        """
        Initialize MeasurementSet with measurements and metadata.

        Parameters
        ----------
        measurements : list[Measurement]
            List of Measurement objects in this set.
        conditions : dict[str, Any] | None, optional
            Experimental conditions for this measurement series.
        details : dict[str, Any] | None, optional
            Additional context for this measurement series.
        """
        self._measurements = tuple(measurements)  # Immutable for JAX compatibility
        self._conditions = conditions if conditions is not None else {}
        self._details = details if details is not None else {}

    @property
    def measurements(self) -> tuple[Measurement, ...]:
        """
        Get all measurements in this set.

        :no-index:

        Returns
        -------
        tuple[Measurement, ...]
            Immutable tuple of Measurement objects.

        Examples
        --------
        >>> ms.measurements
        (<Measurement at 0x...>, <Measurement at 0x...>, <Measurement at 0x...>)
        """
        return self._measurements

    @property
    def conditions(self) -> dict[str, Any]:
        """
        Get experimental conditions for this measurement series.

        :no-index:

        Returns
        -------
        dict[str, Any]
            Dictionary of experimental conditions (sample, date, setup, etc.).

        Examples
        --------
        >>> ms.conditions
        {'sample': 'S1', 'date': '2025-10-18', 'instrument': 'Spec-X'}
        """
        return self._conditions

    @property
    def details(self) -> dict[str, Any]:
        """
        Get additional details for this measurement series.

        :no-index:

        Returns
        -------
        dict[str, Any]
            Dictionary of additional context (notes, quality, etc.).

        Examples
        --------
        >>> ms.details
        {'notes': 'Time series', 'quality': 'good'}
        """
        return self._details

    def __len__(self) -> int:
        """
        Get number of measurements in this set.

        Returns
        -------
        int
            Number of measurements.

        Examples
        --------
        >>> len(ms)
        3
        """
        return len(self._measurements)

    def __iter__(self) -> Iterator[Measurement]:
        """
        Iterate over measurements in this set.

        Yields
        ------
        Measurement
            Each measurement in order.

        Examples
        --------
        >>> for measurement in ms:
        ...     print(len(measurement))
        1
        1
        1
        """
        return iter(self._measurements)

    def __getitem__(self, index: int | slice) -> Measurement | tuple[Measurement, ...]:
        """
        Get measurement by index.

        Parameters
        ----------
        index : int or slice
            Index or slice to access measurements.

        Returns
        -------
        Measurement or tuple[Measurement, ...]
            Measurement at the given index, or tuple of measurements for slice.

        Examples
        --------
        >>> ms[0]
        <Measurement at 0x...>
        >>> ms[0:2]
        (<Measurement at 0x...>, <Measurement at 0x...>)
        """
        return self._measurements[index]
