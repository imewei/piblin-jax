"""
ConsistentMeasurementSet class for piblin-jax.

MeasurementSet variant where all measurements have the same dataset structure.
"""

from typing import Any

from .measurement import Measurement
from .measurement_set import MeasurementSet


class ConsistentMeasurementSet(MeasurementSet):
    """
    MeasurementSet where all measurements have the same dataset structure.

    This specialized variant enforces that all measurements contain datasets
    of the same types in the same order. This is useful for:
    - Replicate measurements (same protocol, multiple runs)
    - Time series measurements (same observables at different times)
    - Consistent multi-channel measurements

    The structural consistency enables array-based operations and
    easier data aggregation.

    Parameters
    ----------
    measurements : list[Measurement]
        List of Measurement objects. All must have the same structure.
    conditions : dict[str, Any] | None, optional
        Experimental conditions for the measurement series.
    details : dict[str, Any] | None, optional
        Additional context for the measurement series.

    Raises
    ------
    ValueError
        If measurements do not all have the same dataset structure.

    Notes
    -----
    Structure is defined as the sequence of dataset types. For example:

    - [OneDimensionalDataset, OneDimensionalDataset] is consistent with itself
    - [OneDimensionalDataset] is NOT consistent with [ZeroDimensionalDataset]
    - [OneDimensionalDataset, ZeroDimensionalDataset] is NOT consistent with
      [ZeroDimensionalDataset, OneDimensionalDataset] (order matters)

    Examples
    --------
    >>> import numpy as np
    >>> from piblin_jax.data.datasets import OneDimensionalDataset
    >>> from piblin_jax.data.collections import Measurement, ConsistentMeasurementSet
    >>>
    >>> # Create replicate measurements with consistent structure
    >>> x = np.linspace(0, 10, 100)
    >>> measurements = []
    >>> for i in range(5):
    ...     y = np.sin(x) + np.random.normal(0, 0.1, len(x))
    ...     ds = OneDimensionalDataset(x, y)
    ...     m = Measurement([ds], conditions={"replicate": i+1})
    ...     measurements.append(m)
    >>>
    >>> # Create consistent measurement set
    >>> cms = ConsistentMeasurementSet(
    ...     measurements=measurements,
    ...     conditions={"sample": "S1", "experiment": "replicates"}
    ... )
    >>>
    >>> len(cms)
    5
    >>>
    >>> # All measurements have the same structure
    >>> for m in cms:
    ...     print(len(m.datasets), type(m.datasets[0]).__name__)
    1 OneDimensionalDataset
    1 OneDimensionalDataset
    1 OneDimensionalDataset
    1 OneDimensionalDataset
    1 OneDimensionalDataset
    >>>
    >>> # This will raise ValueError - inconsistent structures
    >>> from piblin_jax.data.datasets import ZeroDimensionalDataset
    >>> m1 = Measurement([OneDimensionalDataset(np.array([1, 2]), np.array([3, 4]))])
    >>> m2 = Measurement([ZeroDimensionalDataset(5.0)])
    >>> ConsistentMeasurementSet([m1, m2])  # doctest: +SKIP
    ValueError: All measurements must have same structure
    """

    def __init__(
        self,
        measurements: list[Measurement],
        conditions: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
    ):
        """
        Initialize ConsistentMeasurementSet with structure validation.

        Parameters
        ----------
        measurements : list[Measurement]
            List of Measurement objects with consistent structure.
        conditions : dict[str, Any] | None, optional
            Experimental conditions for this measurement series.
        details : dict[str, Any] | None, optional
            Additional context for this measurement series.

        Raises
        ------
        ValueError
            If measurements do not all have the same structure.
        """
        # Validate consistency before calling parent __init__
        if measurements:
            first_structure = self._get_structure(measurements[0])
            for i, measurement in enumerate(measurements[1:], start=1):
                current_structure = self._get_structure(measurement)
                if current_structure != first_structure:
                    raise ValueError(
                        f"All measurements must have same structure. "
                        f"Measurement 0 has structure {first_structure}, "
                        f"but measurement {i} has structure {current_structure}."
                    )

        # Call parent constructor
        super().__init__(measurements, conditions, details)

    @staticmethod
    def _get_structure(measurement: Measurement) -> tuple[str, ...]:
        """
        Extract structure signature from a measurement.

        The structure is defined as the sequence of dataset type names.

        Parameters
        ----------
        measurement : Measurement
            Measurement to analyze.

        Returns
        -------
        tuple[str, ...]
            Tuple of dataset type names in order.

        Examples
        --------
        >>> from piblin_jax.data.datasets import OneDimensionalDataset
        >>> ds1 = OneDimensionalDataset(np.array([1, 2]), np.array([3, 4]))
        >>> ds2 = OneDimensionalDataset(np.array([5, 6]), np.array([7, 8]))
        >>> m = Measurement([ds1, ds2])
        >>> ConsistentMeasurementSet._get_structure(m)
        ('OneDimensionalDataset', 'OneDimensionalDataset')
        """
        return tuple(type(dataset).__name__ for dataset in measurement.datasets)
