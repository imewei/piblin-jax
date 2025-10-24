"""
TidyMeasurementSet class for piblin-jax.

MeasurementSet variant where measurements share comparable experimental conditions.
"""

from typing import Any

from .measurement import Measurement
from .measurement_set import MeasurementSet


class TidyMeasurementSet(MeasurementSet):
    """
    MeasurementSet where measurements share comparable experimental conditions.

    This specialized variant is designed for measurements that can be compared
    across shared experimental conditions, following "tidy data" principles.
    This is useful for:
    - Parameter sweeps (varying temperature, pressure, etc.)
    - Multi-factor experiments (factorial designs)
    - Grouped experimental conditions
    - Long-form data representation

    The shared condition structure enables statistical analysis, grouping,
    and faceted visualization.

    Parameters
    ----------
    measurements : list[Measurement]
        List of Measurement objects with comparable conditions.
    conditions : dict[str, Any] | None, optional
        Experimental conditions for the measurement series.
    details : dict[str, Any] | None, optional
        Additional context for the measurement series.

    Notes
    -----
    "Tidy data" refers to a data organization principle where:
    - Each measurement is an observation
    - Each condition is a variable
    - Each unique condition value identifies a group

    This enables standard statistical and data manipulation tools to work
    effectively with the measurement set.

    Examples
    --------
    >>> import numpy as np
    >>> from piblin_jax.data.datasets import OneDimensionalDataset
    >>> from piblin_jax.data.collections import Measurement, TidyMeasurementSet
    >>>
    >>> # Create measurements with varying conditions
    >>> x = np.linspace(0, 10, 100)
    >>> measurements = []
    >>>
    >>> for temp in [20, 25, 30]:
    ...     for sample in ['A', 'B']:
    ...         y = np.sin(x) * temp / 25
    ...         ds = OneDimensionalDataset(x, y)
    ...         m = Measurement(
    ...             [ds],
    ...             conditions={"temperature": temp, "sample": sample}
    ...         )
    ...         measurements.append(m)
    >>>
    >>> # Create tidy measurement set
    >>> tms = TidyMeasurementSet(
    ...     measurements=measurements,
    ...     conditions={"experiment": "temperature_sweep"},
    ...     details={"date": "2025-10-18"}
    ... )
    >>>
    >>> len(tms)
    6
    >>>
    >>> # Get unique condition values
    >>> unique = tms.get_unique_conditions()
    >>> sorted(unique["temperature"])
    [20, 25, 30]
    >>> sorted(unique["sample"])
    ['A', 'B']
    """

    def __init__(
        self,
        measurements: list[Measurement],
        conditions: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
    ):
        """
        Initialize TidyMeasurementSet.

        Parameters
        ----------
        measurements : list[Measurement]
            List of Measurement objects with comparable conditions.
        conditions : dict[str, Any] | None, optional
            Experimental conditions for this measurement series.
        details : dict[str, Any] | None, optional
            Additional context for this measurement series.
        """
        # Call parent constructor
        # No validation needed - any measurements can be tidy
        super().__init__(measurements, conditions, details)

    def get_unique_conditions(self) -> dict[str, set[Any]]:
        """
        Get all unique values for each condition across measurements.

        This method analyzes all measurements and returns the set of unique
        values for each condition key. This is useful for:
        - Understanding the experimental design
        - Identifying factor levels
        - Grouping measurements
        - Creating faceted plots

        Returns
        -------
        dict[str, set]
            Dictionary mapping condition names to sets of unique values.

        Examples
        --------
        >>> # Continuing from class docstring example
        >>> unique = tms.get_unique_conditions()
        >>> unique["temperature"]
        {20, 25, 30}
        >>> unique["sample"]
        {'A', 'B'}
        >>>
        >>> # Empty measurement set
        >>> tms_empty = TidyMeasurementSet([])
        >>> tms_empty.get_unique_conditions()
        {}
        >>>
        >>> # Measurements with different condition keys
        >>> m1 = Measurement([OneDimensionalDataset(np.array([1]), np.array([2]))],
        ...                  conditions={"temp": 25, "pressure": 1.0})
        >>> m2 = Measurement([OneDimensionalDataset(np.array([3]), np.array([4]))],
        ...                  conditions={"temp": 30, "sample": "A"})
        >>> tms = TidyMeasurementSet([m1, m2])
        >>> unique = tms.get_unique_conditions()
        >>> sorted(unique.keys())
        ['pressure', 'sample', 'temp']
        >>> unique["temp"]
        {25, 30}
        """
        unique_conditions: dict[str, set[Any]] = {}

        for measurement in self.measurements:
            for key, value in measurement.conditions.items():
                if key not in unique_conditions:
                    unique_conditions[key] = set()
                # Handle unhashable types by converting to string
                try:
                    unique_conditions[key].add(value)
                except TypeError:
                    # If value is unhashable (e.g., list, dict), convert to string
                    unique_conditions[key].add(str(value))

        return unique_conditions

    def filter_by_conditions(self, **condition_filters: Any) -> "TidyMeasurementSet":
        """
        Create a new TidyMeasurementSet with measurements matching conditions.

        Parameters
        ----------
        **condition_filters
            Keyword arguments specifying condition values to match.
            Only measurements where ALL specified conditions match
            the given values will be included.

        Returns
        -------
        TidyMeasurementSet
            New TidyMeasurementSet containing only matching measurements.

        Examples
        --------
        >>> # Filter by single condition
        >>> tms_25 = tms.filter_by_conditions(temperature=25)
        >>> len(tms_25)
        2
        >>> all(m.conditions["temperature"] == 25 for m in tms_25)
        True
        >>>
        >>> # Filter by multiple conditions
        >>> tms_25_A = tms.filter_by_conditions(temperature=25, sample="A")
        >>> len(tms_25_A)
        1
        >>> m = tms_25_A[0]
        >>> m.conditions["temperature"]
        25
        >>> m.conditions["sample"]
        'A'
        """
        filtered = []

        for measurement in self.measurements:
            # Check if all specified conditions match
            matches = all(
                measurement.conditions.get(key) == value for key, value in condition_filters.items()
            )
            if matches:
                filtered.append(measurement)

        return TidyMeasurementSet(
            measurements=filtered,
            conditions=self.conditions,
            details=self.details,
        )
