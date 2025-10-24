"""
Experiment class for piblin-jax.

Container for multiple MeasurementSet objects representing a single experiment.
"""

from collections.abc import Iterator
from typing import Any

from .measurement_set import MeasurementSet


class Experiment:
    """
    Container for MeasurementSet objects from a single experiment.

    An Experiment represents a complete experimental run or sample, which may
    contain multiple series of measurements (MeasurementSets). This is useful for:
    - Single sample with multiple measurement types
    - Complete experimental protocol with multiple phases
    - Single experimental run with multiple observables
    - One sample measured under different conditions

    Parameters
    ----------
    measurement_sets : list[MeasurementSet]
        List of MeasurementSet objects from this experiment.
    conditions : dict[str, Any] | None, optional
        Experimental conditions for the entire experiment
        (e.g., sample ID, experimental date, operator).
    details : dict[str, Any] | None, optional
        Additional context for this experiment
        (e.g., sample description, experimental notes, quality flags).

    Attributes
    ----------
    measurement_sets : tuple[MeasurementSet, ...]
        Immutable tuple of measurement sets in this experiment.
    conditions : dict[str, Any]
        Experimental conditions for this experiment.
    details : dict[str, Any]
        Additional metadata for this experiment.

    Notes
    -----
    The measurement sets are stored as a tuple to ensure immutability, which is
    required for JAX transformations. Individual measurement sets can be accessed
    by indexing or iteration.

    Hierarchy level:
    ExperimentSet → **Experiment** → MeasurementSet → Measurement → Dataset

    Examples
    --------
    >>> import numpy as np
    >>> from piblin_jax.data.datasets import OneDimensionalDataset
    >>> from piblin_jax.data.collections import (
    ...     Measurement, MeasurementSet, Experiment
    ... )
    >>>
    >>> # Create first measurement set (absorption spectra)
    >>> x_abs = np.linspace(400, 800, 200)
    >>> measurements_abs = []
    >>> for i in range(3):
    ...     y = np.exp(-(x_abs - 550)**2 / 1000) * (1 + 0.1 * i)
    ...     ds = OneDimensionalDataset(x_abs, y)
    ...     m = Measurement([ds], conditions={"replicate": i+1})
    ...     measurements_abs.append(m)
    >>> ms_abs = MeasurementSet(
    ...     measurements_abs,
    ...     conditions={"measurement_type": "absorption"}
    ... )
    >>>
    >>> # Create second measurement set (fluorescence spectra)
    >>> x_fl = np.linspace(500, 900, 200)
    >>> measurements_fl = []
    >>> for i in range(3):
    ...     y = np.exp(-(x_fl - 650)**2 / 1500) * (0.8 + 0.1 * i)
    ...     ds = OneDimensionalDataset(x_fl, y)
    ...     m = Measurement([ds], conditions={"replicate": i+1})
    ...     measurements_fl.append(m)
    >>> ms_fl = MeasurementSet(
    ...     measurements_fl,
    ...     conditions={"measurement_type": "fluorescence"}
    ... )
    >>>
    >>> # Create experiment combining both measurement types
    >>> exp = Experiment(
    ...     measurement_sets=[ms_abs, ms_fl],
    ...     conditions={"sample": "S001", "date": "2025-10-18"},
    ...     details={"operator": "Jane Doe", "instrument": "Spec-X"}
    ... )
    >>>
    >>> len(exp)
    2
    >>> exp.conditions["sample"]
    'S001'
    >>> exp[0].conditions["measurement_type"]
    'absorption'
    >>> exp[1].conditions["measurement_type"]
    'fluorescence'
    """

    def __init__(
        self,
        measurement_sets: list[MeasurementSet],
        conditions: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
    ):
        """
        Initialize Experiment with measurement sets and metadata.

        Parameters
        ----------
        measurement_sets : list[MeasurementSet]
            List of MeasurementSet objects from this experiment.
        conditions : dict[str, Any] | None, optional
            Experimental conditions for this experiment.
        details : dict[str, Any] | None, optional
            Additional context for this experiment.
        """
        self._measurement_sets = tuple(measurement_sets)  # Immutable for JAX
        self._conditions = conditions if conditions is not None else {}
        self._details = details if details is not None else {}

    @property
    def measurement_sets(self) -> tuple[MeasurementSet, ...]:
        """
        Get all measurement sets in this experiment.

        :no-index:

        Returns
        -------
        tuple[MeasurementSet, ...]
            Immutable tuple of MeasurementSet objects.

        Examples
        --------
        >>> exp.measurement_sets
        (<MeasurementSet at 0x...>, <MeasurementSet at 0x...>)
        """
        return self._measurement_sets

    @property
    def conditions(self) -> dict[str, Any]:
        """
        Get experimental conditions for this experiment.

        :no-index:

        Returns
        -------
        dict[str, Any]
            Dictionary of experimental conditions (sample, date, operator, etc.).

        Examples
        --------
        >>> exp.conditions
        {'sample': 'S001', 'date': '2025-10-18', 'temperature': 25.0}
        """
        return self._conditions

    @property
    def details(self) -> dict[str, Any]:
        """
        Get additional details for this experiment.

        :no-index:

        Returns
        -------
        dict[str, Any]
            Dictionary of additional context (notes, quality, instrument, etc.).

        Examples
        --------
        >>> exp.details
        {'operator': 'Jane Doe', 'instrument': 'Spec-X', 'notes': 'Good quality'}
        """
        return self._details

    def __len__(self) -> int:
        """
        Get number of measurement sets in this experiment.

        Returns
        -------
        int
            Number of measurement sets.

        Examples
        --------
        >>> len(exp)
        2
        """
        return len(self._measurement_sets)

    def __iter__(self) -> Iterator[MeasurementSet]:
        """
        Iterate over measurement sets in this experiment.

        Yields
        ------
        MeasurementSet
            Each measurement set in order.

        Examples
        --------
        >>> for ms in exp:
        ...     print(ms.conditions["measurement_type"])
        absorption
        fluorescence
        """
        return iter(self._measurement_sets)

    def __getitem__(self, index: int | slice) -> MeasurementSet | tuple[MeasurementSet, ...]:
        """
        Get measurement set by index.

        Parameters
        ----------
        index : int or slice
            Index or slice to access measurement sets.

        Returns
        -------
        MeasurementSet or tuple[MeasurementSet, ...]
            MeasurementSet at the given index, or tuple for slice.

        Examples
        --------
        >>> exp[0]
        <MeasurementSet at 0x...>
        >>> exp[0:2]
        (<MeasurementSet at 0x...>, <MeasurementSet at 0x...>)
        """
        return self._measurement_sets[index]
