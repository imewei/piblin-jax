"""
Measurement class for piblin-jax.

Container for multiple Dataset objects representing a single measurement event.
"""

from collections.abc import Iterator
from typing import Any

from piblin_jax.data.datasets import Dataset


class Measurement:
    """
    Container for multiple Dataset objects from a single measurement.

    A Measurement represents a single experimental measurement event that may
    produce multiple datasets (e.g., multiple channels, multiple observables).
    The collection is immutable for JAX compatibility.

    Parameters
    ----------
    datasets : list[Dataset]
        List of Dataset objects from this measurement.
    conditions : dict[str, Any] | None, optional
        Experimental conditions specific to this measurement
        (e.g., timestamp, replicate number, environmental conditions).
    details : dict[str, Any] | None, optional
        Additional context for this measurement
        (e.g., quality flags, operator notes, instrument state).

    Attributes
    ----------
    datasets : tuple[Dataset, ...]
        Immutable tuple of datasets from this measurement.
    conditions : dict[str, Any]
        Experimental conditions for this measurement.
    details : dict[str, Any]
        Additional metadata for this measurement.

    Notes
    -----
    The datasets are stored as a tuple to ensure immutability, which is
    required for JAX transformations. Individual datasets can be accessed
    by indexing or iteration.

    Examples
    --------
    >>> import numpy as np
    >>> from piblin_jax.data.datasets import OneDimensionalDataset
    >>> from piblin_jax.data.collections import Measurement
    >>>
    >>> # Create datasets for multiple channels
    >>> x = np.linspace(0, 10, 100)
    >>> y_ch1 = np.sin(x)
    >>> y_ch2 = np.cos(x)
    >>>
    >>> ds1 = OneDimensionalDataset(x, y_ch1, conditions={"channel": "A"})
    >>> ds2 = OneDimensionalDataset(x, y_ch2, conditions={"channel": "B"})
    >>>
    >>> # Create measurement with both channels
    >>> measurement = Measurement(
    ...     datasets=[ds1, ds2],
    ...     conditions={"temperature": 25.0, "replicate": 1},
    ...     details={"timestamp": "2025-10-18 10:00:00"}
    ... )
    >>>
    >>> # Access datasets
    >>> len(measurement)
    2
    >>> first_dataset = measurement[0]
    >>> for ds in measurement:
    ...     print(ds.conditions["channel"])
    A
    B
    """

    def __init__(
        self,
        datasets: list[Dataset],
        conditions: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
    ):
        """
        Initialize Measurement with datasets and metadata.

        Parameters
        ----------
        datasets : list[Dataset]
            List of Dataset objects from this measurement.
        conditions : dict[str, Any] | None, optional
            Experimental conditions for this measurement.
        details : dict[str, Any] | None, optional
            Additional context for this measurement.
        """
        self._datasets = tuple(datasets)  # Immutable for JAX compatibility
        self._conditions = conditions if conditions is not None else {}
        self._details = details if details is not None else {}

    @property
    def datasets(self) -> tuple[Dataset, ...]:
        """
        Get all datasets in this measurement.

        :no-index:

        Returns
        -------
        tuple[Dataset, ...]
            Immutable tuple of Dataset objects.

        Examples
        --------
        >>> measurement.datasets
        (<OneDimensionalDataset at 0x...>, <OneDimensionalDataset at 0x...>)
        """
        return self._datasets

    @property
    def conditions(self) -> dict[str, Any]:
        """
        Get experimental conditions for this measurement.

        :no-index:

        Returns
        -------
        dict[str, Any]
            Dictionary of experimental conditions (timestamp, replicate, etc.).

        Examples
        --------
        >>> measurement.conditions
        {'temperature': 25.0, 'replicate': 1, 'timestamp': '10:00:00'}
        """
        return self._conditions

    @property
    def details(self) -> dict[str, Any]:
        """
        Get additional details for this measurement.

        :no-index:

        Returns
        -------
        dict[str, Any]
            Dictionary of additional context (quality flags, notes, etc.).

        Examples
        --------
        >>> measurement.details
        {'quality': 'good', 'operator': 'John Doe'}
        """
        return self._details

    def __len__(self) -> int:
        """
        Get number of datasets in this measurement.

        Returns
        -------
        int
            Number of datasets.

        Examples
        --------
        >>> len(measurement)
        2
        """
        return len(self._datasets)

    def __iter__(self) -> Iterator[Dataset]:
        """
        Iterate over datasets in this measurement.

        Yields
        ------
        Dataset
            Each dataset in order.

        Examples
        --------
        >>> for dataset in measurement:
        ...     print(type(dataset).__name__)
        OneDimensionalDataset
        OneDimensionalDataset
        """
        return iter(self._datasets)

    def __getitem__(self, index: int | slice) -> Dataset | tuple[Dataset, ...]:
        """
        Get dataset by index.

        Parameters
        ----------
        index : int or slice
            Index or slice to access datasets.

        Returns
        -------
        Dataset or tuple[Dataset, ...]
            Dataset at the given index, or tuple of datasets for slice.

        Examples
        --------
        >>> measurement[0]
        <OneDimensionalDataset at 0x...>
        >>> measurement[0:2]
        (<OneDimensionalDataset at 0x...>, <OneDimensionalDataset at 0x...>)
        """
        return self._datasets[index]
