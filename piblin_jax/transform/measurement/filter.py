"""
Collection-level transforms for Measurements and MeasurementSets.

This module provides transforms that operate on collections:
- FilterDatasets: Filter datasets within a Measurement
- FilterMeasurements: Filter measurements within a MeasurementSet
- SplitByRegion: Split datasets by regions
- MergeReplicates: Merge measurements with identical conditions
"""

from collections.abc import Callable
from typing import Any

from piblin_jax.backend import jnp
from piblin_jax.data.collections import Measurement, MeasurementSet
from piblin_jax.data.datasets import Dataset, OneDimensionalDataset
from piblin_jax.data.roi import LinearRegion
from piblin_jax.transform.base import MeasurementSetTransform, MeasurementTransform


class FilterDatasets(MeasurementTransform):
    """
    Filter datasets within a Measurement.

    This transform filters datasets based on either their type or
    a custom predicate function. Returns a new Measurement containing
    only the datasets that match the filter criteria.

    Parameters
    ----------
    predicate : Callable[[Dataset], bool] | None
        Function that returns True for datasets to keep.
        Cannot be used with dataset_type.
    dataset_type : type | None
        Filter by dataset type (alternative to predicate).
        Cannot be used with predicate.

    Raises
    ------
    ValueError
        If neither predicate nor dataset_type is provided.
        If both predicate and dataset_type are provided.

    Examples
    --------
    >>> from piblin_jax.data.datasets import OneDimensionalDataset
    >>> from piblin_jax.transform.measurement import FilterDatasets
    >>>
    >>> # Filter by type
    >>> transform = FilterDatasets(dataset_type=OneDimensionalDataset)
    >>> result = transform.apply_to(measurement)
    >>>
    >>> # Filter by predicate
    >>> transform = FilterDatasets(
    ...     predicate=lambda ds: ds.conditions.get('temp', 0) > 25
    ... )
    >>> result = transform.apply_to(measurement)

    Notes
    -----
    - Returns a new Measurement with filtered datasets
    - Preserves measurement-level conditions and details
    - Empty dataset list is allowed if no datasets match
    """

    def __init__(
        self, predicate: Callable[[Dataset], bool] | None = None, dataset_type: type | None = None
    ):
        """
        Initialize FilterDatasets transform.

        Parameters
        ----------
        predicate : Callable[[Dataset], bool] | None
            Function that returns True for datasets to keep
        dataset_type : type | None
            Filter by dataset type
        """
        super().__init__()

        if predicate is None and dataset_type is None:
            raise ValueError("Must provide predicate or dataset_type")

        if predicate is not None and dataset_type is not None:
            raise ValueError("Provide only one of predicate or dataset_type")

        if dataset_type is not None:
            self.predicate: Callable[[Dataset], bool] = lambda ds: isinstance(ds, dataset_type)
        else:
            self.predicate = predicate  # type: ignore[assignment]

    def _apply(self, measurement: Measurement) -> Measurement:
        """
        Filter datasets and return new Measurement.

        Parameters
        ----------
        measurement : Measurement
            Input measurement

        Returns
        -------
        Measurement
            New measurement with filtered datasets
        """
        filtered_datasets = [ds for ds in measurement.datasets if self.predicate(ds)]

        return Measurement(
            datasets=filtered_datasets,
            conditions=measurement.conditions,
            details=measurement.details,
        )


class FilterMeasurements(MeasurementSetTransform):
    """
    Filter measurements within a MeasurementSet.

    This transform filters measurements based on a predicate function.
    Returns a new MeasurementSet containing only the measurements that
    match the filter criteria.

    Parameters
    ----------
    predicate : Callable[[Measurement], bool]
        Function that returns True for measurements to keep

    Raises
    ------
    TypeError
        If predicate is not callable

    Examples
    --------
    >>> from piblin_jax.transform.measurement import FilterMeasurements
    >>>
    >>> # Filter by condition
    >>> transform = FilterMeasurements(
    ...     predicate=lambda m: m.conditions.get('temp', 0) > 25
    ... )
    >>> result = transform.apply_to(measurement_set)
    >>>
    >>> # Filter by replicate number
    >>> transform = FilterMeasurements(
    ...     predicate=lambda m: m.conditions.get('replicate', 0) <= 3
    ... )
    >>> result = transform.apply_to(measurement_set)

    Notes
    -----
    - Returns a new MeasurementSet with filtered measurements
    - Preserves measurement-set-level conditions and details
    - Empty measurement list is allowed if no measurements match
    """

    def __init__(self, predicate: Callable[[Measurement], bool]):
        """
        Initialize FilterMeasurements transform.

        Parameters
        ----------
        predicate : Callable[[Measurement], bool]
            Function that returns True for measurements to keep
        """
        super().__init__()
        if not callable(predicate):
            raise TypeError("predicate must be callable")
        self.predicate = predicate

    def _apply(self, measurement_set: MeasurementSet) -> MeasurementSet:
        """
        Filter measurements and return new MeasurementSet.

        Parameters
        ----------
        measurement_set : MeasurementSet
            Input measurement set

        Returns
        -------
        MeasurementSet
            New measurement set with filtered measurements
        """
        filtered_measurements = [m for m in measurement_set.measurements if self.predicate(m)]

        return MeasurementSet(
            measurements=filtered_measurements,
            conditions=measurement_set.conditions,
            details=measurement_set.details,
        )


class SplitByRegion(MeasurementTransform):
    """
    Split datasets by multiple regions, creating new Measurement.

    This transform splits each OneDimensionalDataset in a Measurement
    into multiple datasets based on the specified regions. Each region
    creates a new dataset containing only the data points within that region.

    Parameters
    ----------
    regions : list[LinearRegion]
        List of regions to split by

    Raises
    ------
    ValueError
        If regions list is empty

    Examples
    --------
    >>> from piblin_jax.data.roi import LinearRegion
    >>> from piblin_jax.transform.measurement import SplitByRegion
    >>>
    >>> regions = [
    ...     LinearRegion(0, 5),
    ...     LinearRegion(5, 10)
    ... ]
    >>> transform = SplitByRegion(regions)
    >>> result = transform.apply_to(measurement)

    Notes
    -----
    - Only processes OneDimensionalDataset objects
    - Non-1D datasets are silently skipped
    - Empty regions (no data points) are included as empty datasets
    - Preserves dataset-level conditions and details for each split
    - Preserves measurement-level conditions and details
    """

    def __init__(self, regions: list[LinearRegion]):
        """
        Initialize SplitByRegion transform.

        Parameters
        ----------
        regions : list[LinearRegion]
            List of regions to split by
        """
        super().__init__()
        if not regions:
            raise ValueError("Must provide at least one region")
        self.regions = regions

    def _apply(self, measurement: Measurement) -> Measurement:
        """
        Split each dataset by regions.

        Parameters
        ----------
        measurement : Measurement
            Input measurement

        Returns
        -------
        Measurement
            New measurement with split datasets
        """
        new_datasets = []

        for dataset in measurement.datasets:
            if isinstance(dataset, OneDimensionalDataset):
                # Split this dataset by all regions
                x_data = dataset.independent_variable_data
                y_data = dataset.dependent_variable_data

                for region in self.regions:
                    mask = region.get_mask(x_data)
                    x_region = x_data[mask]
                    y_region = y_data[mask]

                    if len(x_region) > 0:
                        new_ds = OneDimensionalDataset(
                            independent_variable_data=x_region,
                            dependent_variable_data=y_region,
                            conditions=dataset.conditions,
                            details=dataset.details,
                        )
                        new_datasets.append(new_ds)

        return Measurement(
            datasets=new_datasets,  # type: ignore[arg-type]
            conditions=measurement.conditions,
            details=measurement.details,
        )


class MergeReplicates(MeasurementSetTransform):
    """
    Merge measurements with identical conditions.

    This transform groups measurements by their conditions and merges
    replicates (measurements with identical conditions) using either
    averaging or concatenation.

    Parameters
    ----------
    strategy : str, default='average'
        Merge strategy: 'average' or 'concatenate'

    Raises
    ------
    ValueError
        If strategy is not 'average' or 'concatenate'

    Examples
    --------
    >>> from piblin_jax.transform.measurement import MergeReplicates
    >>>
    >>> # Average replicate measurements
    >>> transform = MergeReplicates(strategy='average')
    >>> result = transform.apply_to(measurement_set)

    Notes
    -----
    - Groups measurements by conditions (all key-value pairs must match)
    - For 'average' strategy:
      - Averages dependent variable data across replicates
      - Assumes all replicates have same independent variable data
      - Assumes all replicates have same dataset structure
    - For 'concatenate' strategy:
      - Currently returns first measurement (not yet implemented)
    - Preserves measurement-set-level conditions and details
    - Single measurements (no replicates) are returned unchanged
    """

    def __init__(self, strategy: str = "average"):
        """
        Initialize MergeReplicates transform.

        Parameters
        ----------
        strategy : str, default='average'
            Merge strategy: 'average' or 'concatenate'
        """
        super().__init__()
        if strategy not in ["average", "concatenate"]:
            raise ValueError("strategy must be 'average' or 'concatenate'")
        self.strategy = strategy

    def _apply(self, measurement_set: MeasurementSet) -> MeasurementSet:
        """
        Merge replicate measurements.

        Parameters
        ----------
        measurement_set : MeasurementSet
            Input measurement set

        Returns
        -------
        MeasurementSet
            New measurement set with merged measurements
        """
        # Group by conditions
        groups: dict[tuple[tuple[str, Any], ...], list[Measurement]] = {}
        for measurement in measurement_set.measurements:
            # Create hashable key from conditions
            key = tuple(sorted(measurement.conditions.items()))
            if key not in groups:
                groups[key] = []
            groups[key].append(measurement)

        # Merge each group
        merged_measurements = []
        for _key, group in groups.items():
            if len(group) == 1:
                merged_measurements.append(group[0])
            else:
                merged = self._merge_group(group)
                merged_measurements.append(merged)

        return MeasurementSet(
            measurements=merged_measurements,
            conditions=measurement_set.conditions,
            details=measurement_set.details,
        )

    def _merge_group(self, measurements: list[Measurement]) -> Measurement:
        """
        Merge a group of measurements with identical conditions.

        Parameters
        ----------
        measurements : list[Measurement]
            List of measurements to merge

        Returns
        -------
        Measurement
            Merged measurement
        """
        if self.strategy == "average":
            # Average the dependent variable data
            # Assumes all measurements have same structure
            first = measurements[0]

            # Average each dataset
            averaged_datasets = []
            for i, dataset in enumerate(first.datasets):
                if isinstance(dataset, OneDimensionalDataset):
                    # Collect y values from all replicates
                    y_values = []
                    for measurement in measurements:
                        if i < len(measurement.datasets):
                            ds = measurement.datasets[i]
                            if isinstance(ds, OneDimensionalDataset):
                                y_values.append(ds.dependent_variable_data)

                    # Average
                    y_avg = jnp.mean(jnp.stack(y_values), axis=0)

                    averaged_ds = OneDimensionalDataset(
                        independent_variable_data=dataset.independent_variable_data,
                        dependent_variable_data=y_avg,
                        conditions=dataset.conditions,
                        details=dataset.details,
                    )
                    averaged_datasets.append(averaged_ds)

            return Measurement(
                datasets=averaged_datasets,  # type: ignore[arg-type]
                conditions=first.conditions,
                details=first.details,
            )
        else:
            # Concatenate not implemented yet
            return measurements[0]


__all__ = [
    "FilterDatasets",
    "FilterMeasurements",
    "MergeReplicates",
    "SplitByRegion",
]
