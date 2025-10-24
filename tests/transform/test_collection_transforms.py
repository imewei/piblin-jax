"""
Tests for collection-level transforms (Task Group 11).

Tests for:
- FilterDatasets: Filter datasets within a Measurement
- FilterMeasurements: Filter measurements within a MeasurementSet
- SplitByRegion: Split datasets by multiple regions
- MergeReplicates: Merge measurements with identical conditions
"""

import numpy as np
import pytest

from piblin_jax.data.collections import Measurement, MeasurementSet
from piblin_jax.data.datasets import OneDimensionalDataset, ZeroDimensionalDataset
from piblin_jax.data.roi import LinearRegion
from piblin_jax.transform.measurement import (
    FilterDatasets,
    FilterMeasurements,
    MergeReplicates,
    SplitByRegion,
)


class TestFilterDatasets:
    """Test FilterDatasets transform."""

    def test_filter_by_type(self):
        """Test filtering datasets by type."""
        # Create measurement with mixed dataset types
        x = np.linspace(0, 10, 50)
        y = np.sin(x)

        ds1 = OneDimensionalDataset(x, y, conditions={"channel": "A"})
        ds2 = ZeroDimensionalDataset(42.0, conditions={"channel": "B"})
        ds3 = OneDimensionalDataset(x, y * 2, conditions={"channel": "C"})

        measurement = Measurement(datasets=[ds1, ds2, ds3], conditions={"temp": 25.0})

        # Filter for only OneDimensionalDataset
        transform = FilterDatasets(dataset_type=OneDimensionalDataset)
        result = transform.apply_to(measurement)

        # Should have 2 datasets (ds1, ds3)
        assert len(result.datasets) == 2
        assert all(isinstance(ds, OneDimensionalDataset) for ds in result.datasets)
        assert result.conditions["temp"] == 25.0  # Metadata preserved

    def test_filter_by_predicate(self):
        """Test filtering datasets by predicate function."""
        # Create measurement with multiple datasets
        x = np.linspace(0, 10, 50)

        ds1 = OneDimensionalDataset(x, np.sin(x), conditions={"temp": 20.0})
        ds2 = OneDimensionalDataset(x, np.cos(x), conditions={"temp": 30.0})
        ds3 = OneDimensionalDataset(x, np.tan(x), conditions={"temp": 40.0})

        measurement = Measurement(datasets=[ds1, ds2, ds3])

        # Filter for datasets with temp > 25
        transform = FilterDatasets(predicate=lambda ds: ds.conditions.get("temp", 0) > 25.0)
        result = transform.apply_to(measurement)

        # Should have 2 datasets (ds2, ds3)
        assert len(result.datasets) == 2
        assert all(ds.conditions["temp"] > 25.0 for ds in result.datasets)

    def test_filter_requires_predicate_or_type(self):
        """Test that FilterDatasets requires either predicate or dataset_type."""
        with pytest.raises(ValueError, match="Must provide predicate or dataset_type"):
            FilterDatasets()

        with pytest.raises(ValueError, match="Provide only one"):
            FilterDatasets(predicate=lambda ds: True, dataset_type=OneDimensionalDataset)


class TestFilterMeasurements:
    """Test FilterMeasurements transform."""

    def test_filter_by_metadata(self):
        """Test filtering measurements by metadata."""
        x = np.linspace(0, 10, 50)

        # Create measurements with different temperatures
        m1 = Measurement(
            datasets=[OneDimensionalDataset(x, np.sin(x))],
            conditions={"temp": 20.0, "replicate": 1},
        )
        m2 = Measurement(
            datasets=[OneDimensionalDataset(x, np.cos(x))],
            conditions={"temp": 30.0, "replicate": 2},
        )
        m3 = Measurement(
            datasets=[OneDimensionalDataset(x, np.tan(x))],
            conditions={"temp": 40.0, "replicate": 3},
        )

        measurement_set = MeasurementSet(measurements=[m1, m2, m3])

        # Filter for temp > 25
        transform = FilterMeasurements(predicate=lambda m: m.conditions.get("temp", 0) > 25.0)
        result = transform.apply_to(measurement_set)

        # Should have 2 measurements (m2, m3)
        assert len(result.measurements) == 2
        assert all(m.conditions["temp"] > 25.0 for m in result.measurements)

    def test_filter_requires_callable(self):
        """Test that FilterMeasurements requires callable predicate."""
        with pytest.raises(TypeError, match="predicate must be callable"):
            FilterMeasurements(predicate="not_callable")


class TestSplitByRegion:
    """Test SplitByRegion transform."""

    def test_split_single_dataset(self):
        """Test splitting a single dataset by regions."""
        x = np.linspace(0, 10, 100)
        y = np.sin(x)

        ds = OneDimensionalDataset(x, y, conditions={"channel": "A"})
        measurement = Measurement(datasets=[ds], conditions={"temp": 25.0})

        # Define two regions
        regions = [LinearRegion(0, 5), LinearRegion(5, 10)]

        transform = SplitByRegion(regions)
        result = transform.apply_to(measurement)

        # Should have 2 datasets (one for each region)
        assert len(result.datasets) == 2

        # First dataset should have x in [0, 5]
        assert np.all(result.datasets[0].independent_variable_data <= 5)
        assert np.all(result.datasets[0].independent_variable_data >= 0)

        # Second dataset should have x in [5, 10]
        assert np.all(result.datasets[1].independent_variable_data >= 5)
        assert np.all(result.datasets[1].independent_variable_data <= 10)

        # Metadata preserved
        assert result.conditions["temp"] == 25.0

    def test_split_requires_regions(self):
        """Test that SplitByRegion requires at least one region."""
        with pytest.raises(ValueError, match="Must provide at least one region"):
            SplitByRegion(regions=[])


class TestMergeReplicates:
    """Test MergeReplicates transform."""

    def test_merge_with_averaging(self):
        """Test merging replicate measurements with averaging."""
        x = np.linspace(0, 10, 50)

        # Create replicate measurements with identical conditions
        m1 = Measurement(
            datasets=[OneDimensionalDataset(x, np.ones_like(x) * 1.0)],
            conditions={"temp": 25.0, "sample": "A"},
        )
        m2 = Measurement(
            datasets=[OneDimensionalDataset(x, np.ones_like(x) * 2.0)],
            conditions={"temp": 25.0, "sample": "A"},
        )
        m3 = Measurement(
            datasets=[OneDimensionalDataset(x, np.ones_like(x) * 3.0)],
            conditions={"temp": 25.0, "sample": "A"},
        )

        # Different condition - should not merge
        m4 = Measurement(
            datasets=[OneDimensionalDataset(x, np.ones_like(x) * 10.0)],
            conditions={"temp": 30.0, "sample": "B"},
        )

        measurement_set = MeasurementSet(measurements=[m1, m2, m3, m4])

        # Merge with averaging
        transform = MergeReplicates(strategy="average")
        result = transform.apply_to(measurement_set)

        # Should have 2 measurements (one merged, one standalone)
        assert len(result.measurements) == 2

        # Find the merged measurement
        merged = None
        for m in result.measurements:
            if m.conditions.get("temp") == 25.0:
                merged = m
                break

        assert merged is not None
        # Average of [1, 2, 3] should be 2.0
        y_avg = merged.datasets[0].dependent_variable_data
        np.testing.assert_array_almost_equal(y_avg, np.ones_like(x) * 2.0)

    def test_merge_invalid_strategy(self):
        """Test that invalid strategy raises error."""
        with pytest.raises(ValueError, match="strategy must be"):
            MergeReplicates(strategy="invalid")
