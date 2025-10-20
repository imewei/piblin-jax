"""
Tests for hierarchical collection classes.

Tests the complete hierarchy:
ExperimentSet → Experiment → MeasurementSet → Measurement → Dataset
"""

import numpy as np
import pytest

from quantiq.data.collections import (
    ConsistentMeasurementSet,
    Experiment,
    ExperimentSet,
    Measurement,
    MeasurementSet,
    TabularMeasurementSet,
    TidyMeasurementSet,
)
from quantiq.data.datasets import (
    OneDimensionalDataset,
    TwoDimensionalDataset,
    ZeroDimensionalDataset,
)


class TestMeasurement:
    """Test Measurement container for multiple datasets."""

    def test_measurement_creation_with_multiple_datasets(self):
        """Test creating a Measurement with multiple Dataset objects."""
        # Create sample datasets
        x = np.linspace(0, 10, 100)
        y1 = np.sin(x)
        y2 = np.cos(x)

        ds1 = OneDimensionalDataset(
            independent_variable_data=x,
            dependent_variable_data=y1,
            conditions={"channel": "A"},
        )
        ds2 = OneDimensionalDataset(
            independent_variable_data=x,
            dependent_variable_data=y2,
            conditions={"channel": "B"},
        )

        # Create measurement
        conditions = {"temperature": 25.0, "pressure": 1.0}
        details = {"operator": "John Doe", "date": "2025-10-18"}
        measurement = Measurement(
            datasets=[ds1, ds2],
            conditions=conditions,
            details=details,
        )

        # Verify properties
        assert len(measurement) == 2
        assert measurement.conditions == conditions
        assert measurement.details == details
        assert isinstance(measurement.datasets, tuple)  # Immutable

    def test_measurement_iteration_and_indexing(self):
        """Test iteration and indexing support for Measurement."""
        x = np.array([1, 2, 3])
        y1 = np.array([4, 5, 6])
        y2 = np.array([7, 8, 9])

        ds1 = OneDimensionalDataset(x, y1)
        ds2 = OneDimensionalDataset(x, y2)

        measurement = Measurement(datasets=[ds1, ds2])

        # Test indexing
        assert measurement[0] is ds1
        assert measurement[1] is ds2

        # Test iteration
        datasets_list = list(measurement)
        assert len(datasets_list) == 2
        assert datasets_list[0] is ds1
        assert datasets_list[1] is ds2

    def test_measurement_immutability(self):
        """Test that datasets collection is immutable."""
        ds1 = OneDimensionalDataset(np.array([1, 2]), np.array([3, 4]))
        measurement = Measurement(datasets=[ds1])

        # Should not be able to modify datasets tuple
        with pytest.raises((TypeError, AttributeError)):
            measurement.datasets[0] = None


class TestMeasurementSetBase:
    """Test base MeasurementSet class."""

    def test_measurement_set_creation(self):
        """Test creating a MeasurementSet with multiple Measurements."""
        # Create measurements
        m1 = Measurement(
            datasets=[OneDimensionalDataset(np.array([1, 2]), np.array([3, 4]))],
            conditions={"replicate": 1},
        )
        m2 = Measurement(
            datasets=[OneDimensionalDataset(np.array([1, 2]), np.array([5, 6]))],
            conditions={"replicate": 2},
        )

        # Create measurement set
        conditions = {"experiment": "A", "sample": "S1"}
        details = {"notes": "Test run"}
        ms = MeasurementSet(
            measurements=[m1, m2],
            conditions=conditions,
            details=details,
        )

        # Verify properties
        assert len(ms) == 2
        assert ms.conditions == conditions
        assert ms.details == details
        assert isinstance(ms.measurements, tuple)  # Immutable

    def test_measurement_set_iteration(self):
        """Test iteration over MeasurementSet."""
        m1 = Measurement(datasets=[OneDimensionalDataset(np.array([1]), np.array([2]))])
        m2 = Measurement(datasets=[OneDimensionalDataset(np.array([3]), np.array([4]))])

        ms = MeasurementSet(measurements=[m1, m2])

        measurements_list = list(ms)
        assert len(measurements_list) == 2
        assert measurements_list[0] is m1
        assert measurements_list[1] is m2


class TestConsistentMeasurementSet:
    """Test ConsistentMeasurementSet with structural validation."""

    def test_consistent_measurement_set_valid(self):
        """Test ConsistentMeasurementSet with consistent structure."""
        # All measurements have same structure: one 1D dataset
        m1 = Measurement(datasets=[OneDimensionalDataset(np.array([1, 2]), np.array([3, 4]))])
        m2 = Measurement(datasets=[OneDimensionalDataset(np.array([5, 6]), np.array([7, 8]))])
        m3 = Measurement(datasets=[OneDimensionalDataset(np.array([9, 10]), np.array([11, 12]))])

        # Should succeed - all have same structure
        cms = ConsistentMeasurementSet(measurements=[m1, m2, m3])

        assert len(cms) == 3
        assert isinstance(cms, MeasurementSet)  # Inherits from base

    def test_consistent_measurement_set_invalid(self):
        """Test ConsistentMeasurementSet rejects inconsistent structure."""
        # First measurement: one 1D dataset
        m1 = Measurement(datasets=[OneDimensionalDataset(np.array([1, 2]), np.array([3, 4]))])

        # Second measurement: one 0D dataset (different structure)
        m2 = Measurement(datasets=[ZeroDimensionalDataset(value=5.0)])

        # Should fail - different structures
        with pytest.raises(ValueError, match="same structure"):
            ConsistentMeasurementSet(measurements=[m1, m2])

    def test_consistent_measurement_set_different_dataset_count(self):
        """Test ConsistentMeasurementSet with different dataset counts."""
        # First measurement: one dataset
        m1 = Measurement(datasets=[OneDimensionalDataset(np.array([1]), np.array([2]))])

        # Second measurement: two datasets (different structure)
        m2 = Measurement(
            datasets=[
                OneDimensionalDataset(np.array([1]), np.array([2])),
                OneDimensionalDataset(np.array([3]), np.array([4])),
            ]
        )

        # Should fail - different number of datasets
        with pytest.raises(ValueError):
            ConsistentMeasurementSet(measurements=[m1, m2])


class TestTidyMeasurementSet:
    """Test TidyMeasurementSet for comparable measurements."""

    def test_tidy_measurement_set_creation(self):
        """Test TidyMeasurementSet creation and unique conditions."""
        # Create measurements with varying conditions
        m1 = Measurement(
            datasets=[OneDimensionalDataset(np.array([1, 2]), np.array([3, 4]))],
            conditions={"temperature": 25.0, "sample": "A"},
        )
        m2 = Measurement(
            datasets=[OneDimensionalDataset(np.array([1, 2]), np.array([5, 6]))],
            conditions={"temperature": 30.0, "sample": "A"},
        )
        m3 = Measurement(
            datasets=[OneDimensionalDataset(np.array([1, 2]), np.array([7, 8]))],
            conditions={"temperature": 25.0, "sample": "B"},
        )

        tms = TidyMeasurementSet(measurements=[m1, m2, m3])

        assert len(tms) == 3
        assert isinstance(tms, MeasurementSet)

    def test_tidy_measurement_set_unique_conditions(self):
        """Test getting unique condition values."""
        m1 = Measurement(
            datasets=[OneDimensionalDataset(np.array([1]), np.array([2]))],
            conditions={"temperature": 25.0, "sample": "A"},
        )
        m2 = Measurement(
            datasets=[OneDimensionalDataset(np.array([3]), np.array([4]))],
            conditions={"temperature": 30.0, "sample": "A"},
        )
        m3 = Measurement(
            datasets=[OneDimensionalDataset(np.array([5]), np.array([6]))],
            conditions={"temperature": 25.0, "sample": "B"},
        )

        tms = TidyMeasurementSet(measurements=[m1, m2, m3])
        unique = tms.get_unique_conditions()

        assert "temperature" in unique
        assert "sample" in unique
        assert unique["temperature"] == {25.0, 30.0}
        assert unique["sample"] == {"A", "B"}


class TestTabularMeasurementSet:
    """Test TabularMeasurementSet with tabular access patterns."""

    def test_tabular_measurement_set_with_labels(self):
        """Test TabularMeasurementSet with row and column labels."""
        measurements = [
            Measurement(datasets=[OneDimensionalDataset(np.array([i]), np.array([i * 2]))])
            for i in range(6)
        ]

        row_labels = ["row1", "row2"]
        col_labels = ["col1", "col2", "col3"]

        tms = TabularMeasurementSet(
            measurements=measurements,
            row_labels=row_labels,
            col_labels=col_labels,
        )

        assert len(tms) == 6
        assert tms.row_labels == row_labels
        assert tms.col_labels == col_labels
        assert isinstance(tms, MeasurementSet)


class TestExperiment:
    """Test Experiment container for MeasurementSets."""

    def test_experiment_creation(self):
        """Test creating Experiment with multiple MeasurementSets."""
        # Create measurement sets
        m1 = Measurement(datasets=[OneDimensionalDataset(np.array([1, 2]), np.array([3, 4]))])
        m2 = Measurement(datasets=[OneDimensionalDataset(np.array([5, 6]), np.array([7, 8]))])

        ms1 = MeasurementSet(measurements=[m1], conditions={"series": 1})
        ms2 = MeasurementSet(measurements=[m2], conditions={"series": 2})

        # Create experiment
        conditions = {"sample": "S1", "date": "2025-10-18"}
        details = {"operator": "Jane Doe"}
        exp = Experiment(
            measurement_sets=[ms1, ms2],
            conditions=conditions,
            details=details,
        )

        # Verify properties
        assert len(exp) == 2
        assert exp.conditions == conditions
        assert exp.details == details
        assert isinstance(exp.measurement_sets, tuple)  # Immutable

    def test_experiment_iteration(self):
        """Test iteration over Experiment."""
        m1 = Measurement(datasets=[OneDimensionalDataset(np.array([1]), np.array([2]))])
        ms1 = MeasurementSet(measurements=[m1])
        ms2 = MeasurementSet(measurements=[m1])

        exp = Experiment(measurement_sets=[ms1, ms2])

        ms_list = list(exp)
        assert len(ms_list) == 2
        assert ms_list[0] is ms1
        assert ms_list[1] is ms2


class TestExperimentSet:
    """Test ExperimentSet top-level container."""

    def test_experiment_set_creation(self):
        """Test creating ExperimentSet with multiple Experiments."""
        # Create nested hierarchy
        m1 = Measurement(datasets=[OneDimensionalDataset(np.array([1, 2]), np.array([3, 4]))])
        ms1 = MeasurementSet(measurements=[m1])
        exp1 = Experiment(measurement_sets=[ms1], conditions={"sample": "S1"})
        exp2 = Experiment(measurement_sets=[ms1], conditions={"sample": "S2"})

        # Create experiment set
        conditions = {"project": "QuantIQ-2025", "instrument": "Spectrometer-X"}
        details = {"pi": "Dr. Smith", "grant": "NSF-12345"}
        exp_set = ExperimentSet(
            experiments=[exp1, exp2],
            conditions=conditions,
            details=details,
        )

        # Verify properties
        assert len(exp_set) == 2
        assert exp_set.conditions == conditions
        assert exp_set.details == details
        assert isinstance(exp_set.experiments, tuple)  # Immutable

    def test_experiment_set_iteration(self):
        """Test iteration over ExperimentSet."""
        m1 = Measurement(datasets=[OneDimensionalDataset(np.array([1]), np.array([2]))])
        ms1 = MeasurementSet(measurements=[m1])
        exp1 = Experiment(measurement_sets=[ms1])
        exp2 = Experiment(measurement_sets=[ms1])

        exp_set = ExperimentSet(experiments=[exp1, exp2])

        exp_list = list(exp_set)
        assert len(exp_list) == 2
        assert exp_list[0] is exp1
        assert exp_list[1] is exp2


class TestMetadataPropagation:
    """Test metadata propagation through the hierarchy."""

    def test_complete_hierarchy_with_metadata(self):
        """Test complete hierarchy construction with metadata at each level."""
        # Dataset level
        ds = OneDimensionalDataset(
            independent_variable_data=np.linspace(0, 10, 100),
            dependent_variable_data=np.sin(np.linspace(0, 10, 100)),
            conditions={"wavelength": 550},
            details={"units": "nm"},
        )

        # Measurement level
        measurement = Measurement(
            datasets=[ds],
            conditions={"replicate": 1, "timestamp": "10:00"},
            details={"quality": "good"},
        )

        # MeasurementSet level
        ms = MeasurementSet(
            measurements=[measurement],
            conditions={"series": "timecourse", "duration": "1h"},
            details={"notes": "First series"},
        )

        # Experiment level
        exp = Experiment(
            measurement_sets=[ms],
            conditions={"sample": "S1", "date": "2025-10-18"},
            details={"operator": "John Doe"},
        )

        # ExperimentSet level
        exp_set = ExperimentSet(
            experiments=[exp],
            conditions={"project": "QuantIQ", "year": 2025},
            details={"pi": "Dr. Smith", "funding": "NSF"},
        )

        # Verify metadata exists at each level
        assert exp_set.conditions["project"] == "QuantIQ"
        assert exp_set.details["pi"] == "Dr. Smith"

        assert exp.conditions["sample"] == "S1"
        assert exp.details["operator"] == "John Doe"

        assert ms.conditions["series"] == "timecourse"
        assert ms.details["notes"] == "First series"

        assert measurement.conditions["replicate"] == 1
        assert measurement.details["quality"] == "good"

        assert ds.conditions["wavelength"] == 550
        assert ds.details["units"] == "nm"

        # Verify hierarchy navigation
        retrieved_ds = exp_set[0][0][0][0]
        assert retrieved_ds is ds
