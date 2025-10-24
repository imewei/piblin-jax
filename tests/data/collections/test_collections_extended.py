"""
Extended tests for collection classes to achieve comprehensive coverage.

This test file focuses on covering uncovered lines identified:
- TabularMeasurementSet: Lines 138, 199-201, 236-251, 282-293, 324-333
- TidyMeasurementSet: Lines 158-160, 199-209
- ExperimentSet: Lines 277-287
"""

import numpy as np
import pytest

from piblin_jax.data.collections import (
    Experiment,
    ExperimentSet,
    Measurement,
    MeasurementSet,
    TabularMeasurementSet,
    TidyMeasurementSet,
)
from piblin_jax.data.datasets import OneDimensionalDataset, ZeroDimensionalDataset


class TestTabularMeasurementSetExtended:
    """Extended tests for TabularMeasurementSet."""

    def test_tabular_measurement_set_dimension_validation(self):
        """Test validation when row/col labels don't match measurement count."""
        measurements = [
            Measurement(datasets=[OneDimensionalDataset(np.array([i]), np.array([i * 2]))])
            for i in range(4)
        ]

        row_labels = ["row1", "row2"]
        col_labels = ["col1", "col2", "col3"]  # 2 * 3 = 6, but only 4 measurements

        # Should raise ValueError
        with pytest.raises(ValueError, match="Number of measurements"):
            TabularMeasurementSet(
                measurements=measurements, row_labels=row_labels, col_labels=col_labels
            )

    def test_tabular_shape_property(self):
        """Test shape property returns correct dimensions."""
        measurements = [
            Measurement(datasets=[OneDimensionalDataset(np.array([i]), np.array([i * 2]))])
            for i in range(6)
        ]

        row_labels = ["row1", "row2"]
        col_labels = ["col1", "col2", "col3"]

        tms = TabularMeasurementSet(
            measurements=measurements, row_labels=row_labels, col_labels=col_labels
        )

        assert tms.shape == (2, 3)

    def test_tabular_shape_property_without_labels(self):
        """Test shape property returns None when labels not provided."""
        measurements = [
            Measurement(datasets=[OneDimensionalDataset(np.array([i]), np.array([i * 2]))])
            for i in range(6)
        ]

        tms = TabularMeasurementSet(measurements=measurements)

        assert tms.shape is None

    def test_get_measurement_by_indices(self):
        """Test getting measurement by row and column indices."""
        measurements = []
        for i in range(2):
            for j in range(3):
                m = Measurement(
                    datasets=[OneDimensionalDataset(np.array([1]), np.array([2]))],
                    conditions={"row": i, "col": j},
                )
                measurements.append(m)

        row_labels = ["row_A", "row_B"]
        col_labels = ["col_1", "col_2", "col_3"]

        tms = TabularMeasurementSet(
            measurements=measurements, row_labels=row_labels, col_labels=col_labels
        )

        # Test various positions
        m_0_0 = tms.get_measurement(0, 0)
        assert m_0_0.conditions["row"] == 0
        assert m_0_0.conditions["col"] == 0

        m_1_2 = tms.get_measurement(1, 2)
        assert m_1_2.conditions["row"] == 1
        assert m_1_2.conditions["col"] == 2

        m_0_2 = tms.get_measurement(0, 2)
        assert m_0_2.conditions["row"] == 0
        assert m_0_2.conditions["col"] == 2

    def test_get_measurement_without_labels_raises_error(self):
        """Test get_measurement raises error when labels not provided."""
        measurements = [
            Measurement(datasets=[OneDimensionalDataset(np.array([i]), np.array([i * 2]))])
            for i in range(6)
        ]

        tms = TabularMeasurementSet(measurements=measurements)

        # Should raise ValueError
        with pytest.raises(ValueError, match="requires row_labels and col_labels"):
            tms.get_measurement(0, 0)

    def test_get_measurement_row_index_out_of_bounds(self):
        """Test get_measurement raises IndexError for invalid row index."""
        measurements = [
            Measurement(datasets=[OneDimensionalDataset(np.array([i]), np.array([i * 2]))])
            for i in range(6)
        ]

        row_labels = ["row1", "row2"]
        col_labels = ["col1", "col2", "col3"]

        tms = TabularMeasurementSet(
            measurements=measurements, row_labels=row_labels, col_labels=col_labels
        )

        # Row index 2 is out of bounds (only 0, 1 valid)
        with pytest.raises(IndexError, match="Row index .* out of bounds"):
            tms.get_measurement(2, 0)

        # Negative indices also invalid
        with pytest.raises(IndexError, match="Row index .* out of bounds"):
            tms.get_measurement(-1, 0)

    def test_get_measurement_col_index_out_of_bounds(self):
        """Test get_measurement raises IndexError for invalid col index."""
        measurements = [
            Measurement(datasets=[OneDimensionalDataset(np.array([i]), np.array([i * 2]))])
            for i in range(6)
        ]

        row_labels = ["row1", "row2"]
        col_labels = ["col1", "col2", "col3"]

        tms = TabularMeasurementSet(
            measurements=measurements, row_labels=row_labels, col_labels=col_labels
        )

        # Col index 3 is out of bounds (only 0, 1, 2 valid)
        with pytest.raises(IndexError, match="Column index .* out of bounds"):
            tms.get_measurement(0, 3)

        # Negative indices also invalid
        with pytest.raises(IndexError, match="Column index .* out of bounds"):
            tms.get_measurement(0, -1)

    def test_get_row(self):
        """Test getting all measurements in a row."""
        measurements = []
        for i in range(2):
            for j in range(3):
                m = Measurement(
                    datasets=[OneDimensionalDataset(np.array([1]), np.array([2]))],
                    conditions={"row": i, "col": j},
                )
                measurements.append(m)

        row_labels = ["row_A", "row_B"]
        col_labels = ["col_1", "col_2", "col_3"]

        tms = TabularMeasurementSet(
            measurements=measurements, row_labels=row_labels, col_labels=col_labels
        )

        # Get first row
        row_0 = tms.get_row(0)
        assert len(row_0) == 3
        assert all(m.conditions["row"] == 0 for m in row_0)
        assert [m.conditions["col"] for m in row_0] == [0, 1, 2]

        # Get second row
        row_1 = tms.get_row(1)
        assert len(row_1) == 3
        assert all(m.conditions["row"] == 1 for m in row_1)
        assert [m.conditions["col"] for m in row_1] == [0, 1, 2]

    def test_get_row_without_labels_raises_error(self):
        """Test get_row raises error when labels not provided."""
        measurements = [
            Measurement(datasets=[OneDimensionalDataset(np.array([i]), np.array([i * 2]))])
            for i in range(6)
        ]

        tms = TabularMeasurementSet(measurements=measurements)

        # Should raise ValueError
        with pytest.raises(ValueError, match="requires row_labels and col_labels"):
            tms.get_row(0)

    def test_get_row_index_out_of_bounds(self):
        """Test get_row raises IndexError for invalid row index."""
        measurements = [
            Measurement(datasets=[OneDimensionalDataset(np.array([i]), np.array([i * 2]))])
            for i in range(6)
        ]

        row_labels = ["row1", "row2"]
        col_labels = ["col1", "col2", "col3"]

        tms = TabularMeasurementSet(
            measurements=measurements, row_labels=row_labels, col_labels=col_labels
        )

        # Row index 2 is out of bounds
        with pytest.raises(IndexError, match="Row index .* out of bounds"):
            tms.get_row(2)

    def test_get_column(self):
        """Test getting all measurements in a column."""
        measurements = []
        for i in range(2):
            for j in range(3):
                m = Measurement(
                    datasets=[OneDimensionalDataset(np.array([1]), np.array([2]))],
                    conditions={"row": i, "col": j},
                )
                measurements.append(m)

        row_labels = ["row_A", "row_B"]
        col_labels = ["col_1", "col_2", "col_3"]

        tms = TabularMeasurementSet(
            measurements=measurements, row_labels=row_labels, col_labels=col_labels
        )

        # Get first column
        col_0 = tms.get_column(0)
        assert len(col_0) == 2
        assert all(m.conditions["col"] == 0 for m in col_0)
        assert [m.conditions["row"] for m in col_0] == [0, 1]

        # Get third column
        col_2 = tms.get_column(2)
        assert len(col_2) == 2
        assert all(m.conditions["col"] == 2 for m in col_2)
        assert [m.conditions["row"] for m in col_2] == [0, 1]

    def test_get_column_without_labels_raises_error(self):
        """Test get_column raises error when labels not provided."""
        measurements = [
            Measurement(datasets=[OneDimensionalDataset(np.array([i]), np.array([i * 2]))])
            for i in range(6)
        ]

        tms = TabularMeasurementSet(measurements=measurements)

        # Should raise ValueError
        with pytest.raises(ValueError, match="requires row_labels and col_labels"):
            tms.get_column(0)

    def test_get_column_index_out_of_bounds(self):
        """Test get_column raises IndexError for invalid col index."""
        measurements = [
            Measurement(datasets=[OneDimensionalDataset(np.array([i]), np.array([i * 2]))])
            for i in range(6)
        ]

        row_labels = ["row1", "row2"]
        col_labels = ["col1", "col2", "col3"]

        tms = TabularMeasurementSet(
            measurements=measurements, row_labels=row_labels, col_labels=col_labels
        )

        # Col index 3 is out of bounds
        with pytest.raises(IndexError, match="Column index .* out of bounds"):
            tms.get_column(3)


class TestTidyMeasurementSetExtended:
    """Extended tests for TidyMeasurementSet."""

    def test_get_unique_conditions_with_unhashable_values(self):
        """Test get_unique_conditions handles unhashable values by converting to string."""
        # Create measurements with unhashable condition values (lists, dicts)
        m1 = Measurement(
            datasets=[OneDimensionalDataset(np.array([1]), np.array([2]))],
            conditions={"settings": [1, 2, 3], "config": {"key": "value1"}},
        )
        m2 = Measurement(
            datasets=[OneDimensionalDataset(np.array([3]), np.array([4]))],
            conditions={"settings": [4, 5, 6], "config": {"key": "value2"}},
        )

        tms = TidyMeasurementSet(measurements=[m1, m2])

        # Should handle unhashable values by converting to strings
        unique = tms.get_unique_conditions()

        assert "settings" in unique
        assert "config" in unique
        # Values should be stringified
        assert "[1, 2, 3]" in unique["settings"]
        assert "[4, 5, 6]" in unique["settings"]

    def test_get_unique_conditions_empty_set(self):
        """Test get_unique_conditions with empty measurement set."""
        tms = TidyMeasurementSet(measurements=[])

        unique = tms.get_unique_conditions()

        assert unique == {}

    def test_get_unique_conditions_varying_keys(self):
        """Test get_unique_conditions with measurements having different keys."""
        m1 = Measurement(
            datasets=[OneDimensionalDataset(np.array([1]), np.array([2]))],
            conditions={"temp": 25, "pressure": 1.0},
        )
        m2 = Measurement(
            datasets=[OneDimensionalDataset(np.array([3]), np.array([4]))],
            conditions={"temp": 30, "sample": "A"},
        )
        m3 = Measurement(
            datasets=[OneDimensionalDataset(np.array([5]), np.array([6]))],
            conditions={"pressure": 2.0, "sample": "B"},
        )

        tms = TidyMeasurementSet(measurements=[m1, m2, m3])

        unique = tms.get_unique_conditions()

        # Should have all keys that appear in any measurement
        assert set(unique.keys()) == {"temp", "pressure", "sample"}
        assert unique["temp"] == {25, 30}
        assert unique["pressure"] == {1.0, 2.0}
        assert unique["sample"] == {"A", "B"}

    def test_filter_by_conditions_single_condition(self):
        """Test filtering by a single condition."""
        m1 = Measurement(
            datasets=[OneDimensionalDataset(np.array([1]), np.array([2]))],
            conditions={"temperature": 25, "sample": "A"},
        )
        m2 = Measurement(
            datasets=[OneDimensionalDataset(np.array([3]), np.array([4]))],
            conditions={"temperature": 30, "sample": "A"},
        )
        m3 = Measurement(
            datasets=[OneDimensionalDataset(np.array([5]), np.array([6]))],
            conditions={"temperature": 25, "sample": "B"},
        )

        tms = TidyMeasurementSet(measurements=[m1, m2, m3])

        # Filter by temperature
        filtered = tms.filter_by_conditions(temperature=25)

        assert len(filtered) == 2
        assert all(m.conditions["temperature"] == 25 for m in filtered)

    def test_filter_by_conditions_multiple_conditions(self):
        """Test filtering by multiple conditions (AND logic)."""
        m1 = Measurement(
            datasets=[OneDimensionalDataset(np.array([1]), np.array([2]))],
            conditions={"temperature": 25, "sample": "A"},
        )
        m2 = Measurement(
            datasets=[OneDimensionalDataset(np.array([3]), np.array([4]))],
            conditions={"temperature": 30, "sample": "A"},
        )
        m3 = Measurement(
            datasets=[OneDimensionalDataset(np.array([5]), np.array([6]))],
            conditions={"temperature": 25, "sample": "B"},
        )

        tms = TidyMeasurementSet(measurements=[m1, m2, m3])

        # Filter by both temperature and sample
        filtered = tms.filter_by_conditions(temperature=25, sample="A")

        assert len(filtered) == 1
        assert filtered[0].conditions["temperature"] == 25
        assert filtered[0].conditions["sample"] == "A"

    def test_filter_by_conditions_no_matches(self):
        """Test filtering returns empty set when no measurements match."""
        m1 = Measurement(
            datasets=[OneDimensionalDataset(np.array([1]), np.array([2]))],
            conditions={"temperature": 25, "sample": "A"},
        )
        m2 = Measurement(
            datasets=[OneDimensionalDataset(np.array([3]), np.array([4]))],
            conditions={"temperature": 30, "sample": "B"},
        )

        tms = TidyMeasurementSet(measurements=[m1, m2])

        # Filter by non-existent combination
        filtered = tms.filter_by_conditions(temperature=25, sample="B")

        assert len(filtered) == 0

    def test_filter_by_conditions_preserves_metadata(self):
        """Test that filtering preserves the measurement set metadata."""
        m1 = Measurement(
            datasets=[OneDimensionalDataset(np.array([1]), np.array([2]))],
            conditions={"temperature": 25},
        )
        m2 = Measurement(
            datasets=[OneDimensionalDataset(np.array([3]), np.array([4]))],
            conditions={"temperature": 30},
        )

        tms = TidyMeasurementSet(
            measurements=[m1, m2],
            conditions={"experiment": "test"},
            details={"operator": "John"},
        )

        # Filter
        filtered = tms.filter_by_conditions(temperature=25)

        # Metadata should be preserved
        assert filtered.conditions == {"experiment": "test"}
        assert filtered.details == {"operator": "John"}


class TestExperimentSetExtended:
    """Extended tests for ExperimentSet."""

    def test_get_experiment_by_condition_single_condition(self):
        """Test getting experiments by single condition."""
        m1 = Measurement(datasets=[OneDimensionalDataset(np.array([1, 2]), np.array([3, 4]))])
        ms1 = MeasurementSet(measurements=[m1])

        exp1 = Experiment(
            measurement_sets=[ms1], conditions={"sample": "S001", "date": "2025-10-18"}
        )
        exp2 = Experiment(
            measurement_sets=[ms1], conditions={"sample": "S002", "date": "2025-10-18"}
        )
        exp3 = Experiment(
            measurement_sets=[ms1], conditions={"sample": "S001", "date": "2025-10-19"}
        )

        exp_set = ExperimentSet(experiments=[exp1, exp2, exp3])

        # Filter by sample
        filtered = exp_set.get_experiment_by_condition(sample="S001")

        assert len(filtered) == 2
        assert all(exp.conditions["sample"] == "S001" for exp in filtered)

    def test_get_experiment_by_condition_multiple_conditions(self):
        """Test getting experiments by multiple conditions (AND logic)."""
        m1 = Measurement(datasets=[OneDimensionalDataset(np.array([1, 2]), np.array([3, 4]))])
        ms1 = MeasurementSet(measurements=[m1])

        exp1 = Experiment(
            measurement_sets=[ms1], conditions={"sample": "S001", "date": "2025-10-18"}
        )
        exp2 = Experiment(
            measurement_sets=[ms1], conditions={"sample": "S002", "date": "2025-10-18"}
        )
        exp3 = Experiment(
            measurement_sets=[ms1], conditions={"sample": "S001", "date": "2025-10-19"}
        )

        exp_set = ExperimentSet(experiments=[exp1, exp2, exp3])

        # Filter by both sample and date
        filtered = exp_set.get_experiment_by_condition(sample="S001", date="2025-10-18")

        assert len(filtered) == 1
        assert filtered[0].conditions["sample"] == "S001"
        assert filtered[0].conditions["date"] == "2025-10-18"

    def test_get_experiment_by_condition_no_matches(self):
        """Test getting experiments returns empty list when no matches."""
        m1 = Measurement(datasets=[OneDimensionalDataset(np.array([1, 2]), np.array([3, 4]))])
        ms1 = MeasurementSet(measurements=[m1])

        exp1 = Experiment(measurement_sets=[ms1], conditions={"sample": "S001"})
        exp2 = Experiment(measurement_sets=[ms1], conditions={"sample": "S002"})

        exp_set = ExperimentSet(experiments=[exp1, exp2])

        # Filter by non-existent sample
        filtered = exp_set.get_experiment_by_condition(sample="S999")

        assert len(filtered) == 0

    def test_get_experiment_by_condition_missing_key(self):
        """Test filtering when some experiments don't have the condition key."""
        m1 = Measurement(datasets=[OneDimensionalDataset(np.array([1, 2]), np.array([3, 4]))])
        ms1 = MeasurementSet(measurements=[m1])

        exp1 = Experiment(
            measurement_sets=[ms1], conditions={"sample": "S001", "date": "2025-10-18"}
        )
        exp2 = Experiment(measurement_sets=[ms1], conditions={"sample": "S002"})  # No date
        exp3 = Experiment(measurement_sets=[ms1], conditions={"date": "2025-10-18"})  # No sample

        exp_set = ExperimentSet(experiments=[exp1, exp2, exp3])

        # Filter by date - should only match exp1 and exp3
        filtered = exp_set.get_experiment_by_condition(date="2025-10-18")

        assert len(filtered) == 2
        assert exp1 in filtered
        assert exp3 in filtered


class TestThreeDimensionalDatasetExtended:
    """Extended tests for ThreeDimensionalDataset to cover missing line 152."""

    def test_three_dimensional_dimension_mismatch_error_message(self):
        """Test detailed error message for dimension mismatch."""
        from piblin_jax.data.datasets import ThreeDimensionalDataset

        x = np.linspace(0, 5, 10)
        y = np.linspace(0, 5, 20)
        z = np.linspace(0, 5, 15)
        data = np.random.randn(10, 20, 10)  # Wrong z dimension (should be 15)

        # Should raise ValueError with detailed message including expected shape
        with pytest.raises(ValueError, match="Dependent variable dimension mismatch"):
            ThreeDimensionalDataset(
                independent_variable_data_1=x,
                independent_variable_data_2=y,
                independent_variable_data_3=z,
                dependent_variable_data=data,
            )

    def test_three_dimensional_correct_dimensions(self):
        """Test that correct dimensions are accepted."""
        from piblin_jax.data.datasets import ThreeDimensionalDataset

        x = np.linspace(0, 5, 10)
        y = np.linspace(0, 5, 20)
        z = np.linspace(0, 5, 15)
        data = np.random.randn(10, 20, 15)  # Correct shape

        # Should succeed
        dataset = ThreeDimensionalDataset(
            independent_variable_data_1=x,
            independent_variable_data_2=y,
            independent_variable_data_3=z,
            dependent_variable_data=data,
        )

        assert dataset.dependent_variable_data.shape == (10, 20, 15)
