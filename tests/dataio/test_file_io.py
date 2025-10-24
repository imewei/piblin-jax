"""Tests for file I/O system including readers, auto-detection, and hierarchy building."""

from pathlib import Path

import numpy as np
import pytest

from piblin_jax.data.collections import ExperimentSet, Measurement
from piblin_jax.dataio import read_directory, read_files
from piblin_jax.dataio.readers import GenericCSVReader, GenericTXTReader, detect_reader, read_file

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "test_data"


class TestGenericCSVReader:
    """Test GenericCSVReader functionality."""

    def test_csv_reader_basic(self):
        """Test basic CSV reading with headers and metadata."""
        reader = GenericCSVReader()
        measurement = reader.read(TEST_DATA_DIR / "sample1.csv")

        # Check that we got a Measurement object
        assert isinstance(measurement, Measurement)

        # Check metadata extraction
        assert "Temperature" in measurement.conditions or "Temperature" in measurement.details

        # Check that we have datasets
        assert len(measurement.datasets) > 0

        # Check data was parsed correctly
        dataset = measurement.datasets[0]
        assert len(dataset.independent_variable_data) == 5
        assert len(dataset.dependent_variable_data) == 5

        # Verify data values
        np.testing.assert_array_equal(
            dataset.independent_variable_data, np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        )
        np.testing.assert_array_equal(
            dataset.dependent_variable_data, np.array([0.0, 1.0, 4.0, 9.0, 16.0])
        )

    def test_csv_reader_multicolumn(self):
        """Test CSV reading with multiple data columns."""
        reader = GenericCSVReader()
        measurement = reader.read(TEST_DATA_DIR / "multicol.csv")

        # Should create multiple datasets (one per y-column)
        assert len(measurement.datasets) == 3

        # All should share the same independent variable
        x_expected = np.array([0.0, 1.0, 2.0, 3.0])
        for dataset in measurement.datasets:
            np.testing.assert_array_equal(dataset.independent_variable_data, x_expected)

    def test_csv_reader_different_delimiter(self):
        """Test CSV reader with different delimiters."""
        # Tab-delimited
        reader_tab = GenericCSVReader(delimiter="\t")
        # Semicolon-delimited
        reader_semi = GenericCSVReader(delimiter=";")

        # Just verify they initialize correctly
        assert reader_tab.delimiter == "\t"
        assert reader_semi.delimiter == ";"


class TestGenericTXTReader:
    """Test GenericTXTReader functionality."""

    def test_txt_reader_basic(self):
        """Test basic TXT reading (whitespace-delimited)."""
        reader = GenericTXTReader()
        measurement = reader.read(TEST_DATA_DIR / "sample3.txt")

        # Check that we got a Measurement object
        assert isinstance(measurement, Measurement)

        # Check we have datasets
        assert len(measurement.datasets) > 0

        # Check data was parsed correctly
        dataset = measurement.datasets[0]
        assert len(dataset.independent_variable_data) == 4

        # Verify data values
        np.testing.assert_array_equal(
            dataset.independent_variable_data, np.array([0.0, 1.0, 2.0, 3.0])
        )
        np.testing.assert_array_equal(
            dataset.dependent_variable_data, np.array([0.0, 3.0, 6.0, 9.0])
        )


class TestAutoDetection:
    """Test file type auto-detection."""

    def test_detect_csv_by_extension(self):
        """Test CSV detection by file extension."""
        reader = detect_reader(TEST_DATA_DIR / "sample1.csv")
        assert isinstance(reader, GenericCSVReader)

    def test_detect_txt_by_extension(self):
        """Test TXT detection by file extension."""
        reader = detect_reader(TEST_DATA_DIR / "sample3.txt")
        assert isinstance(reader, GenericTXTReader)

    def test_read_file_auto_detection(self):
        """Test read_file with auto-detection."""
        # Read CSV file
        measurement_csv = read_file(TEST_DATA_DIR / "sample1.csv")
        assert isinstance(measurement_csv, Measurement)
        assert len(measurement_csv.datasets) > 0

        # Read TXT file
        measurement_txt = read_file(TEST_DATA_DIR / "sample3.txt")
        assert isinstance(measurement_txt, Measurement)
        assert len(measurement_txt.datasets) > 0


class TestMetadataExtraction:
    """Test metadata extraction from file headers."""

    def test_header_metadata_parsing(self):
        """Test that header metadata is correctly extracted."""
        reader = GenericCSVReader()
        measurement = reader.read(TEST_DATA_DIR / "sample1.csv")

        # Check that metadata was extracted
        all_metadata = {**measurement.conditions, **measurement.details}

        # Should have Temperature, Pressure, Sample from headers
        assert "Temperature" in all_metadata
        assert "Pressure" in all_metadata
        assert "Sample" in all_metadata

        # Values should be strings (as parsed)
        assert all_metadata["Temperature"] == "25"
        assert all_metadata["Pressure"] == "1.0"
        assert all_metadata["Sample"] == "A1"


class TestHierarchyBuilding:
    """Test hierarchy building from file lists."""

    def test_read_multiple_files(self):
        """Test reading multiple files and building hierarchy."""
        files = [
            TEST_DATA_DIR / "sample1.csv",
            TEST_DATA_DIR / "sample2.csv",
        ]

        experiment_set = read_files(files)

        # Should get an ExperimentSet
        assert isinstance(experiment_set, ExperimentSet)

        # Should have at least one experiment
        assert len(experiment_set.experiments) > 0

        # Should have measurements
        total_measurements = sum(
            len(ms.measurements)
            for exp in experiment_set.experiments
            for ms in exp.measurement_sets
        )
        assert total_measurements == 2

    def test_read_directory(self):
        """Test reading entire directory."""
        experiment_set = read_directory(TEST_DATA_DIR, pattern="*.csv")

        # Should get an ExperimentSet
        assert isinstance(experiment_set, ExperimentSet)

        # Should have found all CSV files
        total_measurements = sum(
            len(ms.measurements)
            for exp in experiment_set.experiments
            for ms in exp.measurement_sets
        )
        assert total_measurements >= 3  # At least sample1, sample2, multicol
