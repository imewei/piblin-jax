"""
Mocked tests for CSV reader error handling and edge cases.

This module tests error conditions and edge cases in the CSV reader
using mocked file operations to ensure isolated, repeatable tests.
"""

from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import numpy as np
import pytest

from quantiq.dataio.readers.csv import GenericCSVReader


class TestCSVReaderMockedErrors:
    """Test CSV reader error conditions with mocked file operations."""

    def test_file_not_found(self):
        """Test error handling when CSV file does not exist."""
        reader = GenericCSVReader()

        # Mock Path.exists() to return False
        with patch.object(Path, "exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="File not found"):
                reader.read("nonexistent.csv")

    def test_empty_file_no_data(self):
        """Test error handling when file contains no data lines."""
        reader = GenericCSVReader()

        # Mock file with only headers (no data)
        csv_content = "# Temperature: 25\n# Pressure: 1.0\n"

        with patch.object(Path, "exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=csv_content)):
                with pytest.raises(ValueError, match="No data found in file"):
                    reader.read("empty.csv")

    def test_empty_file_only_whitespace(self):
        """Test error handling when file contains only whitespace."""
        reader = GenericCSVReader()

        # Mock file with only whitespace and comments
        csv_content = "# Header\n\n  \n\t\n"

        with patch.object(Path, "exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=csv_content)):
                with pytest.raises(ValueError, match="No data found in file"):
                    reader.read("whitespace.csv")

    def test_inconsistent_columns_in_data(self):
        """Test error handling when data has inconsistent number of columns."""
        reader = GenericCSVReader()

        # Mock file with inconsistent columns
        csv_content = "1,2\n3,4\n5,6,7\n"  # Last row has 3 columns instead of 2

        with patch.object(Path, "exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=csv_content)):
                with pytest.raises(ValueError, match="Inconsistent number of columns"):
                    reader.read("inconsistent.csv")

    def test_data_not_2d_single_column(self):
        """Test error handling when data has only one column."""
        reader = GenericCSVReader()

        # Mock file with single column (needs at least 2 for x,y)
        csv_content = "1\n2\n3\n"

        with patch.object(Path, "exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=csv_content)):
                with pytest.raises(ValueError, match="at least 2 columns"):
                    reader.read("single_column.csv")

    def test_invalid_numeric_data(self):
        """Test error handling when data contains invalid numeric values."""
        reader = GenericCSVReader()

        # Mock file with non-numeric data
        csv_content = "1,2\nabc,def\n3,4\n"

        with patch.object(Path, "exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=csv_content)):
                with pytest.raises(ValueError):
                    reader.read("invalid_data.csv")


class TestCSVReaderMockedFileIO:
    """Test CSV reader with mocked file operations for various scenarios."""

    def test_read_with_mocked_basic_csv(self):
        """Test CSV reading with mocked basic file content."""
        reader = GenericCSVReader()

        # Mock CSV content
        csv_content = "# Temperature: 25\n0.0,0.0\n1.0,1.0\n2.0,4.0\n"

        with patch.object(Path, "exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=csv_content)):
                # Mock metadata functions to avoid complex mocking
                with patch(
                    "quantiq.dataio.readers.csv.metadata.parse_header_metadata",
                    return_value={"Temperature": "25"},
                ):
                    with patch(
                        "quantiq.dataio.readers.csv.metadata.extract_from_filename",
                        return_value={},
                    ):
                        with patch(
                            "quantiq.dataio.readers.csv.metadata.merge_metadata",
                            return_value={"Temperature": "25"},
                        ):
                            with patch(
                                "quantiq.dataio.readers.csv.metadata.separate_conditions_details",
                                return_value=({"Temperature": "25"}, {}),
                            ):
                                measurement = reader.read("test.csv")

                                # Verify basic structure
                                assert measurement is not None
                                assert len(measurement.datasets) > 0

                                # Verify data was parsed correctly
                                dataset = measurement.datasets[0]
                                np.testing.assert_array_equal(
                                    dataset.independent_variable_data, np.array([0.0, 1.0, 2.0])
                                )
                                np.testing.assert_array_equal(
                                    dataset.dependent_variable_data, np.array([0.0, 1.0, 4.0])
                                )

    def test_read_with_different_delimiters(self):
        """Test CSV reading with different delimiters using mocked files."""
        # Test tab-delimited
        reader_tab = GenericCSVReader(delimiter="\t")
        csv_content_tab = "0.0\t0.0\n1.0\t1.0\n"

        with patch.object(Path, "exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=csv_content_tab)):
                with patch(
                    "quantiq.dataio.readers.csv.metadata.parse_header_metadata",
                    return_value={},
                ):
                    with patch(
                        "quantiq.dataio.readers.csv.metadata.extract_from_filename",
                        return_value={},
                    ):
                        with patch(
                            "quantiq.dataio.readers.csv.metadata.merge_metadata",
                            return_value={},
                        ):
                            with patch(
                                "quantiq.dataio.readers.csv.metadata.separate_conditions_details",
                                return_value=({}, {}),
                            ):
                                measurement = reader_tab.read("test.tsv")
                                assert len(measurement.datasets) > 0

        # Test semicolon-delimited
        reader_semi = GenericCSVReader(delimiter=";")
        csv_content_semi = "0.0;0.0\n1.0;1.0\n"

        with patch.object(Path, "exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=csv_content_semi)):
                with patch(
                    "quantiq.dataio.readers.csv.metadata.parse_header_metadata",
                    return_value={},
                ):
                    with patch(
                        "quantiq.dataio.readers.csv.metadata.extract_from_filename",
                        return_value={},
                    ):
                        with patch(
                            "quantiq.dataio.readers.csv.metadata.merge_metadata",
                            return_value={},
                        ):
                            with patch(
                                "quantiq.dataio.readers.csv.metadata.separate_conditions_details",
                                return_value=({}, {}),
                            ):
                                measurement = reader_semi.read("test.csv")
                                assert len(measurement.datasets) > 0

    def test_read_whitespace_delimited(self):
        """Test reading whitespace-delimited data (delimiter=None)."""
        reader = GenericCSVReader(delimiter=None)

        # Mock whitespace-delimited content
        csv_content = "0.0  0.0\n1.0    1.0\n2.0 4.0\n"

        with patch.object(Path, "exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=csv_content)):
                with patch(
                    "quantiq.dataio.readers.csv.metadata.parse_header_metadata",
                    return_value={},
                ):
                    with patch(
                        "quantiq.dataio.readers.csv.metadata.extract_from_filename",
                        return_value={},
                    ):
                        with patch(
                            "quantiq.dataio.readers.csv.metadata.merge_metadata",
                            return_value={},
                        ):
                            with patch(
                                "quantiq.dataio.readers.csv.metadata.separate_conditions_details",
                                return_value=({}, {}),
                            ):
                                measurement = reader.read("test.txt")

                                # Verify data was parsed correctly
                                dataset = measurement.datasets[0]
                                np.testing.assert_array_equal(
                                    dataset.independent_variable_data, np.array([0.0, 1.0, 2.0])
                                )

    def test_multicolumn_data_creation(self):
        """Test that multi-column data creates multiple datasets."""
        reader = GenericCSVReader()

        # Mock multi-column CSV (x, y1, y2, y3)
        csv_content = "0.0,0.0,0.0,0.0\n1.0,1.0,2.0,3.0\n"

        with patch.object(Path, "exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=csv_content)):
                with patch(
                    "quantiq.dataio.readers.csv.metadata.parse_header_metadata",
                    return_value={},
                ):
                    with patch(
                        "quantiq.dataio.readers.csv.metadata.extract_from_filename",
                        return_value={},
                    ):
                        with patch(
                            "quantiq.dataio.readers.csv.metadata.merge_metadata",
                            return_value={},
                        ):
                            with patch(
                                "quantiq.dataio.readers.csv.metadata.separate_conditions_details",
                                return_value=({}, {}),
                            ):
                                measurement = reader.read("multi.csv")

                                # Should create 3 datasets (one per y-column)
                                assert len(measurement.datasets) == 3

                                # All should share the same x values
                                for dataset in measurement.datasets:
                                    np.testing.assert_array_equal(
                                        dataset.independent_variable_data, np.array([0.0, 1.0])
                                    )


class TestCSVReaderEdgeCases:
    """Test CSV reader edge cases with mocked operations."""

    def test_empty_values_stripped(self):
        """Test that empty values are properly stripped/ignored."""
        reader = GenericCSVReader()

        # Mock CSV with trailing commas (empty values)
        csv_content = "1.0,2.0,\n3.0,4.0,\n"

        with patch.object(Path, "exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=csv_content)):
                with patch(
                    "quantiq.dataio.readers.csv.metadata.parse_header_metadata",
                    return_value={},
                ):
                    with patch(
                        "quantiq.dataio.readers.csv.metadata.extract_from_filename",
                        return_value={},
                    ):
                        with patch(
                            "quantiq.dataio.readers.csv.metadata.merge_metadata",
                            return_value={},
                        ):
                            with patch(
                                "quantiq.dataio.readers.csv.metadata.separate_conditions_details",
                                return_value=({}, {}),
                            ):
                                measurement = reader.read("trailing.csv")

                                # Should only parse non-empty values
                                assert measurement is not None
                                assert len(measurement.datasets) > 0

    def test_mixed_comment_and_data_lines(self):
        """Test proper separation of comment and data lines."""
        reader = GenericCSVReader(comment_char="#")

        # Mock CSV with interleaved comments (should only extract headers)
        csv_content = "# Header1\n1.0,2.0\n# Not a header (should be ignored)\n3.0,4.0\n"

        with patch.object(Path, "exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=csv_content)):
                with patch(
                    "quantiq.dataio.readers.csv.metadata.parse_header_metadata"
                ) as mock_parse:
                    mock_parse.return_value = {}
                    with (
                        patch(
                            "quantiq.dataio.readers.csv.metadata.extract_from_filename",
                            return_value={},
                        ),
                        patch(
                            "quantiq.dataio.readers.csv.metadata.merge_metadata",
                            return_value={},
                        ),
                        patch(
                            "quantiq.dataio.readers.csv.metadata.separate_conditions_details",
                            return_value=({}, {}),
                        ),
                    ):
                        measurement = reader.read("mixed.csv")

                        # Verify parse_header_metadata was called with header lines
                        # (including the interleaved comment)
                        assert mock_parse.called
                        header_lines_arg = mock_parse.call_args[0][0]
                        assert len(header_lines_arg) == 2  # Both comment lines

    def test_different_comment_characters(self):
        """Test CSV reader with different comment characters."""
        reader = GenericCSVReader(comment_char="%")

        # Mock CSV with % as comment character
        csv_content = "% Temperature: 25\n1.0,2.0\n3.0,4.0\n"

        with patch.object(Path, "exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=csv_content)):
                with patch(
                    "quantiq.dataio.readers.csv.metadata.parse_header_metadata"
                ) as mock_parse:
                    mock_parse.return_value = {}
                    with (
                        patch(
                            "quantiq.dataio.readers.csv.metadata.extract_from_filename",
                            return_value={},
                        ),
                        patch(
                            "quantiq.dataio.readers.csv.metadata.merge_metadata",
                            return_value={},
                        ),
                        patch(
                            "quantiq.dataio.readers.csv.metadata.separate_conditions_details",
                            return_value=({}, {}),
                        ),
                    ):
                        measurement = reader.read("percent.csv")

                        # Verify header was detected with % character
                        assert mock_parse.called
                        _, kwargs = mock_parse.call_args
                        assert kwargs["comment_char"] == "%"
