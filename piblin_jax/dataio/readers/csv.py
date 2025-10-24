"""Generic CSV file reader with metadata extraction.

This module provides a flexible CSV reader that can handle various delimiters,
extract metadata from file headers, and create appropriate Dataset objects.
"""

from pathlib import Path
from typing import Any

import numpy as np

from piblin_jax.data import metadata
from piblin_jax.data.collections import Measurement
from piblin_jax.data.datasets import OneDimensionalDataset


class GenericCSVReader:
    """Generic CSV file reader with metadata extraction.

    This reader handles CSV files with optional header comments containing metadata.
    It supports various delimiters and automatically creates Dataset objects from
    the parsed data.

    Parameters
    ----------
    delimiter : str, optional
        Column delimiter character (default: ",").
        Common values: "," (CSV), "\\t" (TSV), ";" (European CSV)
    comment_char : str, optional
        Comment character for header lines (default: "#")

    Examples
    --------
    Read a standard CSV file:

    >>> reader = GenericCSVReader()
    >>> measurement = reader.read("data.csv")

    Read a tab-delimited file:

    >>> reader = GenericCSVReader(delimiter="\\t")
    >>> measurement = reader.read("data.tsv")

    File format example::

        # Temperature: 25
        # Pressure: 1.0
        # Sample: A1
        0.0,0.0
        1.0,1.0
        2.0,4.0
    """

    def __init__(self, delimiter: str | None = ",", comment_char: str = "#"):
        """Initialize GenericCSVReader.

        See class docstring for parameter details.
        """
        self.delimiter = delimiter
        self.comment_char = comment_char

    def read(self, filepath: str | Path) -> Measurement:
        """Read CSV file and return Measurement object.

        Parses the CSV file, extracting metadata from headers and creating
        appropriate Dataset objects from the data columns.

        Parameters
        ----------
        filepath : str | Path
            Path to CSV file

        Returns
        -------
        Measurement
            Measurement object containing datasets and metadata

        Raises
        ------
        FileNotFoundError
            If the file does not exist
        ValueError
            If the file format is invalid or cannot be parsed
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Read file and separate headers from data
        with open(filepath) as f:
            lines = f.readlines()

        header_lines = []
        data_lines = []
        for line in lines:
            if line.strip().startswith(self.comment_char):
                header_lines.append(line)
            elif line.strip():  # Non-empty, non-comment
                data_lines.append(line)

        if not data_lines:
            raise ValueError(f"No data found in file: {filepath}")

        # Extract metadata from headers
        file_metadata = metadata.parse_header_metadata(header_lines, comment_char=self.comment_char)

        # Extract from filename
        filename_metadata = metadata.extract_from_filename(filepath)

        # Merge metadata (filename takes priority)
        combined_metadata = metadata.merge_metadata(
            [file_metadata, filename_metadata], strategy="override"
        )

        # Separate conditions and details
        conditions, details = metadata.separate_conditions_details(combined_metadata)

        # Parse data
        data_array = self._parse_data_lines(data_lines)

        # Create datasets based on number of columns
        datasets = self._create_datasets(data_array, conditions, details)

        # Create measurement
        measurement = Measurement(datasets=datasets, conditions=conditions, details=details)  # type: ignore[arg-type]

        return measurement

    def _parse_data_lines(self, data_lines: list[str]) -> np.ndarray:
        """Parse data lines into numpy array.

        Parameters
        ----------
        data_lines : list[str]
            Lines containing data values

        Returns
        -------
        np.ndarray
            2D array of data values (rows x columns)

        Raises
        ------
        ValueError
            If data cannot be parsed or has inconsistent columns
        """
        data_list: list[list[float]] = []
        for line in data_lines:
            if self.delimiter is None:
                # Whitespace-delimited
                values = [float(v.strip()) for v in line.split() if v.strip()]
            else:
                # Specific delimiter
                values = [float(v.strip()) for v in line.split(self.delimiter) if v.strip()]
            data_list.append(values)

        # Convert to numpy array and validate
        try:
            data_array = np.array(data_list)
        except ValueError as e:
            raise ValueError(f"Inconsistent number of columns in data: {e}") from e

        if data_array.size == 0:
            raise ValueError("No valid data found in file")

        if data_array.ndim != 2:
            raise ValueError("Data must be 2-dimensional (rows x columns)")

        return data_array

    def _create_datasets(
        self, data_array: np.ndarray, conditions: dict[str, Any], details: dict[str, Any]
    ) -> list[OneDimensionalDataset]:
        """Create Dataset objects from data array.

        Parameters
        ----------
        data_array : np.ndarray
            2D array of data (rows x columns)
        conditions : dict
            Experimental conditions
        details : dict
            Contextual details

        Returns
        -------
        list[OneDimensionalDataset]
            List of datasets created from the data

        Notes
        -----
        - If data has 2 columns: creates single 1D dataset (x, y)
        - If data has >2 columns: creates multiple 1D datasets sharing the
          first column as independent variable
        """
        datasets = []

        if data_array.shape[1] < 2:
            raise ValueError(f"Data must have at least 2 columns (x, y), got {data_array.shape[1]}")

        if data_array.shape[1] == 2:
            # Single 1D dataset (x, y)
            ds = OneDimensionalDataset(
                independent_variable_data=data_array[:, 0],
                dependent_variable_data=data_array[:, 1],
                conditions=conditions,
                details=details,
            )
            datasets.append(ds)
        else:
            # Multiple 1D datasets sharing independent variable
            x = data_array[:, 0]
            for col_idx in range(1, data_array.shape[1]):
                y = data_array[:, col_idx]
                ds = OneDimensionalDataset(
                    independent_variable_data=x,
                    dependent_variable_data=y,
                    conditions=conditions,
                    details=details,
                )
                datasets.append(ds)

        return datasets
