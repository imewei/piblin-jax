"""
TabularMeasurementSet class for piblin-jax.

MeasurementSet variant with tabular access patterns (rows and columns).
"""

from typing import Any

from .measurement import Measurement
from .measurement_set import MeasurementSet


class TabularMeasurementSet(MeasurementSet):
    """
    MeasurementSet with measurements arranged in tabular format.

    This specialized variant organizes measurements in a logical table structure
    with row and column labels. This is useful for:
    - Experimental design matrices (e.g., multi-factor designs)
    - Microplate/well plate layouts
    - Spatial arrangements of measurements
    - Grid-based sampling patterns

    The tabular structure enables intuitive access patterns and
    natural visualization as tables or heatmaps.

    Parameters
    ----------
    measurements : list[Measurement]
        List of Measurement objects. The order corresponds to
        row-major ordering in the table (row1-col1, row1-col2, ..., row2-col1, ...).
    row_labels : list[str] | None, optional
        Labels for table rows. If provided, must satisfy:
        len(row_labels) * len(col_labels) == len(measurements)
    col_labels : list[str] | None, optional
        Labels for table columns. If provided, must satisfy:
        len(row_labels) * len(col_labels) == len(measurements)
    conditions : dict[str, Any] | None, optional
        Experimental conditions for the measurement series.
    details : dict[str, Any] | None, optional
        Additional context for the measurement series.

    Attributes
    ----------
    row_labels : list[str] | None
        Labels for table rows.
    col_labels : list[str] | None
        Labels for table columns.

    Notes
    -----
    Measurements are stored in row-major order. For a 2x3 table:
    - measurements[0] = row 0, col 0
    - measurements[1] = row 0, col 1
    - measurements[2] = row 0, col 2
    - measurements[3] = row 1, col 0
    - measurements[4] = row 1, col 1
    - measurements[5] = row 1, col 2

    Examples
    --------
    >>> import numpy as np
    >>> from piblin_jax.data.datasets import OneDimensionalDataset
    >>> from piblin_jax.data.collections import Measurement, TabularMeasurementSet
    >>>
    >>> # Create a 2x3 grid of measurements
    >>> x = np.linspace(0, 10, 50)
    >>> measurements = []
    >>>
    >>> for i in range(2):  # rows
    ...     for j in range(3):  # columns
    ...         y = np.sin(x * (i + 1)) * (j + 1)
    ...         ds = OneDimensionalDataset(x, y)
    ...         m = Measurement(
    ...             [ds],
    ...             conditions={"row": i, "col": j}
    ...         )
    ...         measurements.append(m)
    >>>
    >>> # Create tabular measurement set
    >>> tms = TabularMeasurementSet(
    ...     measurements=measurements,
    ...     row_labels=["row_A", "row_B"],
    ...     col_labels=["col_1", "col_2", "col_3"],
    ...     conditions={"plate": "plate_001"},
    ...     details={"date": "2025-10-18"}
    ... )
    >>>
    >>> len(tms)
    6
    >>> tms.row_labels
    ['row_A', 'row_B']
    >>> tms.col_labels
    ['col_1', 'col_2', 'col_3']
    >>>
    >>> # Access measurement at row 1, col 2
    >>> m = tms.get_measurement(1, 2)
    >>> m.conditions["row"]
    1
    >>> m.conditions["col"]
    2
    """

    def __init__(
        self,
        measurements: list[Measurement],
        row_labels: list[str] | None = None,
        col_labels: list[str] | None = None,
        conditions: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
    ):
        """
        Initialize TabularMeasurementSet with optional row/column labels.

        Parameters
        ----------
        measurements : list[Measurement]
            List of Measurement objects in row-major order.
        row_labels : list[str] | None, optional
            Labels for table rows.
        col_labels : list[str] | None, optional
            Labels for table columns.
        conditions : dict[str, Any] | None, optional
            Experimental conditions for this measurement series.
        details : dict[str, Any] | None, optional
            Additional context for this measurement series.

        Raises
        ------
        ValueError
            If row_labels and col_labels are provided but their product
            doesn't match the number of measurements.
        """
        # Validate dimensions if labels are provided
        if row_labels is not None and col_labels is not None:
            expected_count = len(row_labels) * len(col_labels)
            if len(measurements) != expected_count:
                raise ValueError(
                    f"Number of measurements ({len(measurements)}) must equal "
                    f"len(row_labels) * len(col_labels) ({expected_count}). "
                    f"Got {len(row_labels)} rows and {len(col_labels)} columns."
                )

        self._row_labels = row_labels
        self._col_labels = col_labels

        # Call parent constructor
        super().__init__(measurements, conditions, details)

    @property
    def row_labels(self) -> list[str] | None:
        """
        Get row labels for the table.

        Returns
        -------
        list[str] | None
            List of row labels, or None if not provided.

        Examples
        --------
        >>> tms.row_labels
        ['row_A', 'row_B', 'row_C']
        """
        return self._row_labels

    @property
    def col_labels(self) -> list[str] | None:
        """
        Get column labels for the table.

        Returns
        -------
        list[str] | None
            List of column labels, or None if not provided.

        Examples
        --------
        >>> tms.col_labels
        ['col_1', 'col_2', 'col_3', 'col_4']
        """
        return self._col_labels

    @property
    def shape(self) -> tuple[int, int] | None:
        """
        Get the shape of the table (rows, columns).

        Returns
        -------
        tuple[int, int] | None
            (n_rows, n_cols) if labels are provided, None otherwise.

        Examples
        --------
        >>> tms.shape
        (2, 3)
        """
        if self._row_labels is not None and self._col_labels is not None:
            return (len(self._row_labels), len(self._col_labels))
        return None

    def get_measurement(self, row: int, col: int) -> Measurement:
        """
        Get measurement at specified row and column indices.

        Uses row-major ordering: index = row * n_cols + col

        Parameters
        ----------
        row : int
            Row index (0-based).
        col : int
            Column index (0-based).

        Returns
        -------
        Measurement
            Measurement at the specified position.

        Raises
        ------
        ValueError
            If row_labels and col_labels were not provided.
        IndexError
            If row or col indices are out of bounds.

        Examples
        --------
        >>> m = tms.get_measurement(1, 2)
        >>> m.conditions["row"]
        1
        >>> m.conditions["col"]
        2
        """
        if self._row_labels is None or self._col_labels is None:
            raise ValueError(
                "get_measurement() requires row_labels and col_labels. "
                "Use direct indexing instead: tms[index]"
            )

        n_rows = len(self._row_labels)
        n_cols = len(self._col_labels)

        if not (0 <= row < n_rows):
            raise IndexError(f"Row index {row} out of bounds [0, {n_rows})")
        if not (0 <= col < n_cols):
            raise IndexError(f"Column index {col} out of bounds [0, {n_cols})")

        index = row * n_cols + col
        return self._measurements[index]

    def get_row(self, row: int) -> list[Measurement]:
        """
        Get all measurements in a specified row.

        Parameters
        ----------
        row : int
            Row index (0-based).

        Returns
        -------
        list[Measurement]
            List of measurements in the row.

        Raises
        ------
        ValueError
            If row_labels and col_labels were not provided.
        IndexError
            If row index is out of bounds.

        Examples
        --------
        >>> row_measurements = tms.get_row(0)
        >>> len(row_measurements)
        3
        >>> [m.conditions["col"] for m in row_measurements]
        [0, 1, 2]
        """
        if self._row_labels is None or self._col_labels is None:
            raise ValueError("get_row() requires row_labels and col_labels.")

        n_rows = len(self._row_labels)
        n_cols = len(self._col_labels)

        if not (0 <= row < n_rows):
            raise IndexError(f"Row index {row} out of bounds [0, {n_rows})")

        start_idx = row * n_cols
        end_idx = start_idx + n_cols
        return list(self._measurements[start_idx:end_idx])

    def get_column(self, col: int) -> list[Measurement]:
        """
        Get all measurements in a specified column.

        Parameters
        ----------
        col : int
            Column index (0-based).

        Returns
        -------
        list[Measurement]
            List of measurements in the column.

        Raises
        ------
        ValueError
            If row_labels and col_labels were not provided.
        IndexError
            If col index is out of bounds.

        Examples
        --------
        >>> col_measurements = tms.get_column(1)
        >>> len(col_measurements)
        2
        >>> [m.conditions["row"] for m in col_measurements]
        [0, 1]
        """
        if self._row_labels is None or self._col_labels is None:
            raise ValueError("get_column() requires row_labels and col_labels.")

        n_rows = len(self._row_labels)
        n_cols = len(self._col_labels)

        if not (0 <= col < n_cols):
            raise IndexError(f"Column index {col} out of bounds [0, {n_cols})")

        return [self._measurements[row * n_cols + col] for row in range(n_rows)]
