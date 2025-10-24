"""Generic TXT file reader for whitespace-delimited data files.

This module provides a TXT reader that extends the CSV reader for
whitespace-delimited files, commonly used in scientific data.
"""

from .csv import GenericCSVReader


class GenericTXTReader(GenericCSVReader):
    """Generic TXT file reader (whitespace-delimited).

    This reader handles text files with whitespace-delimited columns (spaces or tabs).
    It inherits from GenericCSVReader but uses whitespace splitting instead of
    a specific delimiter.

    Parameters
    ----------
    comment_char : str, optional
        Comment character for header lines (default: "#")

    Examples
    --------
    Read a whitespace-delimited file:

    >>> reader = GenericTXTReader()
    >>> measurement = reader.read("data.txt")

    File format example::

        # Temperature: 25
        # Sample: A1
        0.0 0.0
        1.0 1.0
        2.0 4.0

    Notes
    -----
    This reader is suitable for files where columns are separated by any amount
    of whitespace (spaces, tabs, or combinations). It automatically handles
    varying amounts of whitespace between columns.
    """

    def __init__(self, comment_char: str = "#"):
        """Initialize GenericTXTReader.

        See class docstring for parameter details.
        """
        # Use None as delimiter to trigger whitespace splitting
        super().__init__(delimiter=None, comment_char=comment_char)
