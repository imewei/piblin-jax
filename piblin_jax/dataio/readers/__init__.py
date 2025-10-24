"""File readers and auto-detection system.

This module provides:
- Generic CSV and TXT readers
- Multi-layer auto-detection system
- Extensible reader registry
- read_file function for automatic file reading

The auto-detection system uses four layers:
1. Extension-based (.csv, .txt, etc.)
2. Header-based (instrument signatures)
3. Content-based (parse first lines)
4. Fallback to generic readers
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from piblin_jax.data.collections import Measurement

from .csv import GenericCSVReader
from .txt import GenericTXTReader

# Reader registry mapping file extensions to reader classes or factory functions
_READER_REGISTRY: dict[str, type | Callable[[], Any]] = {
    ".csv": GenericCSVReader,
    ".txt": GenericTXTReader,
    ".tsv": lambda: GenericCSVReader(delimiter="\t"),
    ".dat": GenericTXTReader,  # Common data file extension
    ".data": GenericTXTReader,
}


def register_reader(extension: str, reader_class: type | Callable[[], Any]) -> None:
    """Register a custom reader for a file extension.

    This allows users to add support for custom file formats without modifying
    the core library.

    Parameters
    ----------
    extension : str
        File extension (should include the dot, e.g., ".xyz")
    reader_class : Type | Callable
        Reader class or factory function that returns a reader instance.
        The reader must implement a ``read(filepath)`` method that returns
        a Measurement object.

    Examples
    --------
    Register a custom reader class:

    >>> class MyCustomReader:
    ...     def read(self, filepath):
    ...         # ... custom reading logic
    ...         pass
    >>> register_reader('.xyz', MyCustomReader)

    Register a factory function:

    >>> register_reader('.custom', lambda: GenericCSVReader(delimiter='|'))

    Notes
    -----
    Custom readers should follow the same interface as GenericCSVReader,
    implementing a ``read(filepath)`` method that returns a Measurement.
    """
    _READER_REGISTRY[extension.lower()] = reader_class


def detect_reader(filepath: str | Path) -> GenericCSVReader | GenericTXTReader:
    """Auto-detect appropriate reader for file.

    Uses a multi-layer detection strategy:

    1. **Extension-based**: Matches file extension to registered readers
    2. **Header-based**: Checks file headers for instrument signatures (future)
    3. **Content-based**: Analyzes file content structure (future)
    4. **Fallback**: Returns generic reader based on best guess

    Parameters
    ----------
    filepath : str | Path
        Path to file

    Returns
    -------
    Reader instance
        Instance of appropriate reader class

    Examples
    --------
    >>> reader = detect_reader("data.csv")
    >>> isinstance(reader, GenericCSVReader)
    True

    >>> reader = detect_reader("data.txt")
    >>> isinstance(reader, GenericTXTReader)
    True

    Notes
    -----
    Currently implements Layer 1 (extension-based) and Layer 4 (fallback).
    Layers 2 and 3 are reserved for future extensions to detect specific
    instrument file formats.
    """
    filepath = Path(filepath)

    # Layer 1: Extension-based detection
    ext = filepath.suffix.lower()
    if ext in _READER_REGISTRY:
        reader_class = _READER_REGISTRY[ext]
        if callable(reader_class) and not isinstance(reader_class, type):
            # Factory function
            return reader_class()  # type: ignore[no-any-return]
        else:
            # Class constructor
            return reader_class()  # type: ignore[no-any-return]

    # Layer 2: Header-based detection (future implementation)
    # Could read first few lines and check for instrument signatures
    # e.g., "# Keithley 2400", "# Agilent 34401A"

    # Layer 3: Content-based detection (future implementation)
    # Could analyze file structure to determine format
    # e.g., detect delimiter, number of columns, data types

    # Layer 4: Fallback to generic readers
    # Try to make an educated guess based on extension
    if ext in [".dat", ".data", ""]:
        return GenericTXTReader()
    else:
        # Default to CSV reader for unknown extensions
        return GenericCSVReader()


def read_file(filepath: str | Path) -> Measurement:
    """Read file with automatic format detection.

    This is the main entry point for reading individual files. It automatically
    detects the file format and uses the appropriate reader.

    Parameters
    ----------
    filepath : str | Path
        Path to file

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

    Examples
    --------
    Read a CSV file:

    >>> measurement = read_file("data.csv")

    Read a TXT file:

    >>> measurement = read_file("experiment.txt")

    Read with explicit path:

    >>> from pathlib import Path
    >>> measurement = read_file(Path("/data/experiment/sample1.csv"))

    Notes
    -----
    This function combines detection and reading in a single call. For more
    control over the reading process, you can use ``detect_reader()`` followed
    by calling the reader's ``read()`` method directly.
    """
    reader = detect_reader(filepath)
    return reader.read(filepath)


# Export public API
__all__ = [
    "GenericCSVReader",
    "GenericTXTReader",
    "detect_reader",
    "read_file",
    "register_reader",
]
