"""Data I/O system for piblin-jax.

This module provides a comprehensive file I/O system with:
- Generic CSV and TXT readers
- Auto-detection of file formats
- Batch reading of multiple files
- Automatic hierarchy building from file lists
- Extensible reader registry

Main Functions
--------------
read_file : Read single file with auto-detection
read_files : Read multiple files and build hierarchy
read_directory : Read all matching files in a directory
read_directories : Read multiple directories

Examples
--------
Read a single file:

>>> from piblin_jax.dataio import read_file
>>> measurement = read_file("data.csv")

Read multiple files:

>>> files = ["sample1.csv", "sample2.csv", "sample3.csv"]
>>> experiment_set = read_files(files)

Read an entire directory:

>>> experiment_set = read_directory("/path/to/data", pattern="*.csv")

Read multiple directories:

>>> paths = ["/path/to/exp1", "/path/to/exp2"]
>>> experiment_set = read_directories(paths)
"""

from collections.abc import Sequence
from pathlib import Path

from piblin_jax.data.collections import ExperimentSet

from .hierarchy import build_hierarchy
from .readers import detect_reader, read_file, register_reader


def read_files(file_list: Sequence[str | Path]) -> ExperimentSet:
    """Read multiple files and build hierarchical structure.

    Reads all files in the list, automatically detecting formats, and
    organizes them into a hierarchical ExperimentSet based on their
    experimental conditions.

    Parameters
    ----------
    file_list : Sequence[str | Path]
        List of file paths to read

    Returns
    -------
    ExperimentSet
        Hierarchical organization of all measurements

    Raises
    ------
    FileNotFoundError
        If any file in the list does not exist
    ValueError
        If any file cannot be parsed

    Examples
    --------
    Read specific files:

    >>> files = ["sample1.csv", "sample2.csv", "sample3.csv"]
    >>> experiment_set = read_files(files)
    >>> len(experiment_set.experiments)
    1

    With Path objects:

    >>> from pathlib import Path
    >>> files = list(Path("/data").glob("*.csv"))
    >>> experiment_set = read_files(files)

    Notes
    -----
    All measurements from all files are analyzed together to identify
    constant and varying conditions, which determines the hierarchy
    structure. Files with the same conditions are grouped together.
    """
    if not file_list:
        return ExperimentSet([])

    # Read all files
    measurements = [read_file(f) for f in file_list]

    # Build hierarchy from measurements
    return build_hierarchy(measurements)


def read_directory(
    path: str | Path, pattern: str = "*.csv", recursive: bool = False
) -> ExperimentSet:
    """Read all matching files in a directory.

    Scans a directory for files matching the pattern, reads them all,
    and builds a hierarchical structure.

    Parameters
    ----------
    path : str | Path
        Directory path to scan
    pattern : str, optional
        Glob pattern for file matching (default: ``"*.csv"``).
        Examples: "*.txt", "*.dat", "sample_*.csv"
    recursive : bool, optional
        If True, search recursively in subdirectories (default: False)

    Returns
    -------
    ExperimentSet
        Hierarchical organization of all measurements

    Raises
    ------
    FileNotFoundError
        If the directory does not exist
    ValueError
        If any file cannot be parsed

    Examples
    --------
    Read all CSV files in a directory:

    >>> experiment_set = read_directory("/data/experiment1")

    Read all TXT files:

    >>> experiment_set = read_directory("/data/experiment1", pattern="*.txt")

    Read recursively:

    >>> experiment_set = read_directory(
    ...     "/data",
    ...     pattern="*.csv",
    ...     recursive=True
    ... )

    Read with custom pattern:

    >>> experiment_set = read_directory(
    ...     "/data",
    ...     pattern="sample_A*.csv"
    ... )

    Notes
    -----
    Files are sorted alphabetically before reading for consistent ordering.
    All measurements are analyzed together to build the hierarchy.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")

    if not path.is_dir():
        raise ValueError(f"Path is not a directory: {path}")

    # Find matching files
    if recursive:
        files = sorted(path.rglob(pattern))
    else:
        files = sorted(path.glob(pattern))

    if not files:
        # Return empty ExperimentSet if no files found
        return ExperimentSet([])

    return read_files(files)


def read_directories(
    path_list: Sequence[str | Path], pattern: str = "*.csv", recursive: bool = False
) -> ExperimentSet:
    """Read multiple directories and combine into single hierarchy.

    Scans multiple directories for matching files and builds a unified
    hierarchical structure from all measurements.

    Parameters
    ----------
    path_list : Sequence[str | Path]
        List of directory paths to scan
    pattern : str, optional
        Glob pattern for file matching (default: ``"*.csv"``)
    recursive : bool, optional
        If True, search recursively in subdirectories (default: False)

    Returns
    -------
    ExperimentSet
        Hierarchical organization of all measurements from all directories

    Raises
    ------
    FileNotFoundError
        If any directory does not exist
    ValueError
        If any file cannot be parsed

    Examples
    --------
    Read from multiple directories:

    >>> paths = ["/data/exp1", "/data/exp2", "/data/exp3"]
    >>> experiment_set = read_directories(paths)

    With custom pattern:

    >>> experiment_set = read_directories(
    ...     paths,
    ...     pattern="*.txt"
    ... )

    Recursive search:

    >>> experiment_set = read_directories(
    ...     paths,
    ...     recursive=True
    ... )

    Notes
    -----
    All measurements from all directories are combined and analyzed together
    to build a unified hierarchy. This is useful when an experiment spans
    multiple directories.
    """
    if not path_list:
        return ExperimentSet([])

    # Collect all files from all directories
    all_files = []
    for path in path_list:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")

        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")

        # Find matching files
        if recursive:
            files = list(path.rglob(pattern))
        else:
            files = list(path.glob(pattern))

        all_files.extend(files)

    # Sort for consistent ordering
    all_files = sorted(all_files)

    if not all_files:
        return ExperimentSet([])

    return read_files(all_files)


# Re-export main functions from readers
__all__ = [
    "build_hierarchy",
    "detect_reader",
    "read_directories",
    "read_directory",
    "read_file",
    "read_files",
    "register_reader",
]
