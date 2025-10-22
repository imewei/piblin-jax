"""Metadata utilities for managing, validating, extracting, and merging metadata.

This module provides utilities for working with metadata (conditions and details)
across the data hierarchy. Metadata is separated into:

- **Conditions**: Experimental parameters that define comparability between datasets
  (e.g., temperature, pressure, concentration)
- **Details**: Contextual information that doesn't affect experimental conditions
  (e.g., operator, date, notes)

The module supports:
- Merging metadata from multiple sources with configurable conflict resolution
- Separating conditions from details using explicit keys or heuristics
- Validating metadata against schemas with type checking
- Extracting metadata from filenames, paths, and file headers
"""

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any


def merge_metadata(
    metadata_list: list[dict[str, Any]], strategy: str = "override"
) -> dict[str, Any]:
    """Merge multiple metadata dictionaries.

    Combines metadata from multiple sources with configurable conflict resolution.
    Metadata dictionaries are processed in order, with later dictionaries having
    higher priority (for 'override' strategy).

    Parameters
    ----------
    metadata_list : list[dict[str, Any]]
        List of metadata dictionaries to merge (in priority order).
        Earlier dictionaries have lower priority for conflict resolution.
    strategy : str, optional
        Conflict resolution strategy (default: "override"):

        - 'override': Later values override earlier ones
        - 'keep_first': Keep first value encountered
        - 'raise': Raise ValueError on conflicts
        - 'list': Collect conflicting values in a list (duplicates removed)

    Returns
    -------
    dict[str, Any]
        Merged metadata dictionary

    Raises
    ------
    ValueError
        If strategy is 'raise' and conflicts are detected, or if strategy
        is unknown

    Examples
    --------
    >>> meta1 = {"temp": 20, "sample": "A1"}
    >>> meta2 = {"temp": 25, "pressure": 1.0}
    >>> merge_metadata([meta1, meta2])
    {'temp': 25, 'sample': 'A1', 'pressure': 1.0}

    >>> merge_metadata([meta1, meta2], strategy="keep_first")
    {'temp': 20, 'sample': 'A1', 'pressure': 1.0}

    >>> merge_metadata([meta1, meta2], strategy="list")
    {'temp': [20, 25], 'sample': 'A1', 'pressure': 1.0}
    """
    if not metadata_list:
        return {}

    result = {}
    for metadata in metadata_list:
        for key, value in metadata.items():
            if key not in result:
                result[key] = value
            else:
                # Conflict detected
                if strategy == "override":
                    result[key] = value
                elif strategy == "keep_first":
                    pass  # Keep existing value
                elif strategy == "raise":
                    if result[key] != value:
                        raise ValueError(
                            f"Metadata conflict for key '{key}': {result[key]} vs {value}"
                        )
                elif strategy == "list":
                    if isinstance(result[key], list):
                        if value not in result[key]:
                            result[key].append(value)
                    else:
                        if result[key] != value:
                            result[key] = [result[key], value]
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")

    return result


def separate_conditions_details(
    metadata: dict[str, Any], condition_keys: list[str] | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate metadata into conditions and details.

    Conditions are experimental parameters that define comparability between
    datasets (e.g., temperature, pressure). Details are contextual information
    (e.g., operator, date, notes).

    Parameters
    ----------
    metadata : dict[str, Any]
        Combined metadata dictionary
    condition_keys : list[str] | None, optional
        Known condition keys (experimental parameters).
        If None, heuristics are used to identify conditions based on
        common experimental parameter names.

    Returns
    -------
    conditions : dict[str, Any]
        Experimental conditions (parameters defining comparability)
    details : dict[str, Any]
        Context information (non-experimental metadata)

    Examples
    --------
    >>> metadata = {"temp": 25, "pressure": 1.0, "operator": "John"}
    >>> conditions, details = separate_conditions_details(
    ...     metadata,
    ...     condition_keys=["temp", "pressure"]
    ... )
    >>> conditions
    {'temp': 25, 'pressure': 1.0}
    >>> details
    {'operator': 'John'}

    Using heuristics:

    >>> metadata = {"temperature": 25, "strain": 0.1, "notes": "Trial 1"}
    >>> conditions, details = separate_conditions_details(metadata)
    >>> "temperature" in conditions
    True
    >>> "notes" in details
    True
    """
    if condition_keys is None:
        # Use heuristics to identify conditions
        # Conditions typically: temperature, pressure, concentration, etc.
        condition_key_patterns = [
            "temp",
            "temperature",
            "pressure",
            "concentration",
            "frequency",
            "strain",
            "stress",
            "time",
            "wavelength",
            "ph",
            "humidity",
            "voltage",
            "current",
            "power",
        ]
        condition_keys = [
            key
            for key in metadata
            if any(pattern in key.lower() for pattern in condition_key_patterns)
        ]

    conditions = {k: v for k, v in metadata.items() if k in condition_keys}
    details = {k: v for k, v in metadata.items() if k not in condition_keys}

    return conditions, details


def validate_metadata(
    metadata: dict[str, Any],
    schema: dict[str, type | Callable[[Any], bool]] | None = None,
    required_keys: list[str] | None = None,
) -> bool:
    """Validate metadata against a schema.

    Performs type checking and required key validation. Validation is optional
    and can be configured with schema and required_keys parameters.

    Parameters
    ----------
    metadata : dict[str, Any]
        Metadata to validate
    schema : dict[str, type | Callable[[Any], bool]] | None, optional
        Schema defining expected types or validation functions.
        Keys are metadata field names, values are either:

        - Type objects (e.g., float, str, int) for type checking
        - Callable validators that return True if valid

        Example: ``{'temperature': float, 'sample_id': str}``
    required_keys : list[str] | None, optional
        Keys that must be present in metadata

    Returns
    -------
    bool
        True if valid

    Raises
    ------
    ValueError
        If validation fails (missing required keys, type mismatch,
        or custom validation function returns False)

    Examples
    --------
    Type checking:

    >>> metadata = {"temp": 25.0, "sample": "A1"}
    >>> schema = {"temp": float, "sample": str}
    >>> validate_metadata(metadata, schema=schema)
    True

    Required keys:

    >>> validate_metadata(metadata, required_keys=["temp", "sample"])
    True

    Custom validation:

    >>> schema = {"ph": lambda x: 0 <= x <= 14}
    >>> validate_metadata({"ph": 7.0}, schema=schema)
    True
    """
    # Check required keys
    if required_keys:
        missing = set(required_keys) - set(metadata.keys())
        if missing:
            raise ValueError(f"Missing required metadata keys: {missing}")

    # Type checking
    if schema:
        for key, expected_type in schema.items():
            if key in metadata:
                value = metadata[key]
                if isinstance(expected_type, type):
                    if not isinstance(value, expected_type):
                        raise ValueError(
                            f"Metadata '{key}' has incorrect type: "
                            f"expected {expected_type.__name__}, "
                            f"got {type(value).__name__}"
                        )
                elif callable(expected_type):
                    # Custom validation function
                    if not expected_type(value):
                        raise ValueError(f"Metadata '{key}' failed validation")

    return True


def parse_key_value_string(text: str, separator: str = "=", delimiter: str = ",") -> dict[str, str]:
    """Parse key-value pairs from a string.

    Extracts metadata from delimited key-value strings commonly found in
    filenames, headers, or configuration strings.

    Parameters
    ----------
    text : str
        String containing key-value pairs.
        Example: ``"temp=25,pressure=1.0,sample=A1"``
    separator : str, optional
        Character separating keys from values (default: "=")
    delimiter : str, optional
        Character separating pairs (default: ",")

    Returns
    -------
    dict[str, str]
        Parsed metadata (all values are strings, convert as needed)

    Examples
    --------
    >>> parse_key_value_string("temp=25,pressure=1.0")
    {'temp': '25', 'pressure': '1.0'}

    >>> parse_key_value_string("temp:25;pressure:1.0", separator=":", delimiter=";")
    {'temp': '25', 'pressure': '1.0'}
    """
    metadata = {}
    pairs = text.split(delimiter)
    for pair in pairs:
        pair = pair.strip()
        if separator in pair:
            key, value = pair.split(separator, 1)
            metadata[key.strip()] = value.strip()
    return metadata


def extract_from_filename(filename: str | Path, pattern: str | None = None) -> dict[str, str]:
    """Extract metadata from filename using regex pattern.

    Parses filenames to extract metadata using either custom regex patterns
    or common heuristics for scientific data files.

    Parameters
    ----------
    filename : str | Path
        Filename or path (extension is removed before matching)
    pattern : str | None, optional
        Regex pattern with named groups for extraction.
        If None, uses common heuristics for sample names, temperatures,
        and replicate numbers.

    Returns
    -------
    dict[str, str]
        Extracted metadata (all values are strings)

    Examples
    --------
    Using heuristics:

    >>> extract_from_filename("sample_A1_temp_25C_001.csv")
    {'sample': 'A1', 'temp': '25', 'replicate': '001'}

    Using custom pattern:

    >>> pattern = r"(?P<sample>\\w+)_(?P<temp>\\d+)C"
    >>> extract_from_filename("sample_A1_25C.csv", pattern)
    {'sample': 'A1', 'temp': '25'}
    """
    if isinstance(filename, Path):
        filename = filename.stem  # Remove extension
    else:
        filename = Path(filename).stem

    metadata = {}

    if pattern:
        match = re.search(pattern, filename)
        if match:
            metadata = match.groupdict()
    else:
        # Heuristic patterns for common scientific filename conventions
        # Use non-greedy matching and word boundaries
        patterns = [
            r"sample[_-]?(?P<sample>[A-Za-z0-9]+?)(?:[_-]|$)",
            r"temp[_-]?(?P<temp>\d+\.?\d*)",
            r"(?P<replicate>\d{3,})$",  # Trailing numbers (3+ digits)
        ]
        for p in patterns:
            match = re.search(p, filename, re.IGNORECASE)
            if match:
                metadata.update(match.groupdict())

    return metadata


def extract_from_path(filepath: str | Path, level_names: list[str] | None = None) -> dict[str, str]:
    """Extract metadata from directory structure.

    Parses directory hierarchy to extract metadata based on directory names
    at different levels.

    Parameters
    ----------
    filepath : str | Path
        File path
    level_names : list[str] | None, optional
        Names for each directory level (from deepest to root).
        Example: ``['sample', 'experiment', 'project']`` extracts
        sample from parent directory, experiment from grandparent, etc.
        If None, returns empty dict.

    Returns
    -------
    dict[str, str]
        Extracted metadata

    Examples
    --------
    >>> extract_from_path(
    ...     "/data/ProjectA/ExpB/SampleC/data.csv",
    ...     ['sample', 'experiment', 'project']
    ... )
    {'sample': 'SampleC', 'experiment': 'ExpB', 'project': 'ProjectA'}
    """
    path = Path(filepath)
    parts = path.parts[:-1]  # Exclude filename

    metadata = {}
    if level_names and parts:
        for i, name in enumerate(level_names):
            if i < len(parts):
                metadata[name] = parts[-(i + 1)]

    return metadata


def parse_header_metadata(
    header_lines: list[str], comment_char: str = "#", separator: str = ":"
) -> dict[str, str]:
    """Parse metadata from file header comment lines.

    Extracts metadata from comment lines in file headers, commonly used in
    scientific data files to store experimental conditions and context.

    Parameters
    ----------
    header_lines : list[str]
        Lines from file header
    comment_char : str, optional
        Comment character (default: "#")
    separator : str, optional
        Character separating keys from values (default: ":")

    Returns
    -------
    dict[str, str]
        Parsed metadata (all values are strings)

    Examples
    --------
    >>> lines = [
    ...     "# Temperature: 25",
    ...     "# Pressure: 1.0",
    ...     "# Sample: A1"
    ... ]
    >>> parse_header_metadata(lines)
    {'Temperature': '25', 'Pressure': '1.0', 'Sample': 'A1'}

    With custom separators:

    >>> lines = ["// Temp = 25", "// Sample = A1"]
    >>> parse_header_metadata(lines, comment_char="//", separator="=")
    {'Temp': '25', 'Sample': 'A1'}
    """
    metadata = {}
    for line in header_lines:
        line = line.strip()
        if line.startswith(comment_char):
            line = line[len(comment_char) :].strip()
            if separator in line:
                key, value = line.split(separator, 1)
                metadata[key.strip()] = value.strip()
    return metadata
