"""Hierarchy building algorithm for organizing measurements.

This module provides algorithms for building hierarchical data structures
from flat lists of measurements by analyzing conditions and grouping data.
"""

from typing import Any

from piblin_jax.data.collections import Experiment, ExperimentSet, Measurement, MeasurementSet


def build_hierarchy(measurements: list[Measurement]) -> ExperimentSet:
    """Build hierarchical structure from flat list of measurements.

    Analyzes the conditions across all measurements and organizes them into
    a hierarchical structure:

    1. Extract all conditions from all measurements
    2. Identify constant conditions (same across all) -> Experiment level
    3. Identify varying conditions -> MeasurementSet grouping
    4. Group measurements by conditions

    Parameters
    ----------
    measurements : list[Measurement]
        Flat list of measurements to organize

    Returns
    -------
    ExperimentSet
        Hierarchical organization of measurements

    Notes
    -----
    Current implementation creates a simple hierarchy:
    - One ExperimentSet containing
    - One Experiment containing
    - One MeasurementSet with all measurements

    Future enhancements can implement more sophisticated grouping based on
    varying conditions to create multiple Experiments and MeasurementSets.

    Examples
    --------
    Build hierarchy from file list:

    >>> from piblin_jax.dataio.readers import read_file
    >>> measurements = [
    ...     read_file("sample1.csv"),
    ...     read_file("sample2.csv"),
    ... ]
    >>> experiment_set = build_hierarchy(measurements)
    >>> len(experiment_set.experiments)
    1

    Access measurements:

    >>> for exp in experiment_set.experiments:
    ...     for ms in exp.measurement_sets:
    ...         for m in ms.measurements:
    ...             print(m.conditions)
    """
    if not measurements:
        return ExperimentSet([])

    # Extract all condition keys across all measurements
    all_condition_keys: set[str] = set()
    for m in measurements:
        all_condition_keys.update(m.conditions.keys())

    # Identify constant vs varying conditions
    constant_conditions = {}
    varying_keys = set()

    for key in all_condition_keys:
        # Collect all unique values for this key
        values = set()
        for m in measurements:
            if key in m.conditions:
                # Convert to string for comparison (handles numeric types)
                values.add(str(m.conditions[key]))

        if len(values) == 1:
            # Constant across all measurements
            # Get the original value (not string) from first measurement
            for m in measurements:
                if key in m.conditions:
                    constant_conditions[key] = m.conditions[key]
                    break
        elif len(values) > 1:
            # Varies across measurements
            varying_keys.add(key)

    # For simplicity, create one Experiment with one MeasurementSet
    # containing all measurements
    # More sophisticated grouping can be added in future versions
    measurement_set = MeasurementSet(measurements=measurements, conditions=constant_conditions)

    experiment = Experiment(measurement_sets=[measurement_set], conditions=constant_conditions)

    experiment_set = ExperimentSet(experiments=[experiment], conditions=constant_conditions)

    return experiment_set


def group_by_conditions(
    measurements: list[Measurement], grouping_keys: list[str]
) -> dict[tuple[Any, ...], list[Measurement]]:
    """Group measurements by specific condition keys.

    This is a utility function for more advanced hierarchy building that
    groups measurements based on specific condition values.

    Parameters
    ----------
    measurements : list[Measurement]
        Measurements to group
    grouping_keys : list[str]
        Condition keys to group by

    Returns
    -------
    dict[tuple[Any, ...], list[Measurement]]
        Dictionary mapping condition value tuples to lists of measurements

    Examples
    --------
    Group by temperature:

    >>> groups = group_by_conditions(measurements, ['Temperature'])
    >>> for temp_value, group in groups.items():
    ...     print(f"Temperature {temp_value}: {len(group)} measurements")

    Group by multiple conditions:

    >>> groups = group_by_conditions(
    ...     measurements,
    ...     ['Temperature', 'Pressure']
    ... )

    Notes
    -----
    This function is provided for future extensions to the hierarchy building
    algorithm that may want to create separate Experiments or MeasurementSets
    based on specific conditions.
    """
    groups: dict[tuple[Any, ...], list[Measurement]] = {}

    for measurement in measurements:
        # Extract values for grouping keys
        key_values = tuple(measurement.conditions.get(key, None) for key in grouping_keys)

        if key_values not in groups:
            groups[key_values] = []

        groups[key_values].append(measurement)

    return groups


def identify_varying_conditions(measurements: list[Measurement]) -> set[str]:
    """Identify which conditions vary across measurements.

    Parameters
    ----------
    measurements : list[Measurement]
        Measurements to analyze

    Returns
    -------
    set[str]
        Set of condition keys that have different values across measurements

    Examples
    --------
    >>> varying = identify_varying_conditions(measurements)
    >>> print(varying)
    {'Temperature', 'Sample'}

    Notes
    -----
    This function is useful for determining which conditions should be used
    to group measurements into different MeasurementSets or Experiments.
    """
    if not measurements:
        return set()

    # Collect all condition keys
    all_keys: set[str] = set()
    for m in measurements:
        all_keys.update(m.conditions.keys())

    # Identify varying keys
    varying = set()
    for key in all_keys:
        values = set()
        for m in measurements:
            if key in m.conditions:
                values.add(str(m.conditions[key]))

        if len(values) > 1:
            varying.add(key)

    return varying
