"""
ExperimentSet class for piblin-jax.

Top-level container for multiple Experiment objects representing a study or project.
"""

from collections.abc import Iterator
from typing import Any

from .experiment import Experiment


class ExperimentSet:
    """
    Top-level container for multiple Experiment objects.

    An ExperimentSet represents the highest level of the data hierarchy,
    typically corresponding to:
    - Complete research project or study
    - Multi-sample analysis
    - Entire experimental campaign
    - Publication dataset

    This is the entry point for organizing and managing entire experimental
    datasets with consistent metadata and structure.

    Parameters
    ----------
    experiments : list[Experiment]
        List of Experiment objects in this set.
    conditions : dict[str, Any] | None, optional
        Global conditions for the entire study
        (e.g., project name, year, instrument, principal investigator).
    details : dict[str, Any] | None, optional
        Additional context for the study
        (e.g., publication info, funding source, study objectives).

    Attributes
    ----------
    experiments : tuple[Experiment, ...]
        Immutable tuple of experiments in this set.
    conditions : dict[str, Any]
        Global metadata for the entire study.
    details : dict[str, Any]
        Additional metadata for the study.

    Notes
    -----
    The experiments are stored as a tuple to ensure immutability, which is
    required for JAX transformations. Individual experiments can be accessed
    by indexing or iteration.

    Hierarchy level:
    **ExperimentSet** → Experiment → MeasurementSet → Measurement → Dataset

    This is the top level of the hierarchy and provides global context for
    all contained data.

    Examples
    --------
    >>> import numpy as np
    >>> from piblin_jax.data.datasets import OneDimensionalDataset
    >>> from piblin_jax.data.collections import (
    ...     Measurement, MeasurementSet, Experiment, ExperimentSet
    ... )
    >>>
    >>> # Create experiments for multiple samples
    >>> experiments = []
    >>>
    >>> for sample_id in ['S001', 'S002', 'S003']:
    ...     # Create measurements for this sample
    ...     x = np.linspace(0, 10, 100)
    ...     y = np.sin(x) * (ord(sample_id[-1]) - ord('0'))
    ...     ds = OneDimensionalDataset(x, y)
    ...     m = Measurement([ds])
    ...     ms = MeasurementSet([m])
    ...
    ...     # Create experiment for this sample
    ...     exp = Experiment(
    ...         [ms],
    ...         conditions={"sample": sample_id, "date": "2025-10-18"}
    ...     )
    ...     experiments.append(exp)
    >>>
    >>> # Create experiment set for the complete study
    >>> study = ExperimentSet(
    ...     experiments=experiments,
    ...     conditions={
    ...         "project": "QuantIQ-2025",
    ...         "instrument": "Spectrometer-X",
    ...         "year": 2025
    ...     },
    ...     details={
    ...         "pi": "Dr. Jane Smith",
    ...         "funding": "NSF Grant 12345",
    ...         "description": "Comparative spectroscopy study"
    ...     }
    ... )
    >>>
    >>> len(study)
    3
    >>> study.conditions["project"]
    'QuantIQ-2025'
    >>> study[0].conditions["sample"]
    'S001'
    >>> study.details["pi"]
    'Dr. Jane Smith'
    """

    def __init__(
        self,
        experiments: list[Experiment],
        conditions: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
    ):
        """
        Initialize ExperimentSet with experiments and metadata.

        Parameters
        ----------
        experiments : list[Experiment]
            List of Experiment objects in this set.
        conditions : dict[str, Any] | None, optional
            Global conditions for the entire study.
        details : dict[str, Any] | None, optional
            Additional context for the study.
        """
        self._experiments = tuple(experiments)  # Immutable for JAX compatibility
        self._conditions = conditions if conditions is not None else {}
        self._details = details if details is not None else {}

    @property
    def experiments(self) -> tuple[Experiment, ...]:
        """
        Get all experiments in this set.

        :no-index:

        Returns
        -------
        tuple[Experiment, ...]
            Immutable tuple of Experiment objects.

        Examples
        --------
        >>> study.experiments
        (<Experiment at 0x...>, <Experiment at 0x...>, <Experiment at 0x...>)
        """
        return self._experiments

    @property
    def conditions(self) -> dict[str, Any]:
        """
        Get global conditions for the entire study.

        :no-index:

        Returns
        -------
        dict[str, Any]
            Dictionary of global metadata (project, year, instrument, etc.).

        Examples
        --------
        >>> study.conditions
        {'project': 'QuantIQ-2025', 'instrument': 'Spectrometer-X', 'year': 2025}
        """
        return self._conditions

    @property
    def details(self) -> dict[str, Any]:
        """
        Get additional details for the study.

        :no-index:

        Returns
        -------
        dict[str, Any]
            Dictionary of additional context (PI, funding, objectives, etc.).

        Examples
        --------
        >>> study.details
        {'pi': 'Dr. Jane Smith', 'funding': 'NSF Grant 12345', 'description': '...'}
        """
        return self._details

    def __len__(self) -> int:
        """
        Get number of experiments in this set.

        Returns
        -------
        int
            Number of experiments.

        Examples
        --------
        >>> len(study)
        3
        """
        return len(self._experiments)

    def __iter__(self) -> Iterator[Experiment]:
        """
        Iterate over experiments in this set.

        Yields
        ------
        Experiment
            Each experiment in order.

        Examples
        --------
        >>> for exp in study:
        ...     print(exp.conditions["sample"])
        S001
        S002
        S003
        """
        return iter(self._experiments)

    def __getitem__(self, index: int | slice) -> Experiment | tuple[Experiment, ...]:
        """
        Get experiment by index.

        Parameters
        ----------
        index : int or slice
            Index or slice to access experiments.

        Returns
        -------
        Experiment or tuple[Experiment, ...]
            Experiment at the given index, or tuple of experiments for slice.

        Examples
        --------
        >>> study[0]
        <Experiment at 0x...>
        >>> study[0:2]
        (<Experiment at 0x...>, <Experiment at 0x...>)
        """
        return self._experiments[index]

    def get_experiment_by_condition(self, **condition_filters: Any) -> list[Experiment]:
        """
        Get experiments matching specified conditions.

        Parameters
        ----------
        **condition_filters
            Keyword arguments specifying condition values to match.
            Only experiments where ALL specified conditions match
            the given values will be included.

        Returns
        -------
        list[Experiment]
            List of experiments matching the conditions.

        Examples
        --------
        >>> # Get all experiments for sample S001
        >>> s001_exps = study.get_experiment_by_condition(sample="S001")
        >>> len(s001_exps)
        1
        >>> s001_exps[0].conditions["sample"]
        'S001'
        >>>
        >>> # Get experiments matching multiple conditions
        >>> dated_exps = study.get_experiment_by_condition(
        ...     date="2025-10-18",
        ...     sample="S002"
        ... )
        """
        matching = []

        for experiment in self.experiments:
            # Check if all specified conditions match
            matches = all(
                experiment.conditions.get(key) == value for key, value in condition_filters.items()
            )
            if matches:
                matching.append(experiment)

        return matching
