"""
Base dataset class for piblin-jax.

Provides the abstract base class for all dataset types with metadata support.
"""

import copy
from abc import ABC
from typing import Any, Self


class Dataset(ABC):
    """
    Abstract base class for all dataset types.

    All piblin-jax datasets inherit from this class and provide:
    - Metadata system (conditions and details)
    - Internal storage using backend arrays (JAX or NumPy)
    - External NumPy conversion for API boundaries
    - Immutable design for JAX compatibility

    Parameters
    ----------
    conditions : dict[str, Any] | None, optional
        Experimental conditions (temperature, pressure, flow rate, etc.).
        Default is empty dict.
    details : dict[str, Any] | None, optional
        Additional context (sample ID, operator, instrument, date, etc.).
        Default is empty dict.

    Attributes
    ----------
    conditions : dict[str, Any]
        Experimental conditions associated with the dataset.
    details : dict[str, Any]
        Additional metadata and context for the dataset.

    Notes
    -----
    This class cannot be instantiated directly. Use one of the concrete
    dataset types:
    - ZeroDimensionalDataset (0D)
    - OneDimensionalDataset (1D)
    - TwoDimensionalDataset (2D)
    - ThreeDimensionalDataset (3D)
    - Histogram
    - Distribution
    - OneDimensionalCompositeDataset

    The dataset uses an immutable design pattern to ensure compatibility
    with JAX transformations (jit, grad, vmap). Arrays are stored internally
    as backend arrays (JAX DeviceArray when available, NumPy ndarray otherwise)
    and converted to NumPy arrays when accessed through properties.

    Examples
    --------
    >>> from piblin_jax.data.datasets import OneDimensionalDataset
    >>> import numpy as np
    >>> x = np.linspace(0, 10, 100)
    >>> y = np.sin(x)
    >>> conditions = {"temperature": 25.0, "sample": "A"}
    >>> details = {"operator": "Jane Doe", "date": "2025-10-18"}
    >>> dataset = OneDimensionalDataset(
    ...     independent_variable_data=x,
    ...     dependent_variable_data=y,
    ...     conditions=conditions,
    ...     details=details
    ... )
    >>> dataset.conditions["temperature"]
    25.0
    >>> type(dataset.independent_variable_data)
    <class 'numpy.ndarray'>
    """

    def __init__(
        self, conditions: dict[str, Any] | None = None, details: dict[str, Any] | None = None
    ):
        """
        Initialize Dataset with metadata.

        Parameters
        ----------
        conditions : dict[str, Any] | None, optional
            Experimental conditions.
        details : dict[str, Any] | None, optional
            Additional context and metadata.
        """
        self._conditions = conditions if conditions is not None else {}
        self._details = details if details is not None else {}

        # Uncertainty quantification attributes (Task Group 12)
        self._uncertainty_samples: dict[str, Any] | None = None
        self._credible_intervals: tuple[Any, Any] | None = None
        self._uncertainty_method: str | None = None

    @property
    def conditions(self) -> dict[str, Any]:
        """
        Get experimental conditions.

        :no-index:

        Returns
        -------
        dict[str, Any]
            Dictionary of experimental conditions (temperature, pressure, etc.).

        Examples
        --------
        >>> dataset.conditions
        {'temperature': 25.0, 'pressure': 1.0, 'sample': 'A'}
        """
        return self._conditions

    @property
    def details(self) -> dict[str, Any]:
        """
        Get additional dataset details.

        :no-index:

        Returns
        -------
        dict[str, Any]
            Dictionary of additional context (operator, instrument, date, etc.).

        Examples
        --------
        >>> dataset.details
        {'operator': 'Jane Doe', 'instrument': 'Spectrometer X', 'date': '2025-10-18'}
        """
        return self._details

    @property
    def has_uncertainty(self) -> bool:
        """
        Check if dataset has uncertainty information.

        :no-index:

        Returns
        -------
        bool
            True if dataset has uncertainty information, False otherwise.

        Examples
        --------
        >>> dataset.has_uncertainty
        False
        >>> dataset_with_unc = dataset.with_uncertainty(n_samples=1000)
        >>> dataset_with_unc.has_uncertainty
        True

        Notes
        -----
        This property checks for the presence of either uncertainty samples
        or cached credible intervals. It does not validate the uncertainty
        quantification method or parameter values.
        """
        return self._uncertainty_samples is not None or self._credible_intervals is not None

    @property
    def uncertainty_samples(self) -> Any | None:
        """
        Get uncertainty samples (if keep_samples=True was used).

        :no-index:

        Returns
        -------
        dict | None
            Posterior samples from Bayesian inference if keep_samples=True,
            None otherwise.

        Examples
        --------
        >>> dataset_with_unc = dataset.with_uncertainty(
        ...     n_samples=1000,
        ...     method='bayesian',
        ...     keep_samples=True
        ... )
        >>> samples = dataset_with_unc.uncertainty_samples
        >>> sigma_samples = samples['sigma']

        Notes
        -----
        Storing samples can be memory-intensive for large datasets. Use
        keep_samples=False if you only need credible intervals.
        """
        return self._uncertainty_samples

    @property
    def credible_intervals(self) -> Any | None:
        """
        Get cached credible intervals.

        :no-index:

        Returns
        -------
        tuple | None
            Cached credible intervals (lower, upper) if computed,
            None otherwise.

        Examples
        --------
        >>> dataset_with_unc = dataset.with_uncertainty(n_samples=1000)
        >>> intervals = dataset_with_unc.credible_intervals
        >>> if intervals is not None:
        ...     lower, upper = intervals

        Notes
        -----
        Credible intervals are cached after computation to avoid
        recomputation. Use get_credible_intervals() to compute
        intervals with custom parameters.
        """
        return self._credible_intervals

    def copy(self) -> Self:
        """
        Create a deep copy of this dataset.

        Returns
        -------
        Dataset
            A new dataset instance with copied data and metadata.

        Examples
        --------
        >>> dataset_copy = dataset.copy()
        >>> dataset_copy.conditions is not dataset.conditions
        True

        Notes
        -----
        This creates a deep copy of all data arrays, metadata, and
        uncertainty information. The copied dataset is completely
        independent of the original.
        """
        return copy.deepcopy(self)
