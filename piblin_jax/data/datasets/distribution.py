"""
Distribution dataset for continuous probability density functions.

Used for molecular weight distributions, continuous PDFs, and other
distribution data where the probability density is a continuous function.
"""

from typing import Any

import numpy as np

from piblin_jax.backend import jnp, to_numpy

from .base import Dataset


class Distribution(Dataset):
    """
    Distribution dataset with variable data and probability density.

    This dataset type represents continuous probability density functions:
    - Molecular weight distributions (GPC/SEC)
    - Particle size distributions (continuous)
    - Statistical distributions
    - Probability density functions
    - Any continuous distribution data

    Parameters
    ----------
    variable_data : array_like
        1D array of the variable (e.g., molecular weight, particle size).
    probability_density : array_like
        1D array of probability density values corresponding to variable_data.
        Should have the same length as variable_data.
    conditions : dict[str, Any] | None, optional
        Experimental conditions.
    details : dict[str, Any] | None, optional
        Additional metadata.

    Attributes
    ----------
    variable_data : np.ndarray
        Variable data as NumPy array.
    probability_density : np.ndarray
        Probability density as NumPy array.
    conditions : dict[str, Any]
        Experimental conditions.
    details : dict[str, Any]
        Additional metadata.

    Raises
    ------
    ValueError
        If variable_data and probability_density have different shapes.

    Examples
    --------
    >>> import numpy as np
    >>> from piblin_jax.data.datasets import Distribution
    >>> # Molecular weight distribution from GPC
    >>> molecular_weight = np.linspace(1000, 100000, 500)
    >>> # Gaussian-like distribution centered at 50000
    >>> pdf = np.exp(-((molecular_weight - 50000) ** 2) / (2 * 10000 ** 2))
    >>> # Normalize so integral equals 1
    >>> pdf = pdf / np.trapz(pdf, molecular_weight)
    >>> mwd = Distribution(
    ...     variable_data=molecular_weight,
    ...     probability_density=pdf,
    ...     conditions={"polymer": "polystyrene", "solvent": "THF"},
    ...     details={"technique": "GPC", "standard": "PS"}
    ... )
    >>> mwd.variable_data.shape
    (500,)
    >>> mwd.probability_density.shape
    (500,)

    >>> # Particle size distribution
    >>> diameter = np.linspace(1, 1000, 1000)  # nm
    >>> psd = np.exp(-((np.log(diameter) - np.log(100)) ** 2) / (2 * 0.5 ** 2))
    >>> psd = psd / np.trapz(psd, diameter)
    >>> particle_dist = Distribution(
    ...     variable_data=diameter,
    ...     probability_density=psd,
    ...     conditions={"sample": "nanoparticles_Au"},
    ...     details={"units": "nm", "technique": "DLS"}
    ... )

    >>> # Custom probability distribution
    >>> x = np.linspace(-5, 5, 1000)
    >>> pdf = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
    >>> normal_dist = Distribution(
    ...     variable_data=x,
    ...     probability_density=pdf,
    ...     details={"distribution": "standard normal"}
    ... )

    Notes
    -----
    Unlike Histogram which represents discrete bins, Distribution represents
    a continuous probability density function. The probability density values
    are typically normalized such that the integral over the variable range
    equals 1, but this is not enforced by the class.

    The distinction between Distribution and OneDimensionalDataset is primarily
    semantic: Distribution emphasizes that the dependent variable represents
    a probability density, while OneDimensionalDataset is more general.
    """

    def __init__(
        self,
        variable_data: Any,
        probability_density: Any,
        conditions: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
    ):
        """
        Initialize distribution dataset.

        Parameters
        ----------
        variable_data : array_like
            1D array of variable values.
        probability_density : array_like
            1D array of probability density values.
        conditions : dict[str, Any] | None, optional
            Experimental conditions.
        details : dict[str, Any] | None, optional
            Additional metadata.

        Raises
        ------
        ValueError
            If arrays have different shapes.
        """
        super().__init__(conditions=conditions, details=details)

        # Convert to backend arrays
        self._variable_data = jnp.asarray(variable_data)
        self._probability_density = jnp.asarray(probability_density)

        # Validation: arrays must have same shape
        if self._variable_data.shape != self._probability_density.shape:
            raise ValueError(
                f"Variable and probability density arrays must have same shape. "
                f"Got variable: {self._variable_data.shape}, "
                f"probability_density: {self._probability_density.shape}"
            )

    @property
    def variable_data(self) -> np.ndarray:
        """
        Get variable data as NumPy array.

        Returns
        -------
        np.ndarray
            1D NumPy array of variable values (e.g., molecular weight,
            particle size, x-values).

        Examples
        --------
        >>> dist.variable_data
        array([1000., 1198., 1396., ..., 99604., 99802., 100000.])
        """
        return to_numpy(self._variable_data)

    @property
    def probability_density(self) -> np.ndarray:
        """
        Get probability density as NumPy array.

        Returns
        -------
        np.ndarray
            1D NumPy array of probability density values.

        Examples
        --------
        >>> dist.probability_density
        array([0.000001, 0.000002, ..., 0.000003, 0.000001])
        >>> # Check normalization (should be close to 1)
        >>> np.trapz(dist.probability_density, dist.variable_data)
        1.0000234
        """
        return to_numpy(self._probability_density)
