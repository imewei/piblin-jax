"""
Histogram dataset for binned data with variable-width bins.

Used for particle size distributions, histograms, and other binned data
where bin widths may vary.
"""

from typing import Any

import numpy as np

from piblin_jax.backend import jnp, to_numpy

from .base import Dataset


class Histogram(Dataset):
    """
    Histogram dataset with bin edges and counts.

    This dataset type represents binned data with potentially variable-width bins:
    - Particle size distributions
    - Molecular weight distributions (binned)
    - Intensity histograms
    - Frequency distributions
    - Any data organized into discrete bins

    Parameters
    ----------
    bin_edges : array_like
        1D array of bin edges. For n bins, this array has n+1 elements.
        Bins are defined as [bin_edges[i], bin_edges[i+1]).
    counts : array_like
        1D array of counts or frequencies in each bin. Must have length n
        (one less than bin_edges).
    conditions : dict[str, Any] | None, optional
        Experimental conditions.
    details : dict[str, Any] | None, optional
        Additional metadata.

    Attributes
    ----------
    bin_edges : np.ndarray
        Bin edges as NumPy array.
    counts : np.ndarray
        Counts per bin as NumPy array.
    conditions : dict[str, Any]
        Experimental conditions.
    details : dict[str, Any]
        Additional metadata.

    Raises
    ------
    ValueError
        If counts length is not compatible with bin_edges (must be len(bin_edges) - 1).

    Examples
    --------
    >>> import numpy as np
    >>> from piblin_jax.data.datasets import Histogram
    >>> # Particle size distribution with variable-width bins
    >>> bin_edges = np.array([0, 1, 3, 6, 10, 20])  # 5 bins
    >>> counts = np.array([12, 45, 67, 34, 8])      # 5 counts
    >>> psd = Histogram(
    ...     bin_edges=bin_edges,
    ...     counts=counts,
    ...     conditions={"sample": "nanoparticles_batch_42"},
    ...     details={"units": "nm", "technique": "DLS"}
    ... )
    >>> psd.bin_edges
    array([0, 1, 3, 6, 10, 20])
    >>> psd.counts
    array([12, 45, 67, 34, 8])

    >>> # Intensity histogram
    >>> # Equal-width bins for pixel intensities
    >>> bins = np.linspace(0, 255, 256)  # 255 bins
    >>> hist_counts = np.random.poisson(100, 255)
    >>> intensity_hist = Histogram(
    ...     bin_edges=bins,
    ...     counts=hist_counts,
    ...     conditions={"image": "sample_001.tif"},
    ...     details={"bit_depth": 8}
    ... )

    Notes
    -----
    Unlike Distribution, Histogram represents discrete bins rather than a
    continuous probability density. The bin_edges array has one more element
    than the counts array. For variable-width bins, the bin width can be
    computed as np.diff(bin_edges).
    """

    def __init__(
        self,
        bin_edges: Any,
        counts: Any,
        conditions: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
    ):
        """
        Initialize histogram dataset.

        Parameters
        ----------
        bin_edges : array_like
            1D array of bin edges (n+1 elements for n bins).
        counts : array_like
            1D array of counts (n elements).
        conditions : dict[str, Any] | None, optional
            Experimental conditions.
        details : dict[str, Any] | None, optional
            Additional metadata.

        Raises
        ------
        ValueError
            If counts length is not compatible with bin_edges.
        """
        super().__init__(conditions=conditions, details=details)

        # Convert to backend arrays
        self._bin_edges = jnp.asarray(bin_edges)
        self._counts = jnp.asarray(counts)

        # Validation: counts must have length (len(bin_edges) - 1)
        expected_n_bins = self._bin_edges.shape[0] - 1

        if self._counts.shape[0] != expected_n_bins:
            raise ValueError(
                f"Number of bins mismatch. For {self._bin_edges.shape[0]} bin edges, "
                f"expected {expected_n_bins} bins (counts), but got {self._counts.shape[0]}"
            )

    @property
    def bin_edges(self) -> np.ndarray:
        """
        Get bin edges as NumPy array.

        Returns
        -------
        np.ndarray
            1D NumPy array of bin edges. For n bins, has n+1 elements.

        Examples
        --------
        >>> hist.bin_edges
        array([0, 1, 3, 6, 10, 20])
        >>> # Bin widths can be computed as:
        >>> np.diff(hist.bin_edges)
        array([1, 2, 3, 4, 10])
        """
        return to_numpy(self._bin_edges)

    @property
    def counts(self) -> np.ndarray:
        """
        Get bin counts as NumPy array.

        Returns
        -------
        np.ndarray
            1D NumPy array of counts or frequencies in each bin.

        Examples
        --------
        >>> hist.counts
        array([12, 45, 67, 34, 8])
        >>> # Total count
        >>> hist.counts.sum()
        166
        """
        return to_numpy(self._counts)
