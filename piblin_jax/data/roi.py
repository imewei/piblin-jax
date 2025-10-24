"""
Region of Interest (ROI) classes for piblin-jax.

This module provides classes for defining regions on independent variables:
- LinearRegion: Contiguous region on a 1D independent variable
- CompoundRegion: Container for multiple LinearRegion objects (union)

Regions are used with RegionTransform to apply transformations only
within specified regions while preserving data outside those regions.
"""

import numpy as np


class LinearRegion:
    """
    Represents a contiguous region on a 1D independent variable.

    A LinearRegion defines a contiguous range [x_min, x_max] (inclusive)
    on an independent variable. It can generate boolean masks to select
    data points within this range.

    Parameters
    ----------
    x_min : float
        Lower bound (inclusive)
    x_max : float
        Upper bound (inclusive)

    Raises
    ------
    ValueError
        If x_min >= x_max

    Examples
    --------
    >>> import numpy as np
    >>> from piblin_jax.data.roi import LinearRegion
    >>> # Define region from 2.0 to 5.0
    >>> region = LinearRegion(x_min=2.0, x_max=5.0)
    >>> # Generate mask for data
    >>> x_data = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    >>> mask = region.get_mask(x_data)
    >>> mask
    array([False, False,  True,  True,  True,  True, False, False])
    >>> # Extract data within region
    >>> x_data[mask]
    array([2, 3, 4, 5])

    Notes
    -----
    - Bounds are inclusive: both x_min and x_max are included in the region
    - Masks are generated using NumPy arrays for compatibility
    - Use with RegionTransform to apply selective transformations
    """

    def __init__(self, x_min: float, x_max: float):
        """
        Initialize LinearRegion.

        Parameters
        ----------
        x_min : float
            Lower bound (inclusive)
        x_max : float
            Upper bound (inclusive)

        Raises
        ------
        ValueError
            If x_min >= x_max
        """
        if x_min >= x_max:
            raise ValueError(f"x_min ({x_min}) must be less than x_max ({x_max})")

        self.x_min = x_min
        self.x_max = x_max

    def get_mask(self, x_data: np.ndarray) -> np.ndarray:
        """
        Generate boolean mask for data within region.

        Creates a boolean array where True indicates points within
        the region [x_min, x_max] (inclusive).

        Parameters
        ----------
        x_data : np.ndarray
            Independent variable data

        Returns
        -------
        np.ndarray
            Boolean mask (True for points in region)

        Examples
        --------
        >>> region = LinearRegion(x_min=2.0, x_max=5.0)
        >>> x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        >>> region.get_mask(x_data)
        array([False,  True,  True,  True,  True, False])
        """
        return (x_data >= self.x_min) & (x_data <= self.x_max)

    def __repr__(self) -> str:
        """Return string representation of LinearRegion."""
        return f"LinearRegion(x_min={self.x_min}, x_max={self.x_max})"


class CompoundRegion:
    """
    Container for multiple LinearRegion objects (union of regions).

    A CompoundRegion represents the union of multiple disjoint or
    overlapping LinearRegion objects. It generates combined masks
    that include all points in any of the constituent regions.

    Parameters
    ----------
    regions : list[LinearRegion]
        List of LinearRegion objects

    Raises
    ------
    ValueError
        If regions list is empty
    TypeError
        If any element is not a LinearRegion

    Examples
    --------
    >>> import numpy as np
    >>> from piblin_jax.data.roi import LinearRegion, CompoundRegion
    >>> # Define two disjoint regions
    >>> region1 = LinearRegion(x_min=1.0, x_max=2.0)
    >>> region2 = LinearRegion(x_min=4.0, x_max=5.0)
    >>> compound = CompoundRegion([region1, region2])
    >>> # Generate combined mask
    >>> x_data = np.array([0, 1, 2, 3, 4, 5, 6])
    >>> mask = compound.get_mask(x_data)
    >>> mask
    array([False,  True,  True, False,  True,  True, False])
    >>> # Extract data from both regions
    >>> x_data[mask]
    array([1, 2, 4, 5])

    Notes
    -----
    - The mask is the union (OR) of all constituent region masks
    - Regions can be disjoint or overlapping
    - Access individual regions using indexing: compound[0], compound[1], etc.
    - Get number of regions using len(compound)
    """

    def __init__(self, regions: list[LinearRegion]):
        """
        Initialize CompoundRegion.

        Parameters
        ----------
        regions : list[LinearRegion]
            List of LinearRegion objects

        Raises
        ------
        ValueError
            If regions list is empty
        TypeError
            If any element is not a LinearRegion
        """
        if not regions:
            raise ValueError("CompoundRegion requires at least one region")

        if not all(isinstance(r, LinearRegion) for r in regions):
            raise TypeError("All regions must be LinearRegion objects")

        self.regions = list(regions)

    def get_mask(self, x_data: np.ndarray) -> np.ndarray:
        """
        Generate combined boolean mask (union of all regions).

        Creates a boolean array where True indicates points within
        any of the constituent regions.

        Parameters
        ----------
        x_data : np.ndarray
            Independent variable data

        Returns
        -------
        np.ndarray
            Boolean mask (True for points in any region)

        Examples
        --------
        >>> region1 = LinearRegion(x_min=1.0, x_max=2.0)
        >>> region2 = LinearRegion(x_min=4.0, x_max=5.0)
        >>> compound = CompoundRegion([region1, region2])
        >>> x_data = np.array([0, 1, 2, 3, 4, 5, 6])
        >>> compound.get_mask(x_data)
        array([False,  True,  True, False,  True,  True, False])
        """
        combined_mask = np.zeros_like(x_data, dtype=bool)
        for region in self.regions:
            combined_mask |= region.get_mask(x_data)
        return combined_mask

    def __len__(self) -> int:
        """Return number of regions."""
        return len(self.regions)

    def __getitem__(self, index: int) -> LinearRegion:
        """Get region by index."""
        return self.regions[index]

    def __repr__(self) -> str:
        """Return string representation of CompoundRegion."""
        return f"CompoundRegion({len(self.regions)} regions)"


__all__ = ["CompoundRegion", "LinearRegion"]
