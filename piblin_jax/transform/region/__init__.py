"""
Region-based transforms for piblin-jax.

This module provides transforms that operate on specific regions of data:
- RegionTransform: Base class for region-based transforms
- RegionMultiplyTransform: Example concrete implementation

Region-based transforms apply transformations only within specified regions,
preserving data outside those regions. This is useful for selective processing
such as background subtraction in specific spectral ranges or local smoothing.
"""

from typing import Union

import numpy as np

from piblin_jax.backend import from_numpy, jnp, to_numpy
from piblin_jax.data.datasets import OneDimensionalDataset
from piblin_jax.data.roi import CompoundRegion, LinearRegion
from piblin_jax.transform.base import DatasetTransform


class RegionTransform(DatasetTransform):
    """
    Base class for transforms that operate on specific regions.

    RegionTransform applies a transformation only within specified region(s),
    preserving data outside the regions. This enables selective processing
    of data based on independent variable ranges.

    Subclasses should implement the _apply_to_region() method to define
    the specific transformation to apply within the region(s).

    Parameters
    ----------
    region : LinearRegion | CompoundRegion
        Region(s) to transform

    Examples
    --------
    >>> import numpy as np
    >>> from piblin_jax.data.datasets import OneDimensionalDataset
    >>> from piblin_jax.data.roi import LinearRegion
    >>> from piblin_jax.transform.region import RegionMultiplyTransform
    >>> # Create dataset
    >>> x_data = np.array([0, 1, 2, 3, 4, 5])
    >>> y_data = np.array([1, 1, 1, 1, 1, 1])
    >>> dataset = OneDimensionalDataset(
    ...     independent_variable_data=x_data,
    ...     dependent_variable_data=y_data
    ... )
    >>> # Define region and transform
    >>> region = LinearRegion(x_min=2.0, x_max=4.0)
    >>> transform = RegionMultiplyTransform(region, factor=2.0)
    >>> # Apply transform (only region [2, 4] is multiplied)
    >>> result = transform.apply_to(dataset, make_copy=True)
    >>> result.dependent_variable_data
    array([1., 1., 2., 2., 2., 1.])

    Notes
    -----
    - Currently optimized for OneDimensionalDataset
    - Data outside regions is preserved exactly
    - Transformations use NumPy arrays internally for compatibility
    - Region masks are generated from the independent variable
    """

    def __init__(self, region: LinearRegion | CompoundRegion):
        """
        Initialize RegionTransform.

        Parameters
        ----------
        region : LinearRegion | CompoundRegion
            Region(s) to transform
        """
        super().__init__()
        self.region = region

    def _apply(self, dataset: OneDimensionalDataset) -> OneDimensionalDataset:  # type: ignore[override]
        """
        Apply transform within region, preserve outside.

        This method handles the region masking logic. Subclasses should
        override _apply_to_region() instead to define the specific
        transformation to apply.

        Parameters
        ----------
        dataset : OneDimensionalDataset
            Dataset to transform

        Returns
        -------
        OneDimensionalDataset
            Transformed dataset

        Raises
        ------
        TypeError
            If dataset is not OneDimensionalDataset
        """
        if not isinstance(dataset, OneDimensionalDataset):
            raise TypeError("RegionTransform only works with OneDimensionalDataset")

        # Get data as NumPy for mask generation
        x_data = dataset.independent_variable_data
        y_data = dataset.dependent_variable_data

        # Generate mask
        mask = self.region.get_mask(x_data)

        # Extract region data
        x_region = x_data[mask]
        y_region = y_data[mask]

        # Apply transform to region (subclass implements this)
        y_region_transformed = self._apply_to_region(x_region, y_region)

        # Reconstruct full array with transformed region
        y_data_full = y_data.copy()
        y_data_full[mask] = y_region_transformed

        # Update dataset with backend arrays
        dataset._dependent_variable_data = from_numpy(y_data_full)

        return dataset

    def _apply_to_region(self, x_region: np.ndarray, y_region: np.ndarray) -> np.ndarray:
        """
        Apply transformation to region data.

        Subclasses override this method to implement specific transforms.
        This method receives only the data within the region and should
        return the transformed dependent variable.

        Parameters
        ----------
        x_region : np.ndarray
            Independent variable within region
        y_region : np.ndarray
            Dependent variable within region

        Returns
        -------
        np.ndarray
            Transformed dependent variable

        Raises
        ------
        NotImplementedError
            If subclass doesn't implement this method
        """
        raise NotImplementedError("Subclasses must implement _apply_to_region()")


class RegionMultiplyTransform(RegionTransform):
    """
    Example transform: Multiply region by a factor.

    This is a concrete implementation of RegionTransform that multiplies
    the dependent variable within the specified region(s) by a constant factor.

    Parameters
    ----------
    region : LinearRegion | CompoundRegion
        Region(s) to transform
    factor : float
        Multiplication factor

    Examples
    --------
    >>> import numpy as np
    >>> from piblin_jax.data.datasets import OneDimensionalDataset
    >>> from piblin_jax.data.roi import LinearRegion, CompoundRegion
    >>> from piblin_jax.transform.region import RegionMultiplyTransform
    >>> # Single region example
    >>> x_data = np.linspace(0, 10, 11)
    >>> y_data = np.ones(11)
    >>> dataset = OneDimensionalDataset(
    ...     independent_variable_data=x_data,
    ...     dependent_variable_data=y_data
    ... )
    >>> region = LinearRegion(x_min=3.0, x_max=7.0)
    >>> transform = RegionMultiplyTransform(region, factor=2.0)
    >>> result = transform.apply_to(dataset, make_copy=True)
    >>> # Points in [3, 7] are multiplied by 2.0, others unchanged
    >>> # Multiple disjoint regions example
    >>> region1 = LinearRegion(x_min=1.0, x_max=2.0)
    >>> region2 = LinearRegion(x_min=8.0, x_max=9.0)
    >>> compound = CompoundRegion([region1, region2])
    >>> transform = RegionMultiplyTransform(compound, factor=0.5)
    >>> result = transform.apply_to(dataset, make_copy=True)
    >>> # Points in [1, 2] OR [8, 9] are multiplied by 0.5

    Notes
    -----
    This is a simple example transform for demonstration and testing.
    More complex transforms can be implemented following the same pattern.
    """

    def __init__(self, region: LinearRegion | CompoundRegion, factor: float):
        """
        Initialize RegionMultiplyTransform.

        Parameters
        ----------
        region : LinearRegion | CompoundRegion
            Region(s) to transform
        factor : float
            Multiplication factor
        """
        super().__init__(region)
        self.factor = factor

    def _apply_to_region(self, x_region: np.ndarray, y_region: np.ndarray) -> np.ndarray:
        """
        Multiply region data by factor.

        Parameters
        ----------
        x_region : np.ndarray
            Independent variable within region (not used)
        y_region : np.ndarray
            Dependent variable within region

        Returns
        -------
        np.ndarray
            Transformed dependent variable (y_region * factor)
        """
        return y_region * self.factor


__all__ = ["RegionMultiplyTransform", "RegionTransform"]
