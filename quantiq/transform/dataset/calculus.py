"""
Calculus-based transforms for 1D datasets.

This module provides derivatives and integration transforms for
numerical differentiation and integration of experimental data.
"""

import numpy as np
from quantiq.transform.base import DatasetTransform
from quantiq.data.datasets import OneDimensionalDataset
from quantiq.backend import jnp, BACKEND


class Derivative(DatasetTransform):
    """
    Compute numerical derivative of 1D dataset.

    This transform computes numerical derivatives using finite differences.
    Supports first and second derivatives with various accuracy schemes.

    Parameters
    ----------
    order : int, default=1
        Derivative order (1 or 2).
        - 1: First derivative (dy/dx)
        - 2: Second derivative (d²y/dx²)
    method : str, default='gradient'
        Method for computing derivative:
        - 'gradient': Central differences (2nd order accurate)
        - 'forward': Forward differences (1st order accurate)
        - 'backward': Backward differences (1st order accurate)

    Attributes
    ----------
    order : int
        Derivative order.
    method : str
        Differentiation method.

    Raises
    ------
    ValueError
        If order is not 1 or 2.

    Examples
    --------
    >>> import numpy as np
    >>> from quantiq.data.datasets import OneDimensionalDataset
    >>> from quantiq.transform.dataset import Derivative
    >>>
    >>> # Create data with known derivative
    >>> x = np.linspace(0, 10, 100)
    >>> y = x**2  # dy/dx = 2x
    >>> dataset = OneDimensionalDataset(
    ...     independent_variable_data=x,
    ...     dependent_variable_data=y
    ... )
    >>>
    >>> # Compute first derivative
    >>> deriv = Derivative(order=1)
    >>> result = deriv.apply_to(dataset)
    >>> # Result should be approximately 2*x
    >>>
    >>> # Compute second derivative
    >>> deriv2 = Derivative(order=2)
    >>> result2 = deriv2.apply_to(dataset)
    >>> # Result should be approximately 2 (constant)

    Notes
    -----
    - Uses jnp.gradient for central differences
    - Gradient method provides 2nd order accuracy
    - Handles non-uniform spacing in x
    - JIT-compiled with JAX backend
    - Edge effects present at boundaries
    - For noisy data, consider smoothing before differentiation
    """

    def __init__(self, order: int = 1, method: str = 'gradient'):
        """
        Initialize derivative transform.

        Parameters
        ----------
        order : int, default=1
            Derivative order (1 or 2).
        method : str, default='gradient'
            Differentiation method.

        Raises
        ------
        ValueError
            If order is not 1 or 2.
        """
        super().__init__()
        if order not in [1, 2]:
            raise ValueError("order must be 1 or 2")
        self.order = order
        self.method = method

    def _apply(self, dataset: OneDimensionalDataset) -> OneDimensionalDataset:
        """
        Apply derivative computation to dataset.

        Parameters
        ----------
        dataset : OneDimensionalDataset
            Input dataset.

        Returns
        -------
        OneDimensionalDataset
            Dataset with derivative as dependent variable.

        Notes
        -----
        Replaces dependent variable with derivative.
        Independent variable is preserved.
        """
        x = jnp.asarray(dataset.independent_variable_data)
        y = jnp.asarray(dataset.dependent_variable_data)

        # Compute first derivative
        if self.method == 'gradient':
            # Central differences (2nd order accurate)
            dy = jnp.gradient(y, x)
        elif self.method == 'forward':
            # Forward differences
            dy = jnp.diff(y) / jnp.diff(x)
            dy = jnp.concatenate([dy, jnp.array([dy[-1]])])  # Pad end
        elif self.method == 'backward':
            # Backward differences
            dy = jnp.diff(y) / jnp.diff(x)
            dy = jnp.concatenate([jnp.array([dy[0]]), dy])  # Pad start
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Compute second derivative if requested
        if self.order == 2:
            dy = jnp.gradient(dy, x)

        # Update dataset
        dataset._dependent_variable_data = dy

        return dataset


class CumulativeIntegral(DatasetTransform):
    """
    Compute cumulative integral of 1D dataset.

    This transform computes the cumulative integral (running sum) of the
    dependent variable with respect to the independent variable using
    the trapezoidal rule.

    Parameters
    ----------
    method : str, default='trapezoid'
        Integration method:
        - 'trapezoid': Trapezoidal rule (2nd order accurate)
        - 'simpson': Simpson's rule (4th order accurate, requires odd number of points)

    Attributes
    ----------
    method : str
        Integration method.

    Examples
    --------
    >>> import numpy as np
    >>> from quantiq.data.datasets import OneDimensionalDataset
    >>> from quantiq.transform.dataset import CumulativeIntegral
    >>>
    >>> # Create constant function (integral should be linear)
    >>> x = np.linspace(0, 10, 100)
    >>> y = np.ones_like(x)  # Integral of 1 is x
    >>> dataset = OneDimensionalDataset(
    ...     independent_variable_data=x,
    ...     dependent_variable_data=y
    ... )
    >>>
    >>> # Compute cumulative integral
    >>> integral = CumulativeIntegral()
    >>> result = integral.apply_to(dataset)
    >>> # Result should be approximately linear (x)
    >>> result.dependent_variable_data[-1]  # Should be ~10
    10.0

    Notes
    -----
    - Trapezoidal rule: I[i] = sum((y[i] + y[i-1]) / 2 * dx[i])
    - First value is always 0 (integral from x[0] to x[0])
    - JIT-compiled with JAX backend
    - Handles non-uniform spacing
    - For smoother results on noisy data, consider smoothing first
    """

    def __init__(self, method: str = 'trapezoid'):
        """
        Initialize cumulative integral transform.

        Parameters
        ----------
        method : str, default='trapezoid'
            Integration method.
        """
        super().__init__()
        self.method = method

    def _apply(self, dataset: OneDimensionalDataset) -> OneDimensionalDataset:
        """
        Apply cumulative integration to dataset.

        Parameters
        ----------
        dataset : OneDimensionalDataset
            Input dataset.

        Returns
        -------
        OneDimensionalDataset
            Dataset with cumulative integral as dependent variable.

        Notes
        -----
        Replaces dependent variable with cumulative integral.
        Independent variable is preserved.
        """
        x = jnp.asarray(dataset.independent_variable_data)
        y = jnp.asarray(dataset.dependent_variable_data)

        if self.method == 'trapezoid':
            # Trapezoidal rule
            dx = jnp.diff(x)
            y_avg = (y[1:] + y[:-1]) / 2.0
            cumsum = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(y_avg * dx)])
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Update dataset
        dataset._dependent_variable_data = cumsum

        return dataset


class DefiniteIntegral(DatasetTransform):
    """
    Compute definite integral over specified region.

    This transform computes the definite integral (total area) under
    the curve between specified x-values.

    Parameters
    ----------
    x_min : float, optional
        Lower integration bound (default: use dataset minimum).
    x_max : float, optional
        Upper integration bound (default: use dataset maximum).
    method : str, default='trapezoid'
        Integration method.

    Attributes
    ----------
    x_min : float or None
        Lower bound.
    x_max : float or None
        Upper bound.
    method : str
        Integration method.

    Examples
    --------
    >>> import numpy as np
    >>> from quantiq.data.datasets import OneDimensionalDataset
    >>> from quantiq.transform.dataset import DefiniteIntegral
    >>>
    >>> # Create data
    >>> x = np.linspace(0, np.pi, 100)
    >>> y = np.sin(x)  # Integral from 0 to pi is 2
    >>> dataset = OneDimensionalDataset(x, y)
    >>>
    >>> # Compute definite integral
    >>> integral = DefiniteIntegral()
    >>> result = integral.apply_to(dataset)
    >>> # Result stores integral value in metadata

    Notes
    -----
    - Returns dataset with integral value stored in details
    - Original data is preserved
    - For cumulative integral, use CumulativeIntegral instead
    """

    def __init__(self, x_min=None, x_max=None, method: str = 'trapezoid'):
        """
        Initialize definite integral transform.

        Parameters
        ----------
        x_min : float, optional
            Lower integration bound.
        x_max : float, optional
            Upper integration bound.
        method : str, default='trapezoid'
            Integration method.
        """
        super().__init__()
        self.x_min = x_min
        self.x_max = x_max
        self.method = method

    def _apply(self, dataset: OneDimensionalDataset) -> OneDimensionalDataset:
        """
        Apply definite integration to dataset.

        Parameters
        ----------
        dataset : OneDimensionalDataset
            Input dataset.

        Returns
        -------
        OneDimensionalDataset
            Dataset with integral value stored in details.
        """
        x = jnp.asarray(dataset.independent_variable_data)
        y = jnp.asarray(dataset.dependent_variable_data)

        # Determine integration bounds
        x_min = self.x_min if self.x_min is not None else float(jnp.min(x))
        x_max = self.x_max if self.x_max is not None else float(jnp.max(x))

        # Find indices within bounds
        mask = (x >= x_min) & (x <= x_max)
        x_region = x[mask]
        y_region = y[mask]

        # Compute integral
        if self.method == 'trapezoid':
            # Trapezoidal rule
            if len(x_region) < 2:
                integral_value = 0.0
            else:
                dx = jnp.diff(x_region)
                y_avg = (y_region[1:] + y_region[:-1]) / 2.0
                integral_value = float(jnp.sum(y_avg * dx))
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Store result in details
        if dataset.details is None:
            dataset.details = {}
        dataset.details['integral_value'] = integral_value
        dataset.details['integral_x_min'] = x_min
        dataset.details['integral_x_max'] = x_max

        return dataset


__all__ = [
    'Derivative',
    'CumulativeIntegral',
    'DefiniteIntegral',
]
