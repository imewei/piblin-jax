"""
Lambda transforms and dynamic parameter transforms for piblin-jax.

This module provides:
- LambdaTransform: Wraps arbitrary functions as transforms
- DynamicTransform: Base class for data-driven parameter computation
- AutoScaleTransform: Automatic data scaling to target range
- AutoBaselineTransform: Automatic baseline correction from data

Lambda transforms allow users to create custom transforms from simple functions
without defining new classes. Dynamic transforms compute parameters from the
data itself, enabling adaptive processing.
"""

from collections.abc import Callable
from typing import Any

from piblin_jax.backend import jnp
from piblin_jax.backend.operations import jit
from piblin_jax.data.datasets import OneDimensionalDataset
from piblin_jax.transform.base import DatasetTransform


class LambdaTransform(DatasetTransform):
    """
    Transform that wraps an arbitrary function.

    Allows users to create custom transforms from simple functions
    without defining new classes. Supports both simple functions that
    operate on the dependent variable only, and functions that use
    both independent and dependent variables.

    Parameters
    ----------
    func : Callable
        Function to apply to dependent variable.
        Signature: func(y_data: ndarray) -> ndarray
        Or: func(x_data: ndarray, y_data: ndarray) -> ndarray
    use_x : bool, default=False
        If True, pass both x and y to func.
        If False, pass only y to func.
    jit_compile : bool, default=True
        If True, attempt JIT compilation of the function.
        Falls back to regular function if compilation fails.

    Examples
    --------
    >>> from piblin_jax.transform import LambdaTransform
    >>> from piblin_jax.data.datasets import OneDimensionalDataset
    >>> import numpy as np
    >>>
    >>> # Simple function
    >>> transform = LambdaTransform(lambda y: y * 2.0)
    >>> dataset = OneDimensionalDataset(
    ...     independent_variable_data=np.array([1, 2, 3]),
    ...     dependent_variable_data=np.array([2, 4, 6])
    ... )
    >>> result = transform.apply_to(dataset)
    >>>
    >>> # Function using x and y
    >>> transform = LambdaTransform(
    ...     lambda x, y: y / x.max(),
    ...     use_x=True
    ... )
    >>>
    >>> # JAX-compatible function for JIT
    >>> import jax.numpy as jnp
    >>> transform = LambdaTransform(
    ...     lambda y: jnp.exp(y) * jnp.sin(y),
    ...     jit_compile=True
    ... )

    Notes
    -----
    - Functions should use jnp instead of np for JIT compilation
    - JIT compilation improves performance but requires JAX-compatible code
    - If compilation fails, the transform falls back to the uncompiled function
    - Only works with OneDimensionalDataset
    """

    def __init__(
        self,
        func: Callable[..., Any] | None = None,
        use_x: bool = False,
        jit_compile: bool = True,
        lambda_func: Callable[..., Any] | None = None,
    ):
        """
        Initialize lambda transform.

        Parameters
        ----------
        func : Callable, optional
            Function to wrap as transform (works on arrays)
        use_x : bool
            If True, pass both x and y to func
        jit_compile : bool
            If True, attempt JIT compilation
        lambda_func : Callable, optional
            Alias for func (for piblin compatibility)

        Raises
        ------
        TypeError
            If func is not callable
        ValueError
            If neither func nor lambda_func is provided

        Notes
        -----
        The function should operate on arrays, not dataset objects.
        For example: `lambda y: y * 2.0` not `lambda ds: ds.dependent_variable_data * 2.0`
        """
        super().__init__()

        # Accept either func or lambda_func (for piblin compatibility)
        if lambda_func is not None:
            func = lambda_func

        if func is None:
            raise ValueError("Either func or lambda_func must be provided")

        if not callable(func):
            raise TypeError("func must be callable")

        self.func = func
        self.use_x = use_x
        self.jit_compile = jit_compile

        # Try to JIT compile if requested
        self._compiled_func: Callable[..., Any]
        if jit_compile:
            try:
                self._compiled_func = jit(func)
            except Exception:
                # JIT compilation failed, use regular function
                self._compiled_func = func
        else:
            self._compiled_func = func

    def _apply(self, dataset: OneDimensionalDataset) -> OneDimensionalDataset:  # type: ignore[override]
        """
        Apply lambda function to dataset.

        Parameters
        ----------
        dataset : OneDimensionalDataset
            Dataset to transform

        Returns
        -------
        OneDimensionalDataset
            Transformed dataset (same object, modified in-place)

        Raises
        ------
        TypeError
            If dataset is not OneDimensionalDataset
        """
        if not isinstance(dataset, OneDimensionalDataset):
            raise TypeError("LambdaTransform only works with OneDimensionalDataset")

        # Get backend arrays
        y_data = dataset._dependent_variable_data

        # Apply function
        if self.use_x:
            x_data = dataset._independent_variable_data
            y_transformed = self._compiled_func(x_data, y_data)
        else:
            y_transformed = self._compiled_func(y_data)

        # Update dataset
        dataset._dependent_variable_data = y_transformed

        return dataset


class DynamicTransform(DatasetTransform):
    """
    Base class for transforms with data-driven parameters.

    Dynamic transforms compute parameters from the data itself, then
    apply transformations using those parameters. This enables adaptive
    processing where the transformation depends on data characteristics.

    Subclasses must implement:
    - _compute_parameters(dataset): Extract parameters from data
    - _apply_with_parameters(dataset, params): Apply transformation

    Parameters
    ----------
    None

    Examples
    --------
    >>> from piblin_jax.transform.lambda_transform import DynamicTransform
    >>> from piblin_jax.backend import jnp
    >>>
    >>> class CustomDynamicTransform(DynamicTransform):
    ...     def _compute_parameters(self, dataset):
    ...         y_data = dataset._dependent_variable_data
    ...         return {'mean': jnp.mean(y_data)}
    ...
    ...     def _apply_with_parameters(self, dataset, params):
    ...         dataset._dependent_variable_data -= params['mean']
    ...         return dataset

    Notes
    -----
    - Parameters are computed fresh each time the transform is applied
    - Caching can be implemented in subclasses if needed
    - Only works with OneDimensionalDataset
    """

    def __init__(self) -> None:
        """Initialize dynamic transform."""
        super().__init__()
        self._cached_params: dict[str, Any] | None = None

    def _apply(self, dataset: OneDimensionalDataset) -> OneDimensionalDataset:  # type: ignore[override]
        """
        Apply with dynamically computed parameters.

        Parameters
        ----------
        dataset : OneDimensionalDataset
            Dataset to transform

        Returns
        -------
        OneDimensionalDataset
            Transformed dataset
        """
        # Compute parameters from data
        params = self._compute_parameters(dataset)

        # Apply transformation
        return self._apply_with_parameters(dataset, params)

    def _compute_parameters(self, dataset: OneDimensionalDataset) -> dict[str, Any]:
        """
        Extract parameters from dataset.

        Override in subclasses to implement parameter computation.

        Parameters
        ----------
        dataset : OneDimensionalDataset
            Dataset to analyze

        Returns
        -------
        dict
            Parameters extracted from data
        """
        raise NotImplementedError("Subclasses must implement _compute_parameters")

    def _apply_with_parameters(
        self, dataset: OneDimensionalDataset, params: dict[str, Any]
    ) -> OneDimensionalDataset:
        """
        Apply transformation with parameters.

        Override in subclasses to implement transformation logic.

        Parameters
        ----------
        dataset : OneDimensionalDataset
            Dataset to transform
        params : dict
            Parameters to use for transformation

        Returns
        -------
        OneDimensionalDataset
            Transformed dataset
        """
        raise NotImplementedError("Subclasses must implement _apply_with_parameters")


class AutoScaleTransform(DynamicTransform):
    """
    Automatically scale data to specified range.

    Computes min/max from data and scales to target range.
    Useful for normalizing data to standard ranges like [0, 1] or [-1, 1].

    Parameters
    ----------
    target_min : float, default=0.0
        Target minimum value after scaling
    target_max : float, default=1.0
        Target maximum value after scaling

    Examples
    --------
    >>> from piblin_jax.transform import AutoScaleTransform
    >>> from piblin_jax.data.datasets import OneDimensionalDataset
    >>> import numpy as np
    >>>
    >>> # Scale to [0, 1]
    >>> transform = AutoScaleTransform()
    >>> dataset = OneDimensionalDataset(
    ...     independent_variable_data=np.array([1, 2, 3]),
    ...     dependent_variable_data=np.array([10, 20, 30])
    ... )
    >>> result = transform.apply_to(dataset)
    >>> # result.dependent_variable_data is now [0, 0.5, 1]
    >>>
    >>> # Scale to [-1, 1]
    >>> transform = AutoScaleTransform(target_min=-1.0, target_max=1.0)

    Notes
    -----
    - Handles constant data (where min == max) by setting all values to target_min
    - Preserves data ordering and relative differences
    - Computed parameters: 'scale' (multiplicative factor) and 'offset' (additive shift)
    """

    def __init__(self, target_min: float = 0.0, target_max: float = 1.0):
        """
        Initialize auto-scale transform.

        Parameters
        ----------
        target_min : float
            Target minimum value
        target_max : float
            Target maximum value
        """
        super().__init__()
        self.target_min = target_min
        self.target_max = target_max

    def _compute_parameters(self, dataset: OneDimensionalDataset) -> dict[str, Any]:
        """
        Compute scaling parameters from data.

        Parameters
        ----------
        dataset : OneDimensionalDataset
            Dataset to analyze

        Returns
        -------
        dict
            Dictionary with 'scale' and 'offset' parameters
        """
        y_data = dataset._dependent_variable_data

        data_min = jnp.min(y_data)
        data_max = jnp.max(y_data)

        # Avoid division by zero for constant data
        if data_max == data_min:
            scale = 0.0
            offset = self.target_min
        else:
            scale = (self.target_max - self.target_min) / (data_max - data_min)
            offset = self.target_min - data_min * scale

        return {"scale": scale, "offset": offset}

    def _apply_with_parameters(
        self, dataset: OneDimensionalDataset, params: dict[str, Any]
    ) -> OneDimensionalDataset:
        """
        Apply scaling transformation.

        Parameters
        ----------
        dataset : OneDimensionalDataset
            Dataset to transform
        params : dict
            Parameters with 'scale' and 'offset'

        Returns
        -------
        OneDimensionalDataset
            Scaled dataset
        """
        y_data = dataset._dependent_variable_data

        y_scaled = y_data * params["scale"] + params["offset"]
        dataset._dependent_variable_data = y_scaled

        return dataset


class AutoBaselineTransform(DynamicTransform):
    """
    Automatically subtract baseline computed from data.

    Computes baseline from first/last N points or minimum value,
    then subtracts it from all data. Useful for removing offsets
    and drift in experimental measurements.

    Parameters
    ----------
    n_points : int, default=10
        Number of points to use for baseline computation
        (only used for 'first' and 'last' methods)
    method : str, default='first'
        Method for computing baseline:
        - 'first': Mean of first n_points
        - 'last': Mean of last n_points
        - 'min': Minimum value in data

    Examples
    --------
    >>> from piblin_jax.transform import AutoBaselineTransform
    >>> from piblin_jax.data.datasets import OneDimensionalDataset
    >>> import numpy as np
    >>>
    >>> # Subtract baseline from first 10 points
    >>> transform = AutoBaselineTransform(n_points=10, method='first')
    >>> dataset = OneDimensionalDataset(
    ...     independent_variable_data=np.arange(100),
    ...     dependent_variable_data=np.random.randn(100) + 5.0  # offset by 5
    ... )
    >>> result = transform.apply_to(dataset)
    >>>
    >>> # Subtract minimum value
    >>> transform = AutoBaselineTransform(method='min')

    Notes
    -----
    - 'first' method: Good for time series where initial values are baseline
    - 'last' method: Good for measurements that return to baseline
    - 'min' method: Good for ensuring all values are non-negative
    - Computed parameter: 'baseline' (value to subtract)
    """

    def __init__(self, n_points: int = 10, method: str = "first"):
        """
        Initialize auto-baseline transform.

        Parameters
        ----------
        n_points : int
            Number of points for baseline
        method : str
            Baseline computation method ('first', 'last', 'min')
        """
        super().__init__()
        self.n_points = n_points
        self.method = method

    def _compute_parameters(self, dataset: OneDimensionalDataset) -> dict[str, Any]:
        """
        Compute baseline from data.

        Parameters
        ----------
        dataset : OneDimensionalDataset
            Dataset to analyze

        Returns
        -------
        dict
            Dictionary with 'baseline' parameter

        Raises
        ------
        ValueError
            If method is not 'first', 'last', or 'min'
        """
        y_data = dataset._dependent_variable_data

        if self.method == "first":
            baseline = jnp.mean(y_data[: self.n_points])
        elif self.method == "last":
            baseline = jnp.mean(y_data[-self.n_points :])
        elif self.method == "min":
            baseline = jnp.min(y_data)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return {"baseline": baseline}

    def _apply_with_parameters(
        self, dataset: OneDimensionalDataset, params: dict[str, Any]
    ) -> OneDimensionalDataset:
        """
        Subtract baseline.

        Parameters
        ----------
        dataset : OneDimensionalDataset
            Dataset to transform
        params : dict
            Parameters with 'baseline'

        Returns
        -------
        OneDimensionalDataset
            Baseline-corrected dataset
        """
        y_data = dataset._dependent_variable_data
        y_corrected = y_data - params["baseline"]
        dataset._dependent_variable_data = y_corrected
        return dataset


__all__ = [
    "AutoBaselineTransform",
    "AutoScaleTransform",
    "DynamicTransform",
    "LambdaTransform",
]
