"""
Backend-agnostic array operations.

This module provides unified interfaces for common array operations that work
with both JAX and NumPy backends. It also provides JIT compilation decorators
and device placement utilities that gracefully degrade to no-ops when using
the NumPy backend.
"""

from collections.abc import Callable, Sequence
from functools import wraps
from typing import Any, ParamSpec, TypeVar

import numpy as np

from . import _JAX_AVAILABLE, jnp

# Type variables for generic callable decorators
P = ParamSpec("P")
R = TypeVar("R")

# Array Operations


def copy(arr: Any) -> Any:
    """
    Create a copy of an array.

    Parameters
    ----------
    arr : array_like
        Input array.

    Returns
    -------
    array_like
        Copy of the input array.

    Examples
    --------
    >>> from piblin_jax.backend import jnp
    >>> from piblin_jax.backend.operations import copy
    >>> arr = jnp.array([1, 2, 3])
    >>> arr_copy = copy(arr)
    """
    if _JAX_AVAILABLE:
        # JAX arrays are immutable, so copy is just array creation
        return jnp.array(arr)
    else:
        return np.copy(arr)


def concatenate(arrays: Sequence[Any], axis: int = 0) -> Any:
    """
    Concatenate arrays along an existing axis.

    Parameters
    ----------
    arrays : sequence of array_like
        Arrays to concatenate. All arrays must have the same shape except
        in the dimension corresponding to axis.
    axis : int, optional
        Axis along which to concatenate. Default is 0.

    Returns
    -------
    array_like
        Concatenated array.

    Examples
    --------
    >>> from piblin_jax.backend import jnp
    >>> from piblin_jax.backend.operations import concatenate
    >>> arr1 = jnp.array([1, 2])
    >>> arr2 = jnp.array([3, 4])
    >>> result = concatenate([arr1, arr2])
    """
    return jnp.concatenate(arrays, axis=axis)


def stack(arrays: Sequence[Any], axis: int = 0) -> Any:
    """
    Stack arrays along a new axis.

    Parameters
    ----------
    arrays : sequence of array_like
        Arrays to stack. All arrays must have the same shape.
    axis : int, optional
        Axis along which to stack. Default is 0.

    Returns
    -------
    array_like
        Stacked array with one additional dimension.

    Examples
    --------
    >>> from piblin_jax.backend import jnp
    >>> from piblin_jax.backend.operations import stack
    >>> arr1 = jnp.array([1, 2, 3])
    >>> arr2 = jnp.array([4, 5, 6])
    >>> result = stack([arr1, arr2])
    >>> result.shape
    (2, 3)
    """
    return jnp.stack(arrays, axis=axis)


def reshape(arr: Any, shape: int | Sequence[int]) -> Any:
    """
    Reshape an array.

    Parameters
    ----------
    arr : array_like
        Input array.
    shape : int or sequence of ints
        New shape. One dimension can be -1, in which case it's inferred.

    Returns
    -------
    array_like
        Reshaped array.

    Examples
    --------
    >>> from piblin_jax.backend import jnp
    >>> from piblin_jax.backend.operations import reshape
    >>> arr = jnp.array([1, 2, 3, 4, 5, 6])
    >>> result = reshape(arr, (2, 3))
    >>> result.shape
    (2, 3)
    """
    return jnp.reshape(arr, shape)


# JIT Compilation


def jit[**P, R](
    func: Callable[P, R] | None = None, **kwargs: Any
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Just-in-time compilation decorator.

    For JAX backend, this uses jax.jit for compilation and optimization.
    For NumPy backend, this is a no-op that returns the function unchanged.

    Parameters
    ----------
    func : callable, optional
        Function to JIT compile.
    **kwargs
        Additional arguments passed to jax.jit (ignored for NumPy backend).

    Returns
    -------
    callable
        JIT-compiled function (JAX) or original function (NumPy).

    Examples
    --------
    >>> from piblin_jax.backend.operations import jit
    >>> from piblin_jax.backend import jnp
    >>>
    >>> @jit
    ... def compute(x):
    ...     return x ** 2 + 2 * x + 1
    >>>
    >>> result = compute(jnp.array([1.0, 2.0, 3.0]))
    """

    def decorator(f: Callable[P, R]) -> Callable[P, R]:
        """
        Apply JIT compilation or no-op depending on backend availability.

        Parameters
        ----------
        f : callable
            Function to compile.

        Returns
        -------
        callable
            Compiled function or wrapper.

        Examples
        --------
        >>> def my_func(x):
        ...     return x + 1
        >>> compiled = decorator(my_func)
        """
        if _JAX_AVAILABLE:
            import jax

            return jax.jit(f, **kwargs)
        else:
            # No-op for NumPy backend
            @wraps(f)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                """
                Wrapper function for NumPy backend compatibility.

                Returns
                -------
                Any
                    Result of wrapped function.

                Examples
                --------
                >>> wrapper(1, 2, x=3)  # doctest: +SKIP
                """
                return f(*args, **kwargs)

            return wrapper

    # Support both @jit and @jit(static_argnums=0) syntax
    if func is None:
        return decorator
    else:
        return decorator(func)


def vmap(
    func: Callable[..., Any],
    in_axes: int | Sequence[int | None] = 0,
    out_axes: int = 0,
    **kwargs: Any,
) -> Callable[..., Any]:
    """
    Vectorizing map decorator.

    For JAX backend, this uses jax.vmap for automatic vectorization.
    For NumPy backend, this provides a simple implementation using iteration.

    Parameters
    ----------
    func : callable
        Function to vectorize.
    in_axes : int or sequence of int/None, optional
        Axis to map over for each input. Default is 0.
    out_axes : int, optional
        Axis of output to map over. Default is 0.
    **kwargs
        Additional arguments passed to jax.vmap (ignored for NumPy backend).

    Returns
    -------
    callable
        Vectorized function.

    Examples
    --------
    >>> from piblin_jax.backend.operations import vmap
    >>> from piblin_jax.backend import jnp
    >>>
    >>> def add_one(x):
    ...     return x + 1
    >>>
    >>> batched_add = vmap(add_one)
    >>> result = batched_add(jnp.array([1, 2, 3]))
    """
    if _JAX_AVAILABLE:
        import jax

        return jax.vmap(func, in_axes=in_axes, out_axes=out_axes, **kwargs)
    else:
        # Simple NumPy implementation
        @wraps(func)
        def wrapper(*args: Any) -> Any:
            """
            Simplified vectorization wrapper for NumPy backend.

            Returns
            -------
            array_like
                Stacked results of function applied to each element.

            Examples
            --------
            >>> wrapper([1, 2, 3])  # doctest: +SKIP
            """
            # Basic implementation - map over first axis
            if not args:
                return func()

            # Handle single input case
            if len(args) == 1:
                arr = args[0]
                results = [func(arr[i]) for i in range(len(arr))]
                return np.stack(results, axis=out_axes)

            # Handle multiple inputs - this is simplified
            # Real implementation would need to handle in_axes properly
            raise NotImplementedError(
                "NumPy backend vmap with multiple inputs not fully implemented. "
                "Use JAX backend for full vmap support."
            )

        return wrapper


def grad(
    func: Callable[..., Any], argnums: int | Sequence[int] = 0, **kwargs: Any
) -> Callable[..., Any]:
    """
    Gradient computation decorator.

    For JAX backend, this uses jax.grad for automatic differentiation.
    For NumPy backend, this raises NotImplementedError.

    Parameters
    ----------
    func : callable
        Function to differentiate. Should return a scalar.
    argnums : int or sequence of int, optional
        Which arguments to differentiate with respect to. Default is 0.
    **kwargs
        Additional arguments passed to jax.grad (ignored for NumPy backend).

    Returns
    -------
    callable
        Function that computes gradients.

    Examples
    --------
    >>> from piblin_jax.backend.operations import grad
    >>> from piblin_jax.backend import jnp
    >>>
    >>> def loss(x):
    ...     return jnp.sum(x ** 2)
    >>>
    >>> grad_loss = grad(loss)
    >>> gradient = grad_loss(jnp.array([1.0, 2.0, 3.0]))
    """
    if _JAX_AVAILABLE:
        import jax

        return jax.grad(func, argnums=argnums, **kwargs)
    else:

        def not_implemented(*args: Any, **kwargs: Any) -> Any:
            """
            Raise error when automatic differentiation is unavailable.

            Returns
            -------
            None
                Never returns, always raises.

            Raises
            ------
            NotImplementedError
                Always raised when JAX is unavailable.

            Examples
            --------
            >>> not_implemented()  # doctest: +SKIP
            Traceback (most recent call last):
                ...
            NotImplementedError: Automatic differentiation requires JAX backend
            """
            raise NotImplementedError(
                "Automatic differentiation requires JAX backend. "
                "Install JAX or use numerical differentiation."
            )

        return not_implemented


# Device Management


def device_put(arr: Any, device: Any | None = None) -> Any:
    """
    Transfer array to a specific device.

    For JAX backend, this uses jax.device_put for device placement.
    For NumPy backend, this is a no-op that returns the array unchanged.

    Parameters
    ----------
    arr : array_like
        Input array.
    device : optional
        Target device (JAX device object). Ignored for NumPy backend.

    Returns
    -------
    array_like
        Array on the specified device (JAX) or original array (NumPy).

    Examples
    --------
    >>> from piblin_jax.backend.operations import device_put
    >>> from piblin_jax.backend import jnp
    >>> arr = jnp.array([1, 2, 3])
    >>> arr_on_device = device_put(arr)
    """
    if _JAX_AVAILABLE:
        import jax

        if device is None:
            return jax.device_put(arr)
        else:
            return jax.device_put(arr, device)
    else:
        return arr


def device_get(arr: Any) -> np.ndarray:
    """
    Transfer array from device to host (NumPy array).

    For JAX backend, this converts DeviceArray to NumPy array.
    For NumPy backend, this is effectively a no-op.

    Parameters
    ----------
    arr : array_like
        Input array.

    Returns
    -------
    np.ndarray
        NumPy array.

    Examples
    --------
    >>> from piblin_jax.backend.operations import device_get
    >>> from piblin_jax.backend import jnp
    >>> arr = jnp.array([1, 2, 3])
    >>> np_arr = device_get(arr)
    """
    return np.asarray(arr)


# Type Conversions


def ensure_array(arr: Any, dtype: Any | None = None) -> Any:
    """
    Ensure input is a backend array with optional dtype conversion.

    Parameters
    ----------
    arr : array_like
        Input data (array, list, scalar, etc.).
    dtype : dtype, optional
        Desired data type.

    Returns
    -------
    array_like
        Backend array.

    Examples
    --------
    >>> from piblin_jax.backend.operations import ensure_array
    >>> arr = ensure_array([1, 2, 3], dtype=float)
    """
    if dtype is None:
        return jnp.asarray(arr)
    else:
        return jnp.asarray(arr, dtype=dtype)


def astype(arr: Any, dtype: Any) -> Any:
    """
    Cast array to specified dtype.

    Parameters
    ----------
    arr : array_like
        Input array.
    dtype : dtype
        Target data type.

    Returns
    -------
    array_like
        Array cast to specified dtype.

    Examples
    --------
    >>> from piblin_jax.backend.operations import astype
    >>> from piblin_jax.backend import jnp
    >>> arr = jnp.array([1, 2, 3])
    >>> arr_float = astype(arr, jnp.float32)
    """
    return arr.astype(dtype)


# Export public API
__all__ = [
    "astype",
    "concatenate",
    # Array operations
    "copy",
    "device_get",
    # Device management
    "device_put",
    # Type conversions
    "ensure_array",
    "grad",
    # JIT and vectorization
    "jit",
    "reshape",
    "stack",
    "vmap",
]
