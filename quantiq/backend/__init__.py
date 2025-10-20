"""
Backend abstraction layer for quantiq.

This module provides a unified interface for both JAX and NumPy backends,
with automatic fallback to NumPy when JAX is unavailable.

The backend is detected at module import time and stored in the BACKEND global variable.
All array operations should use the exported `jnp` interface which points to either
jax.numpy or numpy depending on availability.
"""

import warnings
from typing import Any, Union

import numpy as np

# Backend detection
_JAX_AVAILABLE = False
BACKEND = "numpy"  # Default to NumPy
jnp = np  # Default to NumPy

try:
    import jax
    import jax.numpy as jnp_jax

    _JAX_AVAILABLE = True
    BACKEND = "jax"
    jnp = jnp_jax

except ImportError:
    warnings.warn(
        "JAX not available, using NumPy (reduced performance). "
        "Install JAX for GPU acceleration and JIT compilation: pip install jax jaxlib",
        UserWarning,
        stacklevel=2,
    )
    _JAX_AVAILABLE = False
    BACKEND = "numpy"
    jnp = np


def is_jax_available() -> bool:
    """
    Check if JAX backend is available.

    Returns
    -------
    bool
        True if JAX is available and being used, False if using NumPy fallback.

    Examples
    --------
    >>> from quantiq.backend import is_jax_available
    >>> if is_jax_available():
    ...     print("Using JAX backend with GPU acceleration")
    ... else:
    ...     print("Using NumPy fallback")
    """
    return _JAX_AVAILABLE


def get_backend() -> str:
    """
    Get the name of the current backend.

    Returns
    -------
    str
        Either 'jax' or 'numpy' depending on which backend is in use.

    Examples
    --------
    >>> from quantiq.backend import get_backend
    >>> backend = get_backend()
    >>> print(f"Using backend: {backend}")
    """
    return BACKEND


def get_device_info() -> dict[str, Any]:
    """
    Get information about available compute devices.

    Returns
    -------
    dict
        Dictionary containing:
        - 'backend': str, name of backend ('jax' or 'numpy')
        - 'devices': list, available compute devices
        - 'default_device': str, the default device being used
        - Additional JAX-specific info if JAX is available

    Examples
    --------
    >>> from quantiq.backend import get_device_info
    >>> info = get_device_info()
    >>> print(f"Backend: {info['backend']}")
    >>> print(f"Devices: {info['devices']}")
    """
    info = {
        "backend": BACKEND,
        "devices": [],
        "default_device": "cpu",
    }

    if _JAX_AVAILABLE:
        try:
            import jax

            devices = jax.devices()
            info["devices"] = [str(d) for d in devices]
            info["default_device"] = str(jax.devices()[0])
            info["device_count"] = len(devices)

            # Add platform information using updated JAX API
            try:
                from jax.extend import backend as jax_backend

                info["platform"] = jax_backend.get_backend().platform
            except (ImportError, AttributeError):
                # Fallback for older JAX versions
                info["platform"] = str(devices[0]).split(":")[0] if devices else "cpu"

        except Exception as e:
            warnings.warn(f"Could not get JAX device info: {e}", UserWarning, stacklevel=2)
            info["devices"] = ["cpu"]
    else:
        info["devices"] = ["cpu"]
        info["platform"] = "numpy"

    return info


def to_numpy(arr: Any) -> np.ndarray:
    """
    Convert a backend array to NumPy array.

    This function handles conversion from JAX arrays to NumPy arrays,
    and passes through NumPy arrays unchanged. Useful for API boundaries
    where pure NumPy arrays are required.

    Parameters
    ----------
    arr : array_like
        Input array (JAX or NumPy array, or nested structure).

    Returns
    -------
    np.ndarray
        NumPy array with the same data.

    Examples
    --------
    >>> from quantiq.backend import jnp, to_numpy
    >>> jax_arr = jnp.array([1, 2, 3])
    >>> np_arr = to_numpy(jax_arr)
    >>> type(np_arr)
    <class 'numpy.ndarray'>
    """
    if isinstance(arr, np.ndarray):
        return arr

    if _JAX_AVAILABLE:
        # For JAX arrays, use np.asarray which handles DeviceArray conversion
        try:
            return np.asarray(arr)
        except Exception:
            # Fallback for complex types
            return np.array(arr)
    else:
        # Already using NumPy backend
        return np.asarray(arr)


def from_numpy(arr: np.ndarray) -> Any:
    """
    Convert a NumPy array to backend array.

    This function converts NumPy arrays to the current backend format
    (JAX array if JAX available, otherwise returns NumPy array unchanged).

    Parameters
    ----------
    arr : np.ndarray
        Input NumPy array.

    Returns
    -------
    array_like
        Backend array (JAX DeviceArray if JAX available, else NumPy array).

    Examples
    --------
    >>> import numpy as np
    >>> from quantiq.backend import from_numpy
    >>> np_arr = np.array([1, 2, 3])
    >>> backend_arr = from_numpy(np_arr)
    """
    if _JAX_AVAILABLE:
        return jnp.asarray(arr)
    else:
        return arr


def to_numpy_pytree(pytree: Any) -> Any:
    """
    Convert a pytree (nested structure) of arrays to NumPy.

    Handles nested dictionaries, lists, and tuples containing arrays.

    Parameters
    ----------
    pytree : Any
        Nested structure containing arrays.

    Returns
    -------
    Any
        Same structure with all arrays converted to NumPy.

    Examples
    --------
    >>> from quantiq.backend import jnp, to_numpy_pytree
    >>> pytree = {'a': jnp.array([1, 2]), 'b': [jnp.array([3, 4])]}
    >>> np_pytree = to_numpy_pytree(pytree)
    """
    if isinstance(pytree, dict):
        return {k: to_numpy_pytree(v) for k, v in pytree.items()}
    elif isinstance(pytree, (list, tuple)):
        converted = [to_numpy_pytree(item) for item in pytree]
        return type(pytree)(converted)
    elif hasattr(pytree, "__array__"):
        # Anything that looks like an array
        return to_numpy(pytree)
    else:
        return pytree


def from_numpy_pytree(pytree: Any) -> Any:
    """
    Convert a pytree (nested structure) of NumPy arrays to backend arrays.

    Handles nested dictionaries, lists, and tuples containing arrays.

    Parameters
    ----------
    pytree : Any
        Nested structure containing NumPy arrays.

    Returns
    -------
    Any
        Same structure with all arrays converted to backend format.

    Examples
    --------
    >>> import numpy as np
    >>> from quantiq.backend import from_numpy_pytree
    >>> pytree = {'a': np.array([1, 2]), 'b': [np.array([3, 4])]}
    >>> backend_pytree = from_numpy_pytree(pytree)
    """
    if isinstance(pytree, dict):
        return {k: from_numpy_pytree(v) for k, v in pytree.items()}
    elif isinstance(pytree, (list, tuple)):
        converted = [from_numpy_pytree(item) for item in pytree]
        return type(pytree)(converted)
    elif isinstance(pytree, np.ndarray):
        return from_numpy(pytree)
    else:
        return pytree


# Export public API
__all__ = [
    "BACKEND",
    "from_numpy",
    "from_numpy_pytree",
    "get_backend",
    "get_device_info",
    "is_jax_available",
    "jnp",
    "to_numpy",
    "to_numpy_pytree",
]
