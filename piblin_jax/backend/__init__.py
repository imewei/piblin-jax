"""
Backend abstraction layer for piblin-jax.

This module provides a unified interface for both JAX and NumPy backends,
with automatic fallback to NumPy when JAX is unavailable.

The backend is detected at module import time and stored in the BACKEND global variable.
All array operations should use the exported `jnp` interface which points to either
jax.numpy or numpy depending on availability.
"""

import sys
import types
import warnings
from typing import Any, Union

import numpy as np

# Backend detection
_JAX_AVAILABLE = False
BACKEND = "numpy"  # Default to NumPy
jnp: types.ModuleType = np  # Default to NumPy


def _detect_platform() -> str:
    """
    Detect the current operating system platform.

    Returns
    -------
    str
        One of 'linux', 'macos', or 'windows'.
    """
    platform = sys.platform.lower()
    if platform.startswith("linux"):
        return "linux"
    elif platform == "darwin":
        return "macos"
    elif platform.startswith("win"):
        return "windows"
    else:
        # Default to the actual platform string for unknown platforms
        return platform


def _get_cuda_version() -> tuple[int, int] | None:
    """
    Get CUDA version from JAX backend.

    Returns
    -------
    tuple of (int, int) or None
        Tuple of (major, minor) version numbers, or None if CUDA unavailable.
    """
    try:
        import jax

        # Try newer JAX API first (v0.8.0+)
        try:
            from jax.extend import backend as jax_backend

            backend = jax_backend.get_backend()
        except (ImportError, AttributeError):
            # Fallback to older API for JAX < 0.8.0
            backend = jax.lib.xla_bridge.get_backend()

        version_string = backend.platform_version

        # Parse version string (e.g., "12.0", "11.8", "12.3.1")
        parts = version_string.split(".")
        if len(parts) >= 2:
            major = int(parts[0])
            minor = int(parts[1])
            return (major, minor)
        return None
    except Exception:
        # CUDA not available or error accessing version
        return None


def _validate_cuda_version(cuda_version: tuple[int, int] | None) -> bool:
    """
    Validate that CUDA version meets minimum requirements.

    Parameters
    ----------
    cuda_version : tuple of (int, int) or None
        CUDA version tuple (major, minor).

    Returns
    -------
    bool
        True if CUDA version >= 12.0, False otherwise.
    """
    if cuda_version is None:
        return False
    major, _ = cuda_version
    return major >= 12


def _check_legacy_gpu_extras() -> None:
    """
    Check for legacy GPU extras (gpu-metal, gpu-rocm) and issue deprecation warning.

    This function attempts to detect if the user has installed deprecated GPU extras
    and warns them to migrate to gpu-cuda on Linux.
    """
    try:
        # Check if JAX was installed with Metal or ROCm backend
        import jax

        # Try to detect Metal backend (macOS)
        try:
            devices = jax.devices()
            for device in devices:
                device_str = str(device).lower()
                if "metal" in device_str or "gpu" in device_str:
                    platform = _detect_platform()
                    if platform == "macos":
                        warnings.warn(
                            "Detected JAX with Metal backend. gpu-metal is deprecated. "
                            "GPU support is now only available on Linux with CUDA 12+. "
                            "On macOS, CPU-only mode is recommended.",
                            DeprecationWarning,
                            stacklevel=2,
                        )
                        return
        except Exception:  # nosec B110  # Intentional: silently ignore detection errors
            pass

        # Try to detect ROCm backend (AMD GPUs)
        try:
            devices = jax.devices()
            for device in devices:
                device_str = str(device).lower()
                if "rocm" in device_str or "amd" in device_str:
                    warnings.warn(
                        "Detected JAX with ROCm backend. gpu-rocm is deprecated. "
                        "GPU support is now only available on Linux with CUDA 12+.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    return
        except Exception:  # nosec B110  # Intentional: silently ignore detection errors
            pass

    except ImportError:
        # JAX not installed, no legacy extras to check
        pass


try:
    import jax
    import jax.numpy as jnp_jax

    _JAX_AVAILABLE = True
    BACKEND = "jax"
    jnp = jnp_jax

    # Check for legacy GPU extras and warn if detected
    _check_legacy_gpu_extras()

    # Platform validation
    detected_platform = _detect_platform()

    if detected_platform == "linux":
        # On Linux, validate CUDA version
        cuda_version = _get_cuda_version()
        if not _validate_cuda_version(cuda_version):
            warnings.warn(
                "GPU acceleration requires CUDA 12+. Using JAX in CPU mode.",
                UserWarning,
                stacklevel=2,
            )
            # Keep JAX available in CPU mode - don't disable it
    else:
        # Non-Linux platforms: JAX runs in CPU mode (GPU unavailable)
        warnings.warn(
            "GPU acceleration is only available on Linux with CUDA 12+. Using JAX in CPU mode.",
            UserWarning,
            stacklevel=2,
        )
        # Keep JAX available in CPU mode - don't disable it

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
    >>> from piblin_jax.backend import is_jax_available
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
    >>> from piblin_jax.backend import get_backend
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
        - 'os_platform': str, detected OS platform ('linux', 'macos', 'windows')
        - 'gpu_supported': bool, whether GPU is supported on current platform
        - 'cuda_version': tuple or None, CUDA version (major, minor) if available
        - Additional JAX-specific info if JAX is available

    Examples
    --------
    >>> from piblin_jax.backend import get_device_info
    >>> info = get_device_info()
    >>> print(f"Backend: {info['backend']}")
    >>> print(f"Devices: {info['devices']}")
    """
    info = {
        "backend": BACKEND,
        "devices": [],
        "default_device": "cpu",
        "os_platform": _detect_platform(),
        "gpu_supported": False,
        "cuda_version": None,
    }

    # Detect CUDA version if on Linux
    if info["os_platform"] == "linux":
        cuda_version = _get_cuda_version()
        info["cuda_version"] = cuda_version
        info["gpu_supported"] = _validate_cuda_version(cuda_version)

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
    >>> from piblin_jax.backend import jnp, to_numpy
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
    >>> from piblin_jax.backend import from_numpy
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
    >>> from piblin_jax.backend import jnp, to_numpy_pytree
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
    >>> from piblin_jax.backend import from_numpy_pytree
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
