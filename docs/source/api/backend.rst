Backend Abstraction
===================

Overview
--------

The ``piblin_jax.backend`` module provides a unified interface for both JAX and NumPy
array operations, enabling piblin-jax to leverage JAX's performance features (JIT
compilation, GPU acceleration, automatic differentiation) while maintaining full
compatibility with NumPy-only environments.

This abstraction layer is crucial for several reasons:

- **Graceful Degradation**: The module automatically detects available backends at
  import time and falls back to NumPy when JAX is unavailable. This ensures piblin-jax
  works in any Python environment without requiring JAX as a hard dependency.

- **Unified API**: All piblin-jax code uses the ``jnp`` interface exported by this module,
  which points to either ``jax.numpy`` or ``numpy`` depending on availability. This
  enables writing backend-agnostic code that works optimally with both.

- **Performance Benefits**: When JAX is available, piblin-jax automatically benefits from:

  - **JIT Compilation**: Functions are compiled to optimized machine code for significant
    speedups, especially for repeated operations
  - **GPU/TPU Acceleration**: Computations automatically utilize available accelerators
  - **Vectorization**: Advanced automatic vectorization for batch operations
  - **Memory Efficiency**: JAX's functional approach enables better memory management

- **Device Management**: The module provides utilities for querying available compute
  devices (CPU, GPU, TPU) and their properties, enabling device-aware optimizations.

- **Conversion Utilities**: Functions for converting between JAX and NumPy arrays, as
  well as handling nested structures (pytrees), ensure seamless interoperability at
  API boundaries.

The backend abstraction is transparent to most users - you simply import and use
``piblin_jax`` and it will automatically use the best available backend. Advanced users
can query backend status and device information for performance tuning.

Quick Examples
--------------

Basic Backend Detection
^^^^^^^^^^^^^^^^^^^^^^^^

Check which backend is being used::

    from piblin_jax.backend import get_backend, is_jax_available

    # Check backend
    backend = get_backend()
    print(f"Using backend: {backend}")  # 'jax' or 'numpy'

    # Check JAX availability
    if is_jax_available():
        print("JAX available - GPU acceleration enabled")
    else:
        print("Using NumPy fallback")

Using the Unified Array Interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Write backend-agnostic code::

    from piblin_jax.backend import jnp

    # Works with both JAX and NumPy
    def compute_mean_squared(x):
        return jnp.mean(x ** 2)

    # This code runs on either backend
    import numpy as np
    data = np.array([1.0, 2.0, 3.0, 4.0])
    result = compute_mean_squared(data)

Device Information and Management
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Query available compute devices::

    from piblin_jax.backend import get_device_info

    # Get device information
    info = get_device_info()

    print(f"Backend: {info['backend']}")
    print(f"Available devices: {info['devices']}")
    print(f"Default device: {info['default_device']}")

    if 'platform' in info:
        print(f"Platform: {info['platform']}")  # cpu, gpu, tpu

    # Example output with JAX on GPU:
    # Backend: jax
    # Available devices: ['cuda:0', 'cuda:1']
    # Default device: cuda:0
    # Platform: gpu

Array Conversion Utilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Convert between backends for API boundaries::

    from piblin_jax.backend import jnp, to_numpy, from_numpy

    # Create array with current backend
    jax_array = jnp.array([1, 2, 3, 4, 5])

    # Convert to NumPy (for saving, plotting, etc.)
    numpy_array = to_numpy(jax_array)
    print(type(numpy_array))  # <class 'numpy.ndarray'>

    # Convert back to backend format
    backend_array = from_numpy(numpy_array)

    # Handle nested structures (pytrees)
    from piblin_jax.backend import to_numpy_pytree, from_numpy_pytree

    pytree = {
        'params': {'weights': jnp.array([1, 2]), 'bias': jnp.array([0.5])},
        'metrics': [jnp.array([0.95]), jnp.array([0.98])]
    }

    # Convert entire structure to NumPy
    numpy_pytree = to_numpy_pytree(pytree)

See Also
--------

- :doc:`transform` - Transforms that leverage JAX optimization
- :doc:`fitting` - Curve fitting with JAX acceleration
- `JAX Documentation <https://jax.readthedocs.io/>`_ - JAX fundamentals and API
- `JAX Device Management <https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html>`_ - Using multiple devices

API Reference
-------------

Module Contents
^^^^^^^^^^^^^^^

.. automodule:: piblin_jax.backend
   :members:
   :undoc-members:
   :show-inheritance:

Operations
----------

.. automodule:: piblin_jax.backend.operations
   :members:
   :undoc-members:
   :show-inheritance:
