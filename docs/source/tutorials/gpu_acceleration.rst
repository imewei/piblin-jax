GPU Acceleration Best Practices
=================================

This tutorial covers best practices for leveraging GPU acceleration in quantiq
using JAX's automatic device placement and JIT compilation.

.. contents:: Table of Contents
   :local:
   :depth: 2

Prerequisites
-------------

- :doc:`basic_workflow` - Basic quantiq usage
- JAX with GPU support installed:

  - **NVIDIA CUDA**: ``pip install 'jax[cuda12]'``
  - **Apple Silicon**: ``pip install 'jax[metal]'``
  - **AMD ROCm**: ``pip install 'jax[rocm]'``

- Basic understanding of GPU computing concepts

Overview
--------

quantiq leverages JAX's automatic GPU acceleration to deliver dramatic speedups
(10-100x) for large datasets and compute-intensive operations. This tutorial shows
you how to maximize GPU performance in your workflows.

**Key benefits:**

- **Automatic device placement** - No manual GPU management
- **JIT compilation** - Functions compiled once, executed fast
- **Batch processing** - Efficient parallel computation
- **Transparent fallback** - Works on CPU when GPU unavailable

Checking GPU Availability
--------------------------

First, verify GPU access:

.. code-block:: python

    from quantiq.backend import get_backend, get_device_info, is_jax_available

    # Check backend
    backend = get_backend()
    print(f"Backend: {backend}")  # 'jax' or 'numpy'

    # Check device info
    if is_jax_available():
        info = get_device_info()
        print(f"Platform: {info['platform']}")  # 'cpu', 'gpu', or 'tpu'
        print(f"Devices: {info['devices']}")

        if info['platform'] == 'gpu':
            print("✓ GPU acceleration available!")
        else:
            print("⚠ No GPU detected, using CPU")
    else:
        print("⚠ JAX not installed, using NumPy backend")

**Expected output with GPU:**

.. code-block:: text

    Backend: jax
    Platform: gpu
    Devices: ['cuda:0']
    ✓ GPU acceleration available!

Understanding Performance Characteristics
------------------------------------------

CPU vs GPU Trade-offs
^^^^^^^^^^^^^^^^^^^^^

GPUs excel at different workloads than CPUs:

**GPU Advantages:**

- Parallel operations on large arrays (>10,000 elements)
- Matrix operations (transforms, smoothing)
- Repeated operations (MCMC sampling, batch processing)
- Vectorized computations

**CPU Advantages:**

- Small datasets (<1,000 elements)
- Sequential operations
- Complex control flow
- Single operations (no repetition)

**Rule of Thumb:**

.. code-block:: python

    if dataset_size > 10_000 or repeated_operations:
        # Use GPU for significant speedup
        pass
    else:
        # CPU is fine, GPU overhead not worth it
        pass

JIT Compilation
---------------

Basic JIT Usage
^^^^^^^^^^^^^^^

JIT compilation provides automatic optimization:

.. code-block:: python

    from quantiq.backend.operations import jit
    from quantiq.backend import jnp

    # Decorate functions for JIT compilation
    @jit
    def compute_gradient(x):
        """Compute gradient with JIT compilation."""
        return jnp.gradient(x)

    # First call: compiles + executes (~100ms)
    result1 = compute_gradient(data)

    # Subsequent calls: uses cached compilation (~1ms)
    result2 = compute_gradient(data)  # Much faster!

**Performance Tips:**

1. **JIT functions you'll call repeatedly**
2. **First call has compilation overhead** - that's normal
3. **Compiled functions are cached** - reused automatically
4. **Works on both CPU and GPU** - same code, automatic optimization

When to Use JIT
^^^^^^^^^^^^^^^

.. code-block:: python

    # ✓ Good candidates for JIT
    @jit
    def heavy_computation(x):
        """Complex mathematical operation - JIT improves performance."""
        return jnp.sum(jnp.exp(x) * jnp.sin(x) ** 2)

    @jit
    def matrix_operation(x):
        """Matrix ops benefit from JIT."""
        return jnp.dot(x.T, x)

    # ✗ Poor candidates for JIT
    def simple_operation(x):
        """Too simple - JIT overhead not worth it."""
        return x + 1

    def data_dependent_control(x):
        """Data-dependent control flow - harder to compile."""
        if jnp.mean(x) > 0:  # Avoid this pattern
            return x * 2
        else:
            return x / 2

Batch Processing for GPU Efficiency
------------------------------------

Processing Multiple Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GPUs excel at batch operations:

.. code-block:: python

    from quantiq.transform import Pipeline
    from quantiq.transform.dataset import GaussianSmoothing

    # Instead of sequential processing
    results = []
    for dataset in datasets:  # Slow on GPU
        result = pipeline.apply(dataset)
        results.append(result)

    # Better: Process in batches
    # Stack datasets into single array
    stacked_data = jnp.stack([ds.y for ds in datasets])

    # Apply transform to entire batch at once
    @jit
    def batch_smooth(data_batch):
        # Process all datasets in parallel
        return gaussian_filter(data_batch, sigma=2.0, axis=1)

    smoothed_batch = batch_smooth(stacked_data)

    # Unstack results
    results = [OneDimensionalDataset(ds.x, y)
               for ds, y in zip(datasets, smoothed_batch)]

Vectorization with vmap
^^^^^^^^^^^^^^^^^^^^^^^^

Use ``vmap`` for automatic vectorization:

.. code-block:: python

    from quantiq.backend.operations import vmap

    def process_single(x):
        """Process a single 1D array."""
        return jnp.cumsum(x) / jnp.arange(1, len(x) + 1)

    # Vectorize across batch dimension
    process_batch = vmap(process_single)

    # Now process entire batch in parallel
    batch_data = jnp.stack([dataset.y for dataset in datasets])
    results = process_batch(batch_data)  # Parallel on GPU!

Memory Management
-----------------

GPU Memory Constraints
^^^^^^^^^^^^^^^^^^^^^^

GPUs have limited memory compared to CPU RAM:

.. code-block:: python

    # ✗ Bad: May run out of GPU memory
    huge_dataset = create_dataset(size=100_000_000)  # 100M points
    result = pipeline.apply(huge_dataset)  # OOM error!

    # ✓ Good: Process in chunks
    chunk_size = 1_000_000
    results = []

    for i in range(0, len(huge_dataset.x), chunk_size):
        chunk = create_chunk(huge_dataset, i, i + chunk_size)
        result = pipeline.apply(chunk)
        results.append(result)

    # Combine results
    final_result = combine_chunks(results)

Monitoring Memory Usage
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import jax

    # For CUDA GPUs
    if jax.devices()[0].platform == 'gpu':
        # JAX manages memory automatically, but you can monitor:
        print("JAX will use GPU memory as needed")
        print("Set XLA_PYTHON_CLIENT_PREALLOCATE=false to disable preallocation")

    # Best practice: Delete large arrays when done
    large_array = jnp.zeros((10000, 10000))
    result = process(large_array)
    del large_array  # Free memory

Optimizing Transform Pipelines
-------------------------------

Pipeline-Level Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from quantiq.transform import Pipeline
    from quantiq.transform.dataset import (
        GaussianSmoothing,
        MinMaxNormalization,
        Derivative
    )

    # Create pipeline
    pipeline = Pipeline([
        GaussianSmoothing(sigma=2.0),  # GPU-optimized
        Derivative(order=1),           # GPU-optimized
        MinMaxNormalization()          # GPU-optimized
    ])

    # Warm-up: Trigger JIT compilation
    _ = pipeline.apply(sample_dataset)

    # Now process many datasets efficiently
    for dataset in large_dataset_collection:
        result = pipeline.apply(dataset)  # Fast!

Custom GPU-Optimized Transforms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create transforms that leverage GPU:

.. code-block:: python

    from quantiq.transform.base import DatasetTransform
    from quantiq.backend.operations import jit

    class GPUOptimizedTransform(DatasetTransform):
        """Transform optimized for GPU execution."""

        @staticmethod
        @jit  # JIT compile for GPU
        def _compute(y, param):
            """GPU-accelerated computation."""
            # JAX operations automatically use GPU
            return jnp.fft.fft(y * param).real

        def apply(self, dataset):
            """Apply transform."""
            result_y = self._compute(dataset.y, self.param)
            return OneDimensionalDataset(dataset.x, result_y)

Performance Benchmarking
------------------------

Measuring GPU Speedup
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import time
    from quantiq.backend import get_device_info

    def benchmark_pipeline(pipeline, dataset, n_iterations=10):
        """Benchmark pipeline performance."""
        # Warm-up
        _ = pipeline.apply(dataset)

        # Benchmark
        start = time.time()
        for _ in range(n_iterations):
            result = pipeline.apply(dataset)
        end = time.time()

        avg_time = (end - start) / n_iterations
        device = get_device_info()['platform']

        print(f"Device: {device}")
        print(f"Average time: {avg_time*1000:.2f} ms")
        print(f"Throughput: {len(dataset.x)/avg_time:.0f} points/second")

        return avg_time

    # Compare CPU vs GPU
    # (run this twice: once with CPU, once with GPU JAX)
    pipeline = Pipeline([GaussianSmoothing(sigma=2.0)])
    dataset = create_large_dataset(100_000)

    cpu_time = benchmark_pipeline(pipeline, dataset)

    # With GPU:
    # gpu_time = benchmark_pipeline(pipeline, dataset)
    # speedup = cpu_time / gpu_time
    # print(f"GPU Speedup: {speedup:.1f}x")

MCMC/Bayesian Acceleration
---------------------------

Bayesian models benefit enormously from GPU:

.. code-block:: python

    from quantiq.bayesian import PowerLawModel

    # Create model (automatically uses GPU if available)
    model = PowerLawModel(
        n_samples=5000,  # More samples with GPU
        n_warmup=2000,
        n_chains=4       # Parallel chains on GPU
    )

    # Fit model - GPU provides 10-100x speedup
    model.fit(shear_rate, viscosity)

    # Expected performance:
    # CPU: ~60 seconds
    # GPU: ~2-5 seconds (10-30x faster)

**GPU MCMC Tips:**

1. **Use more samples** - GPU makes large sample sizes feasible
2. **Run multiple chains** - Parallel chains improve convergence diagnostics
3. **Batch predictions** - Get posterior predictive for many x values at once

Common Issues and Solutions
----------------------------

Issue: GPU Not Detected
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Symptom: JAX reports 'cpu' instead of 'gpu'

    # Solution 1: Verify JAX GPU installation
    import jax
    print(jax.devices())  # Should show GPU devices

    # Solution 2: Check CUDA/drivers (NVIDIA)
    # Run: nvidia-smi (command line)

    # Solution 3: Reinstall JAX with GPU support
    # pip uninstall jax jaxlib
    # pip install --upgrade "jax[cuda12]"  # or cuda11, metal, rocm

Issue: Out of Memory Errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Symptom: "Out of memory" or "XLA allocation failed"

    # Solution 1: Reduce batch size
    chunk_size = 10_000  # Instead of 100_000

    # Solution 2: Use smaller data types
    data = data.astype(jnp.float32)  # Instead of float64

    # Solution 3: Clear memory between operations
    del large_intermediate_array

    # Solution 4: Disable preallocation
    # Set environment variable:
    # export XLA_PYTHON_CLIENT_PREALLOCATE=false

Issue: Slow First Execution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Symptom: First call to JIT function is very slow

    # This is normal! JIT compilation happens on first call.
    # Subsequent calls use cached compiled version.

    # Solution: Warm up your functions
    @jit
    def my_function(x):
        return jnp.sum(x ** 2)

    # Warm-up call (compile)
    _ = my_function(jnp.array([1, 2, 3]))

    # Now fast for all subsequent calls
    result = my_function(my_data)  # Fast!

Best Practices Summary
----------------------

1. **Use GPU for large datasets** (>10,000 elements) and repeated operations
2. **Apply JIT to performance-critical functions** - first call compiles, subsequent calls are fast
3. **Process in batches** - stack datasets and process together
4. **Use vmap for vectorization** - automatic parallelization
5. **Monitor memory** - chunk large datasets, delete unused arrays
6. **Warm up pipelines** - run once before benchmarking
7. **Leverage Bayesian GPU acceleration** - massive speedup for MCMC

Performance Comparison Table
-----------------------------

Expected speedups (GPU vs CPU):

===============================  ============  ===============
Operation                        Dataset Size  GPU Speedup
===============================  ============  ===============
Gaussian smoothing               10K points    5-10x
Gaussian smoothing               100K points   20-50x
Transform pipeline (3 steps)     100K points   30-70x
Bayesian MCMC (2K samples)       50 points     10-30x
Bayesian MCMC (10K samples)      50 points     50-100x
Batch processing (100 datasets)  10K each      40-80x
===============================  ============  ===============

Next Steps
----------

- See the ``examples/gpu_acceleration_example.py`` file in the repository for complete runnable code
- Explore :doc:`advanced_pipelines` for complex workflows
- Read :doc:`uncertainty_quantification` for Bayesian GPU usage

.. seealso::

   - :doc:`basic_workflow` - Getting started with quantiq
   - :doc:`custom_transforms` - Creating GPU-optimized transforms
   - `JAX Documentation <https://jax.readthedocs.io/>`_ - Deep dive into JAX
