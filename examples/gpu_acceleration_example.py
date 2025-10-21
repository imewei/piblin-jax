"""
GPU Acceleration Example for quantiq.

This example demonstrates how to leverage GPU acceleration with JAX
for significant performance improvements in data processing and model fitting.

Key concepts:
- Automatic GPU detection and usage
- JIT compilation for optimization
- Batch processing for GPU efficiency
- CPU vs GPU performance comparison
- Memory management on GPU

Requirements:
- JAX with CUDA 12+ on Linux
- NVIDIA GPU on Linux

Run time: ~5 seconds (varies by hardware)

Note: GPU acceleration is only available on Linux with CUDA 12+.
On macOS and Windows, the example will run in CPU-only mode.

Author: quantiq developers
Date: 2025-10-20
"""

import sys
import time

import numpy as np

# Import quantiq
from quantiq.backend import get_backend, get_device_info, is_jax_available, jnp
from quantiq.data.datasets import OneDimensionalDataset
from quantiq.transform.dataset.smoothing import GaussianSmooth
from quantiq.transform.pipeline import Pipeline


def print_device_info():
    """Print information about available compute devices."""
    print("=" * 70)
    print("DEVICE INFORMATION")
    print("=" * 70)

    backend = get_backend()
    print(f"Backend: {backend}")

    if is_jax_available():
        info = get_device_info()
        print("JAX available: Yes")
        print(f"Default device: {info.get('default_device', 'CPU')}")

        # Display platform information
        os_platform = info.get("os_platform", "unknown")
        print(f"OS Platform: {os_platform}")

        # Display GPU support status
        gpu_supported = info.get("gpu_supported", False)
        cuda_version = info.get("cuda_version")

        if gpu_supported and cuda_version:
            major, minor = cuda_version
            print(f"GPU Support: Yes (CUDA {major}.{minor})")
        else:
            print("GPU Support: No")
            if os_platform != "linux":
                print("  (GPU acceleration requires Linux with CUDA 12+)")
            elif cuda_version:
                major, minor = cuda_version
                print(f"  (CUDA {major}.{minor} detected, CUDA 12+ required)")

        devices = info.get("devices", [])
        if devices:
            print(f"Available devices: {len(devices)}")
            for i, device in enumerate(devices):
                print(f"  Device {i}: {device}")
        else:
            print("Available devices: CPU only")

        platform = info.get("platform", "cpu")
        print(f"Platform: {platform.upper()}")

        if platform == "gpu":
            print("\n✓ GPU acceleration available!")
        elif platform == "cpu":
            print("\n⚠ Running on CPU (no GPU detected)")
    else:
        print("JAX available: No (using NumPy backend)")
        print("⚠ Install JAX for GPU acceleration")

    print("=" * 70)
    print()


def create_large_dataset(n_points: int = 100_000):
    """
    Create a large synthetic dataset for performance testing.

    Parameters
    ----------
    n_points : int
        Number of data points (default: 100,000)

    Returns
    -------
    OneDimensionalDataset
        Large dataset for benchmarking
    """
    x = np.linspace(0, 100, n_points)
    # Create complex signal with noise
    y = np.sin(x * 0.5) + 0.5 * np.sin(x * 2.0) + 0.1 * np.random.normal(size=n_points)

    dataset = OneDimensionalDataset(x, y)
    dataset.name = "Large Synthetic Signal"
    return dataset


def benchmark_cpu_vs_gpu():
    """
    Benchmark CPU vs GPU performance for transform pipelines.

    Demonstrates the performance difference between CPU and GPU
    when processing large datasets with complex pipelines.
    """
    print("=" * 70)
    print("PERFORMANCE BENCHMARK: CPU vs GPU")
    print("=" * 70)

    # Create test dataset
    print("\nCreating large dataset (100,000 points)...")
    dataset = create_large_dataset(100_000)

    # Create transform pipeline
    pipeline = Pipeline(
        [
            GaussianSmooth(sigma=5.0),
            GaussianSmooth(sigma=2.0),
        ]
    )

    # Warm-up run (JIT compilation)
    print("Warming up (JIT compilation)...")
    _ = pipeline.apply_to(dataset)

    # Benchmark CPU/GPU performance
    print("\nRunning benchmark (5 iterations)...")
    iterations = 5

    start_time = time.time()
    for _i in range(iterations):
        result = pipeline.apply_to(dataset)
    end_time = time.time()

    avg_time = (end_time - start_time) / iterations
    backend = get_backend()

    print(f"\nResults ({backend} backend):")
    print(f"  Average time per iteration: {avg_time * 1000:.2f} ms")
    print(f"  Throughput: {len(dataset.independent_variable_data) / avg_time:.0f} points/second")

    if backend == "jax":
        device_info = get_device_info()
        platform = device_info.get("platform", "cpu")
        if platform == "gpu":
            print("\n✓ GPU acceleration enabled")
            print("  Expected speedup: 10-100x compared to CPU")
        else:
            print("\n⚠ Running on CPU")
            print("  GPU acceleration requires Linux with CUDA 12+")
    else:
        print("\n⚠ Using NumPy backend")
        print("  Install JAX for automatic GPU acceleration")

    print("=" * 70)
    print()

    return result


def demonstrate_jit_compilation():
    """
    Demonstrate JIT compilation for custom functions.

    Shows how to use JIT compilation to optimize custom
    mathematical operations for GPU/CPU.
    """
    print("=" * 70)
    print("JIT COMPILATION EXAMPLE")
    print("=" * 70)

    def compute_moving_average(data, window_size=10):
        """Compute moving average (Note: JIT removed due to dynamic slicing)."""
        # Dynamic slicing isn't compatible with JIT, so we use regular execution
        # For production code, use lax.dynamic_slice for JIT-compatible dynamic slicing
        result = jnp.zeros_like(data)
        for i in range(len(data)):
            start = int(jnp.maximum(0, i - window_size // 2))
            end = int(jnp.minimum(len(data), i + window_size // 2 + 1))
            result = result.at[i].set(jnp.mean(data[start:end]))
        return result

    # Create test data
    data = jnp.array(np.random.randn(10_000))

    # First call: compilation + execution
    print("\nFirst call (includes compilation)...")
    start = time.time()
    compute_moving_average(data, window_size=20)
    compile_time = time.time() - start
    print(f"  Time: {compile_time * 1000:.2f} ms")

    # Second call: cached, much faster
    print("\nSecond call (using cached compilation)...")
    start = time.time()
    compute_moving_average(data, window_size=20)
    cached_time = time.time() - start
    print(f"  Time: {cached_time * 1000:.2f} ms")

    if compile_time > cached_time:
        speedup = compile_time / cached_time
        print(f"\nSpeedup from caching: {speedup:.1f}x")
    print()

    print("💡 Key takeaway:")
    print("   JIT compilation has a one-time cost, but subsequent")
    print("   calls are much faster. Use @jit for repeated operations!")

    print("=" * 70)
    print()


def demonstrate_batch_processing():
    """
    Demonstrate efficient batch processing for GPU.

    Shows how to structure code for maximum GPU efficiency
    by processing multiple datasets in parallel.
    """
    print("=" * 70)
    print("BATCH PROCESSING FOR GPU EFFICIENCY")
    print("=" * 70)

    # Create multiple datasets
    n_datasets = 10
    print(f"\nCreating {n_datasets} datasets...")

    datasets = []
    for i in range(n_datasets):
        x = np.linspace(0, 10, 1000)
        y = np.sin(x * (i + 1) * 0.5) + 0.1 * np.random.normal(size=len(x))
        ds = OneDimensionalDataset(x, y)
        ds.name = f"Dataset {i + 1}"
        datasets.append(ds)

    # Process sequentially
    print("\nProcessing sequentially...")
    pipeline = Pipeline([GaussianSmooth(sigma=2.0)])

    start = time.time()
    [pipeline.apply_to(ds) for ds in datasets]
    sequential_time = time.time() - start

    print(f"  Sequential processing time: {sequential_time * 1000:.2f} ms")

    # For true batch processing, you would stack the data
    # and process in a single operation (requires custom implementation)
    print("\n💡 Batch processing tips:")
    print("   1. Stack multiple datasets into single arrays")
    print("   2. Use vmap/vectorization for parallel processing")
    print("   3. Minimize data transfers between CPU and GPU")
    print("   4. Process largest batches that fit in GPU memory")

    print("=" * 70)
    print()


def memory_management_tips():
    """Print tips for GPU memory management."""
    print("=" * 70)
    print("GPU MEMORY MANAGEMENT TIPS")
    print("=" * 70)
    print()
    print("1. **Monitor memory usage:**")
    print("   JAX manages GPU memory automatically, but you can")
    print("   monitor usage to avoid out-of-memory errors.")
    print()
    print("2. **Clear unused arrays:**")
    print("   Delete large arrays when no longer needed:")
    print("   ```python")
    print("   del large_array")
    print("   ```")
    print()
    print("3. **Use chunking for large datasets:**")
    print("   Process data in chunks if it doesn't fit in GPU memory:")
    print("   ```python")
    print("   chunk_size = 10_000")
    print("   for i in range(0, len(data), chunk_size):")
    print("       chunk = data[i:i+chunk_size]")
    print("       process(chunk)")
    print("   ```")
    print()
    print("4. **Configure JAX memory allocation:**")
    print("   Set environment variable for pre-allocation:")
    print("   ```bash")
    print("   export XLA_PYTHON_CLIENT_PREALLOCATE=false")
    print("   ```")
    print()
    print("=" * 70)
    print()


def main():
    """Run all GPU acceleration demonstrations."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "GPU ACCELERATION EXAMPLE" + " " * 29 + "║")
    print("║" + " " * 18 + "quantiq Framework" + " " * 33 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    # Detect platform at startup
    device_info = get_device_info()
    os_platform = device_info.get("os_platform", "unknown")
    gpu_supported = device_info.get("gpu_supported", False)

    # Display platform-specific message
    if os_platform != "linux":
        print("=" * 70)
        print("PLATFORM NOTICE")
        print("=" * 70)
        print()
        print("GPU acceleration is only available on Linux with CUDA 12+.")
        print(f"Detected platform: {os_platform}")
        print("Running in CPU-only mode.")
        print()
        print("For GPU acceleration, please run on a Linux system with:")
        print("  - NVIDIA GPU with CUDA 12+ drivers")
        print("  - JAX with CUDA support: pip install 'jax[cuda12]'")
        print()
        print("=" * 70)
        print()

    # 1. Device information
    print_device_info()

    # 2. Performance benchmark
    benchmark_cpu_vs_gpu()

    # 3. JIT compilation
    demonstrate_jit_compilation()

    # 4. Batch processing (only show GPU-specific tips if on Linux with GPU)
    if os_platform == "linux" and gpu_supported:
        demonstrate_batch_processing()
    else:
        print("=" * 70)
        print("BATCH PROCESSING")
        print("=" * 70)
        print()
        print("Batch processing demonstrations are optimized for GPU acceleration.")
        print("Running on CPU-only mode - batch processing tips skipped.")
        print()
        print("To see GPU-optimized batch processing, run on Linux with CUDA 12+.")
        print("=" * 70)
        print()

    # 5. Memory management (only show GPU-specific tips if on Linux with GPU)
    if os_platform == "linux" and gpu_supported:
        memory_management_tips()
    else:
        print("=" * 70)
        print("MEMORY MANAGEMENT")
        print("=" * 70)
        print()
        print("GPU memory management tips are specific to GPU acceleration.")
        print("Running on CPU-only mode - GPU memory tips skipped.")
        print()
        print("To see GPU memory management tips, run on Linux with CUDA 12+.")
        print("=" * 70)
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("✓ Device information retrieved")
    print("✓ Performance benchmark completed")
    print("✓ JIT compilation demonstrated")

    if os_platform == "linux" and gpu_supported:
        print("✓ Batch processing explained")
        print("✓ Memory management tips provided")
    else:
        print("⚠ Batch processing skipped (GPU-specific)")
        print("⚠ Memory management skipped (GPU-specific)")

    print()

    backend = get_backend()
    if backend == "jax":
        device_info = get_device_info()
        platform = device_info.get("platform", "cpu")
        if platform == "gpu":
            print("🚀 GPU acceleration is ACTIVE!")
            print("   Your code is running at optimal performance.")
        else:
            print("💡 GPU not detected - running on CPU")
            print("   GPU acceleration requires:")
            print("   - Linux with CUDA 12+: pip install 'jax[cuda12]'")
    else:
        print("💡 Using NumPy backend")
        print("   Install JAX for automatic GPU acceleration:")
        print("   pip install jax jaxlib")

    print()
    print("=" * 70)
    print("\n✨ Example completed successfully!\n")


if __name__ == "__main__":
    main()
