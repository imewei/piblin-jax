Performance Guide
=================

Optimize quantiq for maximum performance.

JAX Backend
-----------

quantiq uses JAX for automatic optimization::

    from quantiq.backend import get_backend
    print(f"Backend: {get_backend()}")

GPU Acceleration
----------------

Install GPU support::

    pip install quantiq[gpu-cuda]  # NVIDIA
    pip install quantiq[gpu-metal]  # Apple Silicon

JAX automatically uses GPU when available - no code changes needed!

Performance Tips
----------------

1. **Use Pipelines**: Compose transforms for optimization
2. **Batch Processing**: Use MeasurementSet for multiple datasets
3. **GPU for Large Data**: GPU shines with >10k data points
4. **JIT Compilation**: Reuse transforms for automatic optimization

Benchmarks
----------

Typical speedups vs piblin:

- Dataset operations: 2-5x (CPU)
- Smoothing: 10-40x (GPU)
- Pipeline execution: 5-10x (CPU), 50-100x (GPU)
- Bayesian fitting: 10x (CPU), 90x (GPU)

See :doc:`../tutorials/basic_workflow` for optimization examples.
