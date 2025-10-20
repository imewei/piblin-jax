# quantiq Examples

This directory contains complete, runnable examples demonstrating key features of the quantiq framework.

## Prerequisites

```bash
# Install quantiq with examples dependencies
pip install -e ".[dev]"

# For GPU examples (optional)
pip install -e ".[gpu-cuda]"  # NVIDIA GPUs
# or
pip install -e ".[gpu-metal]"  # Apple Silicon
```

## Quick Start

Each example is a standalone Python script that can be run directly:

```bash
cd examples
python basic_usage_example.py
```

## Available Examples

### 1. **basic_usage_example.py** - Getting Started
**What it demonstrates:** Core workflow for reading data, applying transforms, and visualization.

**Key concepts:**
- Creating datasets from arrays
- Building transform pipelines
- Applying transformations
- Basic visualization

**Run time:** <1 second

```bash
python basic_usage_example.py
```

---

### 2. **transform_pipeline_example.py** - Transform Pipelines
**What it demonstrates:** Composable data transformation pipelines.

**Key concepts:**
- Pipeline composition
- Sequential transforms
- Transform reuse
- Data flow through pipelines

**Run time:** <1 second

```bash
python transform_pipeline_example.py
```

---

### 3. **bayesian_rheological_models.py** - Bayesian Model Fitting
**What it demonstrates:** Fitting rheological models with full uncertainty quantification.

**Key concepts:**
- Power Law model fitting
- Cross model for shear-thinning
- Carreau-Yasuda model for complex rheology
- Arrhenius thermal activation
- MCMC sampling with NumPyro
- Posterior distributions
- Credible intervals

**Run time:** ~30 seconds (MCMC sampling)

```bash
python bayesian_rheological_models.py
```

**Output:** PNG files showing fitted models with uncertainty bands

---

### 4. **bayesian_parameter_estimation.py** - Advanced Bayesian Inference
**What it demonstrates:** Custom Bayesian models and advanced inference techniques.

**Key concepts:**
- Custom model definition
- Prior specification
- MCMC diagnostics
- Posterior predictive checks
- Model comparison

**Run time:** ~30 seconds

```bash
python bayesian_parameter_estimation.py
```

---

### 5. **uncertainty_propagation_example.py** - Uncertainty Quantification
**What it demonstrates:** Propagating uncertainty through transform pipelines.

**Key concepts:**
- Monte Carlo uncertainty propagation
- Transform composition with uncertainty
- Credible interval computation
- Visualization of uncertainty

**Run time:** ~10 seconds

```bash
python uncertainty_propagation_example.py
```

---

### 6. **piblin_migration_example.py** - piblin Compatibility
**What it demonstrates:** Migrating from piblin to quantiq.

**Key concepts:**
- Drop-in replacement (`import quantiq as piblin`)
- API compatibility
- Performance comparison
- Migration best practices

**Run time:** <1 second

```bash
python piblin_migration_example.py
```

---

### 7. **gpu_acceleration_example.py** - GPU Performance 🆕
**What it demonstrates:** Leveraging GPU acceleration with JAX.

**Key concepts:**
- Automatic GPU detection and usage
- JIT compilation
- Batch processing for GPU efficiency
- CPU vs GPU performance comparison
- Memory management on GPU

**Requirements:** CUDA, Metal, or ROCm compatible GPU

**Run time:** ~5 seconds (varies by hardware)

```bash
python gpu_acceleration_example.py
```

---

### 8. **custom_transforms_example.py** - Extending quantiq 🆕
**What it demonstrates:** Building custom transform classes.

**Key concepts:**
- Subclassing DatasetTransform
- Implementing apply() method
- Adding custom parameters
- Integration with pipelines
- JIT-compiled custom transforms

**Run time:** <1 second

```bash
python custom_transforms_example.py
```

---

## Sample Data

The `data/` directory contains sample datasets for running examples:

- **`sample_viscosity.csv`** - Rheological flow curve data (shear rate vs viscosity)
- **`sample_timeseries.csv`** - Time-dependent measurement data

These files are automatically used by the examples. You can also use your own data by modifying the file paths in the scripts.

## Directory Structure

```
examples/
├── *.py                  # 8 runnable example scripts
├── README.md             # This file
├── data/                 # Sample input data
│   ├── sample_viscosity.csv
│   └── sample_timeseries.csv
└── output/               # Generated plots (gitignored)
    └── *.png             # Output from examples
```

## Example Output

Each example produces:
- **Console output**: Numerical results, statistics, and diagnostics
- **Plots** (where applicable): PNG files saved to the `examples/output/` directory
- **Return codes**: 0 for success, non-zero for errors

The `output/` directory is automatically created when examples generate plots. This directory is gitignored to keep the repository clean.

## Performance Notes

### CPU Performance
All examples run efficiently on CPU. Expected performance on modern CPUs:
- Basic examples: <1 second
- Bayesian MCMC: 10-60 seconds
- Large pipelines: 1-5 seconds

### GPU Acceleration
GPU examples automatically detect and use available accelerators:
- **NVIDIA CUDA**: 10-100x speedup for large datasets
- **Apple Silicon (Metal)**: 5-20x speedup
- **AMD ROCm**: 10-100x speedup

If no GPU is available, examples fall back to CPU automatically.

## Customization

All examples are designed to be easily modified:

1. **Change parameters**: Adjust model parameters, transform settings, etc.
2. **Use your data**: Replace sample data paths with your own files
3. **Add transforms**: Extend pipelines with additional processing steps
4. **Modify visualization**: Customize plots for your needs

## Troubleshooting

### Common Issues

**Import errors:**
```bash
pip install -e ".[dev]"  # Install all dependencies
```

**MCMC sampling slow:**
- Reduce `n_samples` and `n_warmup` in Bayesian examples
- Or use faster models (e.g., NLSQ instead of Bayesian)

**GPU not detected:**
```python
from quantiq.backend import get_device_info
print(get_device_info())  # Check available devices
```

**Out of memory (GPU):**
- Reduce batch sizes in GPU examples
- Use CPU backend for small datasets

## Next Steps

After running the examples:

1. **Read the tutorials**: `docs/source/tutorials/`
2. **Explore the API**: `docs/source/api/`
3. **Build your own**: Start with `basic_usage_example.py` and adapt

## Contributing Examples

Have a useful example to share? Contributions are welcome!

1. Create a standalone, documented script
2. Add it to this directory
3. Update this README
4. Submit a pull request

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

---

## Example Index

| Example | Concepts | Run Time | GPU |
|---------|----------|----------|-----|
| basic_usage | Core workflow | <1s | ❌ |
| transform_pipeline | Pipelines | <1s | ❌ |
| bayesian_rheological | Bayesian fitting | ~30s | ✅ |
| bayesian_parameter | Advanced Bayesian | ~30s | ✅ |
| uncertainty_propagation | UQ | ~10s | ✅ |
| piblin_migration | Compatibility | <1s | ❌ |
| gpu_acceleration | GPU performance | ~5s | ✅ Required |
| custom_transforms | Extensibility | <1s | ✅ |

---

**Questions?** See the [main documentation](https://quantiq.readthedocs.io) or open an [issue](https://github.com/quantiq/quantiq/issues).
