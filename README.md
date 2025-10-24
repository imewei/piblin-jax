# piblin-jax

**Modern JAX-Powered Framework for Measurement Data Science**

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

---

> **Acknowledgement**: piblin-jax is an enhanced fork of [piblin](https://github.com/3mcloud/piblin) by 3M. We gratefully acknowledge the original piblin project for establishing the foundational concepts and API design that piblin-jax builds upon. piblin-jax extends piblin with JAX-powered performance improvements, advanced Bayesian inference capabilities, and modern Python features while maintaining backward compatibility.

---

## Overview

**piblin-jax** is a high-performance framework for measurement data science, providing a complete reimplementation of piblin with dramatic performance improvements and advanced Bayesian uncertainty quantification capabilities.

Built on JAX, piblin-jax delivers 5-10x CPU speedup and 50-100x GPU acceleration while maintaining 100% backward compatibility with piblin. Whether you're analyzing rheological data, performing uncertainty quantification, or building complex data transformation pipelines, piblin-jax provides the tools you need with modern Python ergonomics.

## Key Features

- **JAX-Powered Performance**: Leverage automatic differentiation and GPU/TPU acceleration
  - 5-10x speedup on CPU compared to piblin
  - 50-100x speedup on GPU for large datasets
  - Automatic JIT compilation for optimized execution

- **Bayesian Uncertainty Quantification**: Rigorous statistical inference with NumPyro
  - Built-in rheological models (Power Law, Arrhenius, Cross, Carreau-Yasuda)
  - Full posterior distributions for parameter estimates
  - Uncertainty propagation through transformations

- **100% piblin Compatibility**: Drop-in replacement for existing code
  ```python
  import piblin_jax as piblin  # Just change the import!
  ```

- **Modern Python 3.12+**: Type-safe, functional programming approach
  - Runtime: Python 3.12+ supported
  - Development: Python 3.13+ required for pre-commit hooks
  - Comprehensive type hints throughout
  - NumPy-style docstrings
  - Immutable data structures

- **Flexible Transform Pipeline**: Composable data transformations
  - Dataset transformations (smoothing, interpolation, calculus)
  - Region-based operations
  - Custom lambda transforms
  - Pipeline composition and reuse

- **Rich Data Structures**: Hierarchical data organization
  - Datasets (0D, 1D, 2D, 3D, composite, distributions)
  - Measurements and measurement sets
  - Experiments and experiment sets
  - Metadata and provenance tracking

## Installation

### Basic Installation

Install piblin-jax with JAX CPU support using pip:

```bash
pip install piblin-jax
```

**Note**: The package name is `piblin-jax` on PyPI, and you import it as `piblin_jax`:

```python
import piblin_jax  # Package installation: pip install piblin-jax
# or for piblin compatibility:
import piblin_jax as piblin
```

### GPU Support (Linux + CUDA 12+ Only)

**Platform Constraints:**
- ✅ **Linux + NVIDIA GPU + CUDA 12**: Full GPU acceleration (50-100x speedup)
- ❌ **macOS**: CPU backend only (no NVIDIA GPU support, still 5-10x faster than piblin)
- ❌ **Windows**: CPU backend only (CUDA support experimental/unstable in JAX)

**Requirements for GPU Acceleration:**
- Linux operating system
- NVIDIA GPU with CUDA Compute Capability 7.5 or newer
- CUDA 12.1-12.9 installed on system
- NVIDIA driver >= 525

**Performance Impact:** 50-100x speedup for large datasets (>1M points)

---

#### ✅ Recommended: Makefile Installation (All Package Managers)

**From repository (works with pip, uv, conda/mamba):**

```bash
git clone https://github.com/piblin/piblin-jax.git
cd piblin-jax
make init
make install-gpu-cuda  # Handles everything automatically
```

This single command:
- ✓ Validates platform (Linux only)
- ✓ Detects package manager (uv/conda/pip)
- ✓ Uninstalls CPU-only JAX
- ✓ Installs GPU-enabled JAX with CUDA 12
- ✓ Verifies GPU detection
- ✓ Shows installation summary

---

#### ⚙️ Manual Installation (Advanced Users)

**Why manual installation requires uninstall:** JAX has separate CPU and GPU builds. You MUST remove the CPU build before installing GPU to prevent silent failures where you think you have GPU but are actually using CPU.

**Using pip:**
```bash
# Step 1: Uninstall CPU-only version (REQUIRED - prevents conflicts)
pip uninstall -y jax jaxlib

# Step 2: Install GPU-enabled JAX
pip install "jax[cuda12-local]>=0.8.0,<0.9.0"

# Step 3: Install piblin-jax (if not already installed)
pip install piblin-jax

# Step 4: Verify GPU detection
python -c "import jax; print('Devices:', jax.devices())"
# Expected: [cuda(id=0)] NOT [CpuDevice(id=0)]
```

**Using uv:**
```bash
uv pip uninstall -y jax jaxlib
uv pip install "jax[cuda12-local]>=0.8.0,<0.9.0"
python -c "import jax; print(jax.devices())"
```

**Using conda/mamba:**

Option A: Using environment file (recommended):
```bash
# Using conda
conda env create -f environment-gpu.yml
conda activate piblin-jax-gpu

# Using mamba (faster)
mamba env create -f environment-gpu.yml
mamba activate piblin-jax-gpu
```

Option B: Manual within conda environment:
```bash
conda activate your-env
pip uninstall -y jax jaxlib
pip install "jax[cuda12-local]>=0.8.0,<0.9.0"
```

**Note:** Conda's extras syntax is not supported. Always use pip within your conda environment for JAX GPU installation.

---

#### 🔍 Verify GPU Installation

After installation, verify GPU is detected:

```bash
python -c "from piblin_jax.backend import get_device_info; print(get_device_info())"
```

**Expected output:**
```python
{'backend': 'jax', 'device_type': 'gpu', 'device_count': 1, ...}
```

**If you see `'device_type': 'cpu'`**, GPU installation failed. See troubleshooting below.

---

#### 🔧 Troubleshooting

**Issue: "GPU not detected" warning or `device_type: 'cpu'`**

```bash
# 1. Check GPU hardware
nvidia-smi  # Should show your GPU

# 2. Check CUDA version (need 12.1-12.9)
nvcc --version

# 3. Verify JAX sees GPU
python -c "import jax; print(jax.devices())"
# Expected: [cuda(id=0)]
# If showing: [CpuDevice(id=0)] → JAX is using CPU

# 4. If still CPU, reinstall with explicit uninstall:
pip uninstall -y jax jaxlib
pip install "jax[cuda12-local]>=0.8.0,<0.9.0"
```

**Issue: ImportError or "CUDA library not found"**

```bash
# Set CUDA library path
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Make permanent (add to ~/.bashrc)
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

**Issue: "An NVIDIA GPU may be present... but a CUDA-enabled jaxlib is not installed"**

This means you have GPU hardware but CPU-only JAX. Solution:
```bash
pip uninstall -y jax jaxlib
pip install "jax[cuda12-local]>=0.8.0,<0.9.0"
```

**Issue: Works in one environment but not another**

Different package managers may install different versions. Always use the same installation method:
- **Development**: `make install-gpu-cuda` (recommended)
- **Production**: Docker with explicit JAX version
- **Notebooks**: Manual pip installation with version pinning

**Note**: The package name on PyPI is `piblin-jax`, and the import name is `piblin_jax`.

### Development Installation

**Prerequisites:**
- **Runtime**: Python 3.12+ supported
- **Development**: Python 3.13+ required (for pre-commit hooks)
- **Package Manager**: uv recommended for development (not pip or conda)

For development with all optional dependencies:

```bash
git clone https://github.com/piblin/piblin-jax.git
cd piblin-jax

# Using uv (recommended for development)
uv pip install -e ".[dev]"

# Or using pip
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
import piblin_jax

# Read experimental data
data = piblin_jax.read_file('experiment.csv')

# Create a transform pipeline
from piblin_jax.transform import Pipeline, Interpolate1D, GaussianSmoothing

pipeline = Pipeline([
    Interpolate1D(new_x=new_points),
    GaussianSmoothing(sigma=2.0)
])

# Apply transformations
result = pipeline.apply_to(data)

# Visualize results
result.visualize()
```

### Bayesian Parameter Estimation

```python
from piblin_jax.bayesian import PowerLawModel
import numpy as np

# Your experimental data
shear_rate = np.array([0.1, 1.0, 10.0, 100.0])
viscosity = np.array([50.0, 15.8, 5.0, 1.58])

# Fit power-law model with uncertainty quantification
model = PowerLawModel(n_samples=2000, n_warmup=1000)
model.fit(shear_rate, viscosity)

# Get parameter estimates with credible intervals
summary = model.summary()
print(summary)

# Make predictions with uncertainty
new_shear_rate = np.logspace(-1, 2, 50)
predictions = model.predict(new_shear_rate)

# Visualize fit with uncertainty
model.plot_fit(shear_rate, viscosity, show_uncertainty=True)
```

### piblin Migration

Migrating from piblin is seamless:

```python
# Old code
import piblin
data = piblin.read_file('data.csv')

# New code - just change the import!
import piblin_jax as piblin
data = piblin.read_file('data.csv')
# All your existing piblin code works unchanged
```

## Performance Comparison

| Operation | piblin (CPU) | piblin-jax (CPU) | piblin-jax (GPU) | Speedup (GPU) |
|-----------|--------------|---------------|---------------|---------------|
| Dataset creation | 180 μs | 70 μs | - | 2.6x |
| Gaussian smoothing | 2000 μs | 1100 μs | 50 μs | 40x |
| Pipeline execution | 5 ms | 1.2 ms | 100 μs | 50x |
| Bayesian fitting | 45 s | 4.5 s | 0.5 s | 90x |

*Benchmarks on M1 Max (CPU) and NVIDIA A100 (GPU)*

## Documentation

Full documentation is available at [piblin-jax.readthedocs.io](https://piblin-jax.readthedocs.io)

- [Installation Guide](https://piblin-jax.readthedocs.io/en/latest/user_guide/installation.html)
- [Quick Start Tutorial](https://piblin-jax.readthedocs.io/en/latest/user_guide/quickstart.html)
- [Core Concepts](https://piblin-jax.readthedocs.io/en/latest/user_guide/concepts.html)
- [API Reference](https://piblin-jax.readthedocs.io/en/latest/api/)
- [Examples](https://piblin-jax.readthedocs.io/en/latest/tutorials/)

## Examples

Explore complete, runnable examples in the [`examples/`](examples/) directory:

- **basic_usage_example.py** - Core workflow: data loading, transforms, visualization
- **transform_pipeline_example.py** - Composable transform pipelines
- **bayesian_rheological_models.py** - Rheological model fitting (Power Law, Cross, Carreau-Yasuda, Arrhenius)
- **bayesian_parameter_estimation.py** - Advanced Bayesian inference techniques
- **uncertainty_propagation_example.py** - Propagating uncertainty through pipelines
- **piblin_migration_example.py** - Migrating from piblin to piblin-jax
- **gpu_acceleration_example.py** - Leveraging GPU for 10-100x speedups (Linux + CUDA 12+ only)
- **custom_transforms_example.py** - Building domain-specific transforms

Each example is fully documented and can be run directly:

```bash
cd examples
python basic_usage_example.py
```

See [`examples/README.md`](examples/README.md) for detailed descriptions and usage instructions.

## Development

### Prerequisites

**Python Version Requirements:**
- **Runtime**: Python 3.12+ supported for using the library
- **Development**: Python 3.13+ required for pre-commit hooks

**Package Manager:**
- **Recommended**: uv (not pip or conda) for development
- **User Installation**: pip is fine for end users

### Setup Development Environment

```bash
# Verify Python 3.13+ is available
python3.13 --version

# Clone and install in development mode with uv (recommended)
git clone https://github.com/piblin/piblin-jax.git
cd piblin-jax
uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"

# Install pre-commit hooks (requires Python 3.13+)
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=piblin_jax --cov-report=html

# Run only fast tests (skip slow/GPU tests)
pytest -m "not slow and not gpu"

# Run benchmarks
pytest -m benchmark
```

### Code Quality

piblin-jax maintains high code quality standards:

- **Test Coverage**: >95% required
- **Type Checking**: mypy strict mode
- **Code Formatting & Linting**: ruff (line length 100, 10-100x faster than legacy tools)
- **Documentation**: NumPy-style docstrings

Pre-commit hooks enforce these standards automatically.

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

**Quick Contribution Checklist:**
- Tests pass with >95% coverage
- Type hints added
- NumPy-style docstrings
- Pre-commit hooks pass
- Documentation updated

## Roadmap

- [x] Core data structures and transforms
- [x] JAX backend with automatic differentiation
- [x] NumPyro Bayesian inference integration
- [x] piblin compatibility layer
- [ ] Additional rheological models
- [ ] Time series analysis tools
- [ ] Advanced visualization capabilities
- [ ] Distributed computing support
- [ ] Real-time data acquisition integration

## Citation

If you use piblin-jax in your research, please cite:

```bibtex
@software{piblin_jax2025,
  author = {Chen, Wei},
  title = {piblin-jax: Modern JAX-Powered Framework for Measurement Data Science},
  year = {2025},
  url = {https://github.com/piblin/piblin-jax}
}
```

## License

piblin-jax is licensed under the [MIT License](LICENSE).

## Acknowledgments

- Forked from [piblin](https://github.com/3mcloud/piblin) by 3M - the original framework for measurement data science
- Built on [JAX](https://github.com/google/jax) for high-performance numerical computing
- Uses [NumPyro](https://github.com/pyro-ppl/numpyro) for Bayesian inference

## Support

- **Documentation**: [piblin-jax.readthedocs.io](https://piblin-jax.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/piblin/piblin-jax/issues)
- **Discussions**: [GitHub Discussions](https://github.com/piblin/piblin-jax/discussions)

---

Made with ❤️ by Wei Chen and the piblin-jax developers
