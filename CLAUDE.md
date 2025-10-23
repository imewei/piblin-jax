# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Last Updated**: 2025-10-23

## Recent Updates

**Major changes since last version:**
- **GPU Installation Overhaul (BREAKING)**: Removed pip extras (`gpu-cuda`), now requires explicit installation via Makefile or manual process. Eliminates silent CPU/GPU conflicts across package managers.
- **Type Safety Achievement**: 100% mypy strict mode compliance (eliminated 130+ type errors)
- **Test Suite Expansion**: 97.14% coverage with 617+ tests (5,317 new test lines added)
- **Critical Bug Fixes**: Fixed Transform._copy_tree immutability issue using copy.deepcopy()
- **Documentation Improvements**: Enhanced GPU installation guide, restructured README

## Project Overview

**quantiq** is a modern JAX-powered framework for measurement data science, providing a high-performance reimplementation of the [piblin](https://github.com/3mcloud/piblin) library. Key characteristics:

- **Performance**: 5-10x CPU speedup, 50-100x GPU acceleration over piblin
- **Backend abstraction**: Dual JAX/NumPy backend with automatic fallback
- **piblin compatibility**: 100% API behavioral compatibility (import quantiq as piblin)
- **Bayesian inference**: NumPyro integration for uncertainty quantification
- **Functional design**: Immutable data structures, lazy evaluation, JIT compilation

## Development Commands

### Environment Setup
```bash
# Initialize development environment (uses uv package manager)
make init                    # Creates .venv and installs dev dependencies

# Install different configurations
make install-dev             # Development dependencies
make install-test            # Test dependencies only
make install-gpu-cuda        # CUDA GPU support (Linux only)
```

**IMPORTANT - GPU Installation (Breaking Change)**:
- The `quantiq[gpu-cuda]` pip extra has been removed
- GPU installation now requires `make install-gpu-cuda` (recommended) or manual installation
- Reason: pip extras cause silent CPU/GPU conflicts across package managers
- See README.md GPU Installation section for details

### Testing
```bash
# Run tests
make test                    # Fast tests only (no GPU, no slow tests)
make test-fast               # Fastest iteration (no benchmarks)
make test-cov                # Tests with coverage report (requires >95%)
make test-slow               # Include slow tests
make test-gpu                # GPU-specific tests
make test-all                # All tests including slow and GPU

# Run single test or test file
uv run pytest tests/path/to/test_file.py::test_function_name -v

# Run tests matching a pattern
uv run pytest -k "pattern" -v
```

**Test Coverage**: Currently at **97.14%** with **617+ tests** passing.

### Code Quality
```bash
# Formatting and linting
make format                  # Auto-format with ruff
make format-check            # Check formatting without changes
make lint                    # Run ruff linter
make type-check              # Run mypy type checker (100% strict compliance)
make check                   # Run all checks (format + lint + type)

# Quick iteration (format + fast tests)
make quick

# Full QA (check + test-cov)
make qa
```

### Pre-commit Hooks
```bash
make pre-commit-install      # Install hooks (requires Python 3.13)
make pre-commit-run          # Run on all files
```

### CI/CD
The project uses GitHub Actions with a multi-stage pipeline:
1. **validate-dependencies**: Ensures uv.lock is in sync with pyproject.toml
2. **lint**: Ruff linting and formatting, mypy type checking
3. **test**: Matrix testing on Python 3.12/3.13 across Ubuntu/macOS/Windows
4. **test-gpu**: GPU tests (CPU fallback in CI)
5. **security**: pip-audit, bandit, trivy, gitleaks
6. **build**: Package building and SBOM generation

## Architecture

### Module Structure

```
quantiq/
├── backend/               # JAX/NumPy abstraction layer
├── data/
│   ├── datasets/         # Core dataset classes (0D, 1D, 2D, 3D, composite, distributions)
│   │   └── base.py       # Abstract Dataset base class
│   ├── collections/      # Measurement, MeasurementSet, Experiment, ExperimentSet
│   ├── metadata.py       # Metadata handling
│   └── roi.py           # Region of interest operations
├── transform/
│   ├── base.py          # Transform abstract base classes
│   ├── pipeline.py      # Pipeline composition
│   ├── lambda_transform.py  # Lambda/custom transforms
│   ├── dataset/         # Dataset-level transforms (smoothing, interpolation, calculus)
│   ├── measurement/     # Measurement-level transforms
│   └── region/          # Region-based operations
├── bayesian/            # NumPyro Bayesian inference (JAX-dependent)
│   ├── base.py          # BayesianModel base class
│   └── models/          # PowerLaw, Arrhenius, Cross, CarreauYasuda
├── dataio/
│   ├── readers/         # CSV readers and file I/O
│   └── writers/         # Data export
└── fitting/             # NLSQ non-linear fitting
```

### Key Design Patterns

#### Backend Abstraction
- `quantiq.backend` provides a unified interface (`jnp`) that maps to either `jax.numpy` or `numpy`
- **Platform detection**: GPU support only on Linux with CUDA 12+
- **Auto-fallback**: Automatically uses NumPy if JAX unavailable
- **Conditional imports**: Bayesian module only imported when JAX available
- Check backend: `backend.is_jax_available()`, `backend.get_backend()`, `backend.get_device_info()`

```python
from quantiq.backend import jnp  # Either jax.numpy or numpy
```

#### Dataset Hierarchy
All datasets inherit from `Dataset` abstract base class:
- **Immutable design**: Compatible with JAX transformations (jit, grad, vmap)
- **Metadata system**: `conditions` (experimental parameters) and `details` (context)
- **Internal storage**: Backend arrays (JAX DeviceArray or NumPy ndarray)
- **External API**: Properties return NumPy arrays for user-facing APIs

#### Transform Pipeline
Transforms follow a hierarchical design:
- **Transform[T]**: Abstract base class with lazy evaluation and JIT compilation support
- **DatasetTransform**: Operates on Dataset objects
- **MeasurementTransform**: Operates on Measurement objects
- **Pipeline**: Composable chain of transforms with `apply_to()` method
- **LambdaTransform**: Custom user-defined transforms

All transforms support:
- Lazy evaluation (`_lazy` flag)
- JIT compilation (`_compiled` flag)
- Immutability via `make_copy` parameter
- Pipeline composition

**CRITICAL - Transform Immutability**:
- Use `copy.deepcopy()` for copying tree structures (NOT JAX tree operations)
- See `Transform._copy_tree()` in `transform/base.py:188-200`
- JAX tree operations don't preserve non-JAX-compatible types

## Important Implementation Details

### JAX Backend Requirements
- **Platform restrictions**: GPU acceleration requires Linux + CUDA 12+
- **Platform detection**: Use `backend._detect_platform()` to check OS
- **CUDA validation**: `backend._validate_cuda_version()` checks for CUDA 12+
- **Installation**: Use `make install-gpu-cuda` (auto-detects uv/conda/mamba/pip)

### Testing Markers
```python
# Use pytest markers to categorize tests
@pytest.mark.slow         # Slow-running tests
@pytest.mark.gpu          # Requires GPU
@pytest.mark.benchmark    # Performance benchmarks
@pytest.mark.visual       # Visual regression tests
```

### Type Hints
- **Strict mypy**: All code must pass `mypy --strict` (100% compliance achieved)
- **Modern Python**: Uses Python 3.12+ syntax (generics via `class Transform[T]`)
- **Type stubs**: External libraries (JAX, NumPyro) have `ignore_missing_imports = true`

### Code Style
- **Line length**: 100 characters
- **Ruff rules**: E, W, F, I, N, UP, B, C4, SIM, RUF enabled
- **Explicit imports**: Zero star imports (`from X import *`) allowed in codebase
- **Scientific naming**: Allow single-letter math names (N802, N803, N806, N816 ignored)
- **Unicode support**: Greek letters allowed in code/docs (RUF001, RUF002, RUF003 ignored)
- **NumPy-style docstrings**: All public APIs must have complete docstrings

### Coverage Requirements
- **Minimum coverage**: 95% required (`--cov-fail-under=95`)
- **Current coverage**: 97.14% (617+ tests)
- **Exclusions**: Tests, `__init__.py`, setup.py, abstract methods
- **Reports**: term-missing, HTML, and XML formats

### Testing with Mock JAX
When testing backend functionality that needs to simulate JAX unavailable:
```python
# Use pytest-mock to patch JAX availability
def test_numpy_fallback(monkeypatch):
    monkeypatch.setattr("quantiq.backend._JAX_AVAILABLE", False)
    # Test NumPy fallback behavior
```

## piblin Compatibility

quantiq maintains 100% behavioral compatibility with piblin:
- Users can `import quantiq as piblin` and existing code works unchanged
- Same API surface, same behavior, dramatically better performance
- All dataset types, transforms, and I/O operations are compatible

## Development Best Practices

1. **Always use the backend abstraction**: Import from `quantiq.backend` not directly from JAX/NumPy
2. **Test both backends**: Ensure code works with JAX and NumPy fallback
3. **Immutable design**: Datasets should not be modified in-place (use `make_copy=True`)
4. **Type everything**: Add type hints to all functions, use `TypeVar` for generics
5. **Document with NumPy style**: Include Parameters, Returns, Examples sections
6. **Test coverage**: Add tests to maintain >95% coverage
7. **Platform awareness**: Consider Linux/macOS/Windows differences for GPU code
8. **Use copy.deepcopy()**: For transform tree copying (not JAX tree operations)

## Common Workflows

### Adding a New Transform
1. Create class inheriting from appropriate base (`DatasetTransform`, etc.)
2. Implement `_apply(self, target: T) -> T` method
3. Add type hints and NumPy-style docstring
4. Add tests with >95% coverage
5. Update `transform/__init__.py` exports
6. Use `copy.deepcopy()` if copying tree structures

### Adding a Bayesian Model
1. Create model in `bayesian/models/`
2. Inherit from `BayesianModel` base class
3. Implement `_model()` method with NumPyro syntax
4. Add GPU and slow test markers
5. Ensure conditional import handling (JAX-dependent)
6. Use `@jit` + `@staticmethod` pattern for performance

### Fixing Backend Issues
- Check platform detection first: `backend._detect_platform()`
- Verify CUDA version on Linux: `backend._get_cuda_version()`
- Test both JAX available and unavailable scenarios
- Use `backend.to_numpy()` and `backend.from_numpy()` for conversions

## Package Management

This project uses **uv** (not pip or conda) for all package management:
- **Lock file**: `uv.lock` must be kept in sync with `pyproject.toml`
- **Sync dependencies**: `uv sync --frozen` (CI) or `uv sync` (local)
- **Run commands**: `uv run pytest`, `uv run ruff`, etc.
- **Build**: `uv build` (not `python -m build`)
- **GPU installation**: `make install-gpu-cuda` auto-detects uv/conda/mamba/pip

## Security and Quality

- **Security scanning**: pip-audit, bandit, trivy, gitleaks in CI
- **SBOM generation**: CycloneDX software bill of materials
- **Dependency review**: Automated on pull requests
- **License restrictions**: GPL-3.0 and AGPL-3.0 denied

## Breaking Changes

### v0.1.0 (Current Development)
- **GPU Installation**: Removed `quantiq[gpu-cuda]` pip extra
  - **Migration**: Use `make install-gpu-cuda` or manual installation
  - **Rationale**: pip extras cause silent CPU/GPU conflicts
  - **Documentation**: See README.md GPU Installation section

## Known Issues & Limitations

1. **GPU Support Platform Constraints**:
   - Linux + CUDA 12+ only
   - macOS: CPU-only (no NVIDIA GPU support)
   - Windows: CPU-only (JAX CUDA support experimental/unstable)

2. **Type Checking**:
   - Some JAX/NumPyro types require `ignore_missing_imports`
   - Generics require Python 3.12+ syntax

3. **Transform Immutability**:
   - Must use `copy.deepcopy()` for tree copying
   - JAX tree operations don't preserve non-JAX types
