# Contributing to quantiq

Thank you for your interest in contributing to quantiq! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment Setup](#development-environment-setup)
- [Development Workflow](#development-workflow)
- [Code Quality Standards](#code-quality-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation Requirements](#documentation-requirements)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

## Code of Conduct

By participating in this project, you agree to maintain a respectful, inclusive, and collaborative environment. We expect all contributors to:

- Be respectful and constructive in all communications
- Welcome newcomers and help them get started
- Focus on what is best for the community and the project
- Show empathy towards other community members

## Getting Started

### Prerequisites

**Python Version Requirements:**
- **Runtime**: Python 3.12+ supported for using the library
- **Development**: Python 3.13+ required for pre-commit hooks

**Package Manager:**
- **Recommended**: uv (not pip or conda) for development
- **Why uv**: Faster dependency resolution, deterministic builds, better lock file management
- **User Installation**: pip is fine for end users installing the package

**Other Requirements:**
- Git
- Basic understanding of JAX and NumPy
- Familiarity with scientific computing concepts

> **Note**: Pre-commit hooks require Python 3.13 to align with CI/CD environments and ruff target version. While the package runs on Python 3.12+, development requires 3.13.

### Quick Contribution Checklist

Before submitting a pull request, ensure:

- ✅ Tests pass with >95% coverage
- ✅ Type hints added to all new code
- ✅ NumPy-style docstrings for all public APIs
- ✅ Pre-commit hooks pass (ruff, mypy)
- ✅ Documentation updated
- ✅ CHANGELOG.md updated (for user-facing changes)

## Development Environment Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/quantiq.git
cd quantiq
```

### 2. Install Development Dependencies

**Using uv (Recommended):**

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode with all dependencies
uv pip install -e ".[dev,test,docs]"
```

**Using pip (Alternative):**

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev,test,docs]"
```

### 3. Install Pre-commit Hooks

```bash
# Install pre-commit hooks (requires Python 3.13+)
pre-commit install

# (Optional) Run against all files to verify setup
pre-commit run --all-files
```

### 4. GPU Development Setup (Linux Only)

**Note**: GPU acceleration is only available on Linux with NVIDIA GPUs. macOS and Windows will use CPU backend (which still provides 5-10x speedup over piblin).

**Requirements:**
- Linux operating system
- NVIDIA GPU with CUDA Compute Capability 7.5 or newer
- CUDA 12.1+ installed on system

**Quick Installation via Makefile:**

```bash
# Automated installation (recommended)
make install-gpu-cuda
```

This command will:
1. Validate you're running on Linux (fails on macOS/Windows with clear message)
2. Check that virtual environment exists
3. Uninstall CPU-only JAX to avoid conflicts
4. Install GPU-enabled JAX with CUDA 12 support
5. Verify GPU detection

**Manual Installation:**

```bash
# Uninstall CPU-only JAX first
uv pip uninstall -y jax jaxlib

# Install with GPU support
uv sync --extra gpu-cuda
```

**Verification:**

After installation, verify GPU is detected:

```bash
python -c "from quantiq.backend import get_device_info; print(get_device_info())"
```

Expected output:
```python
{'backend': 'jax', 'device_type': 'gpu', 'device_count': 1, ...}
```

**Troubleshooting:**

If GPU is not detected:
1. Check CUDA version: `nvidia-smi` (should show CUDA 12.1+)
2. Verify JAX sees GPU: `python -c "import jax; print(jax.devices())"`
3. Ensure `LD_LIBRARY_PATH` is not set or points to correct CUDA libraries
4. Reinstall following the manual instructions above

**Testing GPU Code:**

Run GPU-specific tests:

```bash
# Run only GPU tests
make test-gpu

# Or with pytest directly
uv run pytest tests/ -m gpu -v
```

**Important Notes:**
- JAX CPU and GPU versions use different `jaxlib` packages - they cannot coexist
- Always uninstall CPU version before installing GPU version
- The `make install-gpu-cuda` target handles this automatically
- CI/CD runs on CPU even for GPU tests (uses JAX CPU fallback)

### Migrating from Python 3.12 or 3.14

If you previously had the repository set up with a different Python version:

```bash
# 1. Verify Python 3.13 is installed
python3.13 --version

# 2. Pull latest changes (includes Python 3.13 requirement)
git pull origin main

# 3. Clean pre-commit cache
pre-commit clean

# 4. Reinstall pre-commit hooks with Python 3.13
pre-commit install --install-hooks

# 5. Verify hooks work correctly
pre-commit run --all-files
```

**Why Python 3.13?** The pre-commit configuration now explicitly requires Python 3.13 to match:
- CI/CD environment (Python 3.13)
- Ruff target version (py313)
- Ensures consistent linting and formatting behavior across all developer environments

### 4. Verify Installation

```bash
# Run tests to verify setup
pytest

# Check type hints
mypy quantiq

# Verify code formatting
ruff check quantiq tests
ruff format --check quantiq tests
```

## Development Workflow

### 1. Create a Feature Branch

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

- Write your code following the [Code Quality Standards](#code-quality-standards)
- Add tests following the [Testing Requirements](#testing-requirements)
- Update documentation following the [Documentation Requirements](#documentation-requirements)

### 3. Run Quality Checks

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=quantiq --cov-report=html

# Check type hints
mypy quantiq

# Format code (pre-commit hooks will do this automatically)
ruff format quantiq tests

# Run linter
ruff check quantiq tests
```

### 4. Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add Bayesian uncertainty propagation

- Implement Monte Carlo uncertainty propagation
- Add credible interval computation
- Update documentation with examples
- Add comprehensive tests achieving 98% coverage"
```

**Commit Message Guidelines**:
- Use conventional commits format: `type: description`
- Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`
- Keep first line under 72 characters
- Add detailed description in body if needed
- Reference issues: `Fixes #123` or `Relates to #456`

### 5. Push and Create Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create pull request on GitHub
# Fill out the PR template with details
```

## Code Quality Standards

quantiq maintains high code quality through automated tooling and manual review.

### Code Style

**Formatting** (Enforced by pre-commit hooks):
- **Ruff**: Unified linter and formatter (10-100x faster than black/isort/flake8)
  - Line length: 100 characters
  - Python 3.13+ target
  - Auto-fixes imports, code style, and common errors
  - Supports scientific Python conventions (Greek letters, single-letter math names)

**Configuration**: See `pyproject.toml` for detailed settings.

### Type Hints

**Required** for all new code:

```python
# Good: Complete type hints
def process_data(
    x: np.ndarray,
    y: np.ndarray,
    conditions: dict[str, Any] | None = None
) -> OneDimensionalDataset:
    """Process experimental data into dataset."""
    pass

# Bad: Missing or incomplete type hints
def process_data(x, y, conditions=None):
    pass
```

**Rules**:
- Use modern Python 3.12+ syntax: `dict[str, Any]` not `Dict[str, Any]`
- Use union syntax: `str | None` not `Optional[str]`
- Type all function signatures (parameters and return types)
- Type all class attributes
- Pass `mypy --strict` validation

### Naming Conventions

Follow PEP 8:

- **Classes**: PascalCase (`OneDimensionalDataset`, `PowerLawModel`)
- **Functions/Methods**: snake_case (`get_credible_intervals`, `apply_to`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_TOLERANCE`, `MAX_ITERATIONS`)
- **Private members**: `_leading_underscore` (`_samples`, `_mcmc`)
- **Module-level "private"**: `_leading_underscore` (`_validate_input`)

### Import Organization

Imports organized in three groups (ruff enforces this):

```python
# Standard library
import copy
from typing import Any

# Third-party
import numpy as np
import jax.numpy as jnp

# Local/quantiq
from quantiq.backend import jnp, to_numpy
from quantiq.data.datasets import Dataset
```

### Backend Abstraction

**Always use the backend abstraction layer**:

```python
# Good: Use backend abstraction
from quantiq.backend import jnp, to_numpy

def transform_data(x):
    x_backend = jnp.asarray(x)
    result = jnp.sin(x_backend)
    return to_numpy(result)

# Bad: Direct JAX import
import jax.numpy as jnp  # Don't do this in quantiq modules
```

## Testing Requirements

### Coverage Target

- **Minimum**: 95% code coverage (enforced by pytest-cov)
- **Goal**: >98% coverage with meaningful tests

### Test Organization

See [TESTING.md](TESTING.md) for comprehensive testing strategy including:
- When to consolidate vs. separate test files
- Fixture organization and reuse
- Test naming conventions
- Property-based testing guidelines

### Quick Testing Guide

```python
# Good test structure
class TestPowerLawModel:
    """Test PowerLawModel: η = K * γ̇^(n-1)"""

    @pytest.fixture
    def power_law_data(self):
        """Generate synthetic power-law data."""
        np.random.seed(42)
        shear_rate = np.logspace(-2, 3, 50)
        # ... generate test data
        return shear_rate, viscosity

    def test_fit_recovers_parameters(self, power_law_data):
        """Test that model recovers true parameters from data."""
        shear_rate, viscosity = power_law_data

        model = PowerLawModel(n_samples=500)
        model.fit(shear_rate, viscosity)

        # Verify parameter recovery
        K_mean = np.mean(model.samples["K"])
        assert_allclose(K_mean, true_K, rtol=0.3)
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/bayesian/test_numpyro_integration.py

# Run specific test
pytest tests/bayesian/test_numpyro_integration.py::test_power_law_fit

# Skip slow tests
pytest -m "not slow"

# Run with coverage
pytest --cov=quantiq --cov-report=html

# Run benchmarks
pytest -m benchmark
```

### Test Markers

Use markers to categorize tests:

```python
@pytest.mark.slow
def test_large_mcmc_chain():
    """Test MCMC with 10,000 samples (slow)."""
    pass

@pytest.mark.gpu
def test_jax_gpu_acceleration():
    """Test GPU performance improvement."""
    pass

@pytest.mark.benchmark
def test_transform_pipeline_performance(benchmark):
    """Benchmark transform pipeline execution."""
    pass
```

## Documentation Requirements

### Docstring Style

Use NumPy-style docstrings for all public APIs:

```python
def fit_curve(
    x: np.ndarray,
    y: np.ndarray,
    model: str = "power_law",
    initial_params: dict[str, float] | None = None
) -> dict[str, Any]:
    """
    Fit a rheological model to viscosity data.

    This function fits various rheological models to experimental data
    using non-linear least squares optimization. Supports power-law,
    Arrhenius, Cross, and Carreau-Yasuda models.

    Parameters
    ----------
    x : np.ndarray
        Independent variable data (e.g., shear rate, temperature)
    y : np.ndarray
        Dependent variable data (e.g., viscosity)
    model : str, default='power_law'
        Model type: 'power_law', 'arrhenius', 'cross', or 'carreau_yasuda'
    initial_params : dict[str, float] | None, optional
        Initial parameter guesses. If None, uses automatic initialization.

    Returns
    -------
    dict[str, Any]
        Fitted parameters and optimization results:
        - 'params': dict of fitted parameter values
        - 'covariance': parameter covariance matrix
        - 'residuals': fitting residuals
        - 'success': optimization success flag

    Raises
    ------
    ValueError
        If model name is not recognized
    ValueError
        If x and y have different shapes

    Examples
    --------
    >>> import numpy as np
    >>> from quantiq import fit_curve
    >>> shear_rate = np.logspace(-1, 2, 30)
    >>> viscosity = 5.0 * shear_rate ** (0.6 - 1)
    >>> result = fit_curve(shear_rate, viscosity, model='power_law')
    >>> print(result['params'])
    {'K': 5.02, 'n': 0.598}

    >>> # With initial guesses
    >>> initial = {'K': 3.0, 'n': 0.5}
    >>> result = fit_curve(shear_rate, viscosity, initial_params=initial)

    Notes
    -----
    The power-law model is defined as:

    .. math::

        \\eta(\\dot{\\gamma}) = K \\dot{\\gamma}^{n-1}

    where η is viscosity, γ̇ is shear rate, K is consistency index,
    and n is the power-law index.

    For more advanced uncertainty quantification, use the Bayesian
    models from `quantiq.bayesian.models`.

    See Also
    --------
    quantiq.bayesian.models.PowerLawModel : Bayesian power-law fitting
    estimate_initial_parameters : Automatic parameter initialization

    References
    ----------
    .. [1] Bird, R.B., Armstrong, R.C., and Hassager, O. (1987).
           "Dynamics of Polymeric Liquids", Vol. 1, 2nd ed., Wiley.
    """
    pass
```

**Required Sections**:
- Short summary (one line)
- Extended description (if needed)
- `Parameters` - All parameters with types and descriptions
- `Returns` - Return value type and description
- `Raises` - Exceptions that may be raised
- `Examples` - At least one working example
- `Notes` - Implementation details, equations, caveats
- `See Also` - Related functions/classes
- `References` - Academic citations (if applicable)

### Property Docstrings

Use `:no-index:` to avoid Sphinx warnings:

```python
@property
def samples(self) -> dict[str, np.ndarray] | None:
    """
    Get posterior samples from MCMC.

    :no-index:

    Returns
    -------
    dict[str, np.ndarray] | None
        Dictionary mapping parameter names to sample arrays,
        or None if model has not been fit.

    Examples
    --------
    >>> model.fit(x, y)
    >>> samples = model.samples
    >>> slope_samples = samples['slope']
    """
    return self._samples
```

### Module Docstrings

All modules should have docstrings explaining purpose:

```python
"""
Transform system for quantiq.

This module provides the transform framework for data processing:
- Base transform classes for each hierarchy level
- Pipeline composition for sequential transforms
- Lazy evaluation for JAX optimization
- JIT compilation support

Hierarchy:
- Transform: Abstract base class
- DatasetTransform: Operates on Dataset objects
- MeasurementTransform: Operates on Measurement objects
...
"""
```

### Documentation Updates

When adding new features:

1. **API Reference**: Docstrings auto-generate API docs
2. **User Guide**: Update `docs/source/user_guide/` RST files
3. **Examples**: Add example to `examples/` directory
4. **README**: Update README.md if it's a major feature
5. **CHANGELOG**: Add entry to CHANGELOG.md

## Pull Request Process

### Before Submitting

- [ ] All tests pass (`pytest`)
- [ ] Coverage remains >95% (`pytest --cov=quantiq`)
- [ ] Type checking passes (`mypy quantiq`)
- [ ] Code formatted (`ruff format quantiq tests`)
- [ ] Linting passes (`ruff check quantiq tests`)
- [ ] Documentation built successfully (`cd docs && make html`)
- [ ] CHANGELOG.md updated (for user-facing changes)

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update

## Motivation and Context
Why is this change required? What problem does it solve?
If it fixes an open issue, please link: Fixes #(issue)

## How Has This Been Tested?
Describe tests added/modified:
- [ ] New unit tests added
- [ ] Integration tests added
- [ ] Property-based tests added
- [ ] Manual testing performed

## Coverage
- Previous coverage: XX.XX%
- New coverage: XX.XX%

## Checklist
- [ ] Tests pass locally
- [ ] Type hints added
- [ ] Docstrings added/updated
- [ ] Pre-commit hooks pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
```

### Review Process

1. **Automated Checks**: CI runs tests, type checking, linting
2. **Code Review**: Maintainers review code quality and design
3. **Discussion**: Address feedback and questions
4. **Approval**: At least one maintainer approval required
5. **Merge**: Squash and merge to main branch

## Release Process

### Versioning

quantiq follows [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes (e.g., 1.0.0 → 2.0.0)
- **MINOR**: Backwards-compatible functionality (e.g., 0.1.0 → 0.2.0)
- **PATCH**: Backwards-compatible bug fixes (e.g., 0.1.0 → 0.1.1)

### Release Checklist (Maintainers)

- [ ] All tests pass on main branch
- [ ] Coverage >95%
- [ ] CHANGELOG.md updated with all changes
- [ ] Version bumped in `pyproject.toml` and `quantiq/__init__.py`
- [ ] Documentation built and deployed
- [ ] Git tag created: `git tag v0.1.0`
- [ ] Tag pushed: `git push --tags`
- [ ] GitHub release created with release notes
- [ ] Package built: `uv build` (or `python -m build`)
- [ ] Package uploaded to PyPI: `twine upload dist/*`

## Platform Constraints Reference

When documenting features or contributing code, ensure consistency with these platform constraints:

**GPU Support:**
- **Linux with CUDA 12+**: Full GPU support (50-100x speedup)
- **macOS**: CPU backend only (5-10x speedup via JAX)
- **Windows**: CPU backend only (5-10x speedup via JAX)
- **No support**: AMD ROCm, Apple Metal (deprecated)

**Python Versions:**
- **Runtime**: Python 3.12+ supported for library usage
- **Development**: Python 3.13+ required for pre-commit hooks

**Package Manager:**
- **Development**: uv recommended (not pip or conda)
- **User Installation**: pip is standard for end users

## Additional Resources

- [TESTING.md](TESTING.md) - Comprehensive testing strategy
- [README.md](README.md) - Project overview and quick start
- [Documentation](https://quantiq.readthedocs.io) - Full documentation
- [GitHub Issues](https://github.com/quantiq/quantiq/issues) - Bug reports and feature requests
- [GitHub Discussions](https://github.com/quantiq/quantiq/discussions) - Questions and community

## Getting Help

- **Questions**: Open a [GitHub Discussion](https://github.com/quantiq/quantiq/discussions)
- **Bugs**: File an [Issue](https://github.com/quantiq/quantiq/issues)
- **Security**: Email security concerns to maintainers

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to quantiq!
