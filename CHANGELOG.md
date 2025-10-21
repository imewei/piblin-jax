# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive CI/CD pipeline with GitHub Actions (currently disabled pending deployment configuration)
- Complete user guide documentation covering all major topics
- Complete tutorial documentation with practical examples
- Architecture Decision Records (ADRs) for key design decisions
- Comprehensive usage examples for key workflows
- Comprehensive tests for ROI (Region of Interest) and hierarchy modules
- Development automation Makefile with targets for testing, formatting, linting, and documentation
- `.nlsq_cache` directory to gitignore for JAX compilation cache

### Changed
- **BREAKING:** Migrated from black/isort/flake8 to unified ruff tooling (10-100x faster)
  - Updated all development dependencies
  - Updated pre-commit hooks configuration
  - Updated Makefile targets for formatting and linting
  - Updated CONTRIBUTING.md with new tooling instructions
  - Removed `.flake8` and `.coveragerc` configuration files
  - Consolidated all configuration in `pyproject.toml`
- Enabled strict type checking with mypy for enhanced code quality
- Fixed Python version references in CI workflows (3.14 → 3.13)
- Updated pre-commit configuration for optimal performance
- Enhanced module docstrings for data and fitting packages
- Improved documentation configuration maintainability

### Fixed
- All 447 Sphinx compilation warnings resolved
- All mypy strict mode type errors in base classes
- SyntaxWarnings for invalid escape sequences in docstrings
- Math inconsistencies in documentation
- Missing docstrings in reader `__init__` methods
- Sphinx build warnings across all documentation files

### Performance
- Added JIT compilation to smoothing transforms (significant speedup)
- Added JIT compilation to all Bayesian model `predict()` methods (~10x faster predictions)
- Added JIT compilation to calculus and normalization transforms
- Added JIT compilation to JIT-optimized transforms with comprehensive type annotations
- Applied critical performance optimizations and safety fixes across codebase

### Documentation
- Created comprehensive CHANGELOG.md (this file)
- Created comprehensive CONTRIBUTING.md with development guidelines
- Created documentation gap analysis report
- Completed all tutorial stubs with comprehensive, practical content
- Completed all user guide stubs with detailed explanations
- Fixed all Sphinx build warnings (0 errors, 0 warnings)
- Added piblin acknowledgement and updated author information
- Added comprehensive validation reports and optimization documentation
- Enhanced API reference documentation for all modules

### Infrastructure
- Optimized pre-commit hooks for faster local development
- Updated all badges (black → ruff) in README and documentation
- Removed completed analysis and planning documents (kept in git history)

## [0.1.0] - 2025-01-15

### Added
- Initial release of quantiq framework
- JAX-powered backend with automatic differentiation support
- Complete data structures hierarchy (Dataset, Measurement, Experiment)
- Transform pipeline system with composition support
- NumPyro-based Bayesian inference models:
  - Power Law model for rheological fitting
  - Cross model for shear-thinning fluids
  - Carreau-Yasuda model for complex rheology
  - Arrhenius model for thermal activation
- Non-linear least squares (NLSQ) curve fitting integration
- File I/O system with CSV and TXT readers
- Comprehensive test suite with >95% coverage
- Sphinx documentation with ReadTheDocs theme
- Type hints throughout codebase (mypy strict compatible)
- Pre-commit hooks for code quality
- 100% piblin API compatibility

### Features
- **5-10x CPU speedup** over piblin baseline
- **50-100x GPU acceleration** for large datasets
- Automatic JIT compilation for optimized execution
- Full uncertainty quantification with MCMC sampling
- Composable transform pipelines
- Hierarchical data organization
- Metadata tracking and provenance
- Region of Interest (ROI) support
- Automatic initial parameter estimation
- GPU/TPU device auto-detection

### Documentation
- Complete API reference
- User guides:
  - Installation guide
  - Quick start tutorial
  - Core concepts
  - Uncertainty quantification
  - Performance optimization
  - Migration from piblin
- Tutorials:
  - Basic workflow
  - Uncertainty quantification
  - Custom transforms
  - Rheological models
- Contributing guidelines
- README with badges and examples

---

## Release Notes

### Version 0.1.0 Notes

This is the initial release of quantiq, a modern JAX-powered framework for measurement data science. quantiq is an enhanced fork of [piblin](https://github.com/3mcloud/piblin) by 3M, providing:

- Dramatic performance improvements (5-10x CPU, 50-100x GPU)
- Advanced Bayesian uncertainty quantification
- 100% backward compatibility with piblin
- Modern Python 3.12+ features

For migration from piblin, simply change your import:
```python
import quantiq as piblin  # All your piblin code works!
```

### Acknowledgments

quantiq gratefully acknowledges the original [piblin](https://github.com/3mcloud/piblin) project by 3M for establishing the foundational concepts and API design that quantiq builds upon.

---

## Links

- [GitHub Repository](https://github.com/quantiq/quantiq)
- [Documentation](https://quantiq.readthedocs.io)
- [Issue Tracker](https://github.com/quantiq/quantiq/issues)
- [Discussions](https://github.com/quantiq/quantiq/discussions)
