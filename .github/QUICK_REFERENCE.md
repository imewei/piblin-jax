# CI/CD Quick Reference Card

## Essential Commands

### Dependency Management (uv)

```bash
# Generate/update lock file
uv lock

# Update all dependencies
uv lock --upgrade

# Update specific package
uv lock --upgrade-package numpy

# Install from lock file
uv sync --frozen

# Install with dev dependencies
uv sync --frozen --group dev

# Install specific groups
uv sync --frozen --group test
uv sync --frozen --group docs
```

### Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov

# Run specific tests
uv run pytest tests/test_core.py

# Run excluding slow tests
uv run pytest -m "not slow"

# Run in parallel
uv run pytest -n auto
```

### Code Quality

```bash
# Lint code
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .

# Format code
uv run ruff format .

# Type checking
uv run mypy quantiq
```

### Pre-commit Hooks

```bash
# Install hooks
pre-commit install

# Run all hooks
pre-commit run --all-files

# Run specific hook
pre-commit run ruff --all-files
pre-commit run uv-lock-check --all-files

# Update hook versions
pre-commit autoupdate
```

### Docker

```bash
# Development environment
docker-compose up dev

# Run tests in Docker
docker-compose run --rm test

# Jupyter notebook
docker-compose up jupyter
# Access at http://localhost:8888

# Build documentation
docker-compose up docs
# Access at http://localhost:8080

# GPU support
docker-compose up gpu

# Clean up
docker-compose down -v
```

### Git Workflow

```bash
# Feature development
git checkout -b feature/my-feature
# ... make changes ...
git add .
git commit -m "feat: add my feature"
git push origin feature/my-feature

# Update dependencies
vim pyproject.toml
uv lock
git add pyproject.toml uv.lock
git commit -m "chore(deps): add new-package"

# Create release
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin v0.2.0
```

## CI/CD Pipeline Stages

| Stage | Runs On | Purpose |
|-------|---------|---------|
| validate-dependencies | All branches | Verify uv.lock is up to date |
| lint | All branches | Code quality checks |
| test | All branches | Multi-version/platform testing |
| test-slow | main only | Slow/benchmark tests |
| security | All branches | Vulnerability scanning |
| build | All branches | Build Python packages |
| docs | All branches | Build documentation |
| deploy-docs | main only | Deploy to GitHub Pages |
| publish-test-pypi | develop only | Publish to Test PyPI |
| publish-pypi | tags only | Publish to PyPI |

## Required Status Checks

For `main` branch protection:
- `validate-dependencies`
- `lint`
- `test (3.12, ubuntu-latest)`
- `security`
- `build`
- `all-checks-passed`

## Environment Variables

### CI/CD
- `CODECOV_TOKEN` - Codecov upload token (optional)

### Docker
- `PYTHONUNBUFFERED=1` - Unbuffered Python output
- `JAX_PLATFORM_NAME=gpu` - Force GPU backend
- `CUDA_VISIBLE_DEVICES=0` - GPU device selection

## File Locations

| File | Purpose |
|------|---------|
| `.github/workflows/ci.yml` | Main CI/CD pipeline |
| `.github/dependabot.yml` | Dependency updates |
| `.github/CODEOWNERS` | Code ownership |
| `.pre-commit-config.yaml` | Pre-commit hooks |
| `pyproject.toml` | Project metadata & deps |
| `uv.lock` | Dependency lock file |
| `Dockerfile` | Multi-stage builds |
| `docker-compose.yml` | Development environment |

## Common Issues

### Lock file out of sync
```bash
uv lock
git add uv.lock
git commit -m "chore: update uv.lock"
```

### Tests failing
```bash
# Ensure using locked dependencies
rm -rf .venv
uv sync --frozen
uv run pytest
```

### Pre-commit failures
```bash
# See what's wrong
pre-commit run --all-files

# Skip hooks (not recommended)
git commit --no-verify
```

### Docker build issues
```bash
docker-compose down -v
docker-compose build --no-cache
```

## URLs

- **Repository:** https://github.com/imewei/quantiq
- **Documentation:** https://imewei.github.io/quantiq
- **Test PyPI:** https://test.pypi.org/project/quantiq
- **PyPI:** https://pypi.org/project/quantiq
- **Codecov:** https://codecov.io/gh/imewei/quantiq

## Support

- **Setup Guide:** [SETUP_CICD.md](../SETUP_CICD.md)
- **Full Documentation:** [CI_CD_SETUP.md](./CI_CD_SETUP.md)
- **Issues:** https://github.com/imewei/quantiq/issues

---

**Pro Tips:**

- Always run `uv lock` after changing `pyproject.toml`
- Use `uv sync --frozen` in CI for reproducible builds
- Commit `uv.lock` to version control
- Run pre-commit hooks before pushing
- Test locally before creating PRs
- Use feature branches, not main
- Keep commits atomic and well-described

Last updated: 2025-10-20
