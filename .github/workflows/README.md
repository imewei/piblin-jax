# CI/CD Pipeline Documentation

This directory contains GitHub Actions workflows for continuous integration and deployment.

## Overview

The CI/CD pipeline enforces **dependency version consistency** to ensure reproducible builds across local development and CI environments.

## Workflows

### `ci.yml` - Main CI/CD Pipeline

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`
- Manual workflow dispatch

**Jobs:**

1. **validate-dependencies** - Ensures lock file consistency
   - Verifies Python version matches local environment (3.13.9)
   - Checks if `uv.lock` exists and is up to date
   - Creates initial lock file if missing (with warning)
   - Installs exact dependency versions from lock file

2. **lint** - Code quality checks
   - Ruff linter (check mode)
   - Ruff formatter (check mode)
   - MyPy type checking (non-blocking)

3. **test** - Multi-platform testing
   - Matrix: Python 3.12 & 3.13 on Ubuntu, macOS, Windows
   - CPU-only tests (excludes GPU and slow tests)
   - Coverage reporting to Codecov
   - Uses exact dependencies from `uv.lock`

4. **test-gpu** - GPU support testing (Linux only)
   - Runs on main/develop branches or manual trigger
   - Tests GPU-marked tests in CPU fallback mode
   - Notes about requiring self-hosted runner for real GPU testing

5. **security** - Security scanning
   - `pip-audit` for dependency vulnerabilities
   - `bandit` for security linting
   - Trivy filesystem scanning
   - Gitleaks secret detection
   - Results uploaded to GitHub Security

6. **build** - Package building
   - Builds distribution packages (wheel + sdist)
   - Generates SBOM (Software Bill of Materials)
   - Uploads build artifacts (7 days retention)
   - Uploads SBOM (30 days retention)

7. **dependency-review** - PR dependency analysis
   - Only runs on pull requests
   - Blocks moderate+ severity vulnerabilities
   - Denies GPL-3.0 and AGPL-3.0 licenses

8. **all-checks-passed** - Final status check
   - Aggregates all job results
   - Required for branch protection rules

## Dependency Version Consistency

### Key Principle

**Lock files are the source of truth.** The CI pipeline uses the exact same Python version and dependencies as your local environment.

### How It Works

1. **Python Version Matching**
   - Local: Specified in `.python-version` (currently 3.13.9)
   - CI: Reads from `.python-version` file
   - Verification step ensures exact match

2. **Dependency Locking**
   - Local: `uv lock` creates/updates `uv.lock`
   - CI: `uv sync --frozen` installs exact versions from `uv.lock`
   - Lock check: `uv lock --check` fails if pyproject.toml changed

3. **Installation Commands**
   ```bash
   # ✅ CORRECT - Uses lock file
   uv sync --frozen

   # ❌ WRONG - Resolves dependencies (non-deterministic)
   uv sync
   uv pip install piblin_jax
   ```

### Local Development Workflow

1. **Initial Setup**
   ```bash
   # Create virtual environment and install dependencies
   uv venv
   source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

   # Install with exact versions (if uv.lock exists)
   uv sync --frozen

   # OR create lock file if it doesn't exist
   uv lock
   uv sync
   ```

2. **Adding Dependencies**
   ```bash
   # Edit pyproject.toml manually, then:
   uv lock                    # Update lock file
   uv sync --frozen           # Install updated dependencies
   git add pyproject.toml uv.lock
   git commit -m "chore(deps): add new-package"
   ```

3. **Updating Dependencies**
   ```bash
   # Update specific package
   uv lock --upgrade-package requests

   # Update all packages
   uv lock --upgrade

   # Install updated versions
   uv sync --frozen

   # Commit lock file
   git add uv.lock
   git commit -m "chore(deps): update dependencies"
   ```

### Pre-commit Hooks

The repository includes pre-commit hooks that validate lock file consistency:

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

**Key Hook:** `uv-lock-check`
- Runs when `pyproject.toml` or `uv.lock` changes
- Verifies lock file is up to date
- Fails if `uv.lock` is missing or out of sync

### Troubleshooting

#### Lock File Out of Sync

**Symptom:** CI fails with "uv.lock is out of sync"

**Solution:**
```bash
uv lock
git add uv.lock
git commit -m "chore: update lock file"
git push
```

#### Python Version Mismatch

**Symptom:** "Python version mismatch" in CI

**Solution:**
```bash
# Check your local version
python --version

# Update .python-version if needed
echo "3.13.9" > .python-version

# Commit the change
git add .python-version
git commit -m "chore: update Python version"
```

#### "Works on My Machine" but CI Fails

**Common Causes:**
1. **Different Python version** - Check `.python-version` matches your local `python --version`
2. **Lock file not committed** - Ensure `uv.lock` is in git and up to date
3. **Environment variables** - Check if local env vars affect behavior

**Debug Steps:**
```bash
# 1. Verify you're in venv
which python  # Should point to .venv/bin/python

# 2. Verify lock file exists and is current
uv lock --check

# 3. Clean install (matches CI)
rm -rf .venv
uv venv
uv sync --frozen

# 4. Run tests locally
uv run pytest
```

#### CI Stuck on "Creating lock file"

**Symptom:** First CI run creates lock file instead of using existing one

**Solution:**
```bash
# Lock file wasn't committed
git add uv.lock
git commit -m "chore: add uv.lock"
git push
```

## Security Features

### Automated Security Scanning

- **pip-audit**: Scans Python dependencies for known vulnerabilities
- **bandit**: Static analysis for security issues in code
- **Trivy**: Container and filesystem vulnerability scanning
- **Gitleaks**: Prevents committing secrets

### Dependency Review

Pull requests automatically check for:
- Vulnerable dependencies (moderate+ severity blocked)
- License compliance (GPL-3.0, AGPL-3.0 denied)
- Supply chain security issues

### SBOM Generation

Every build generates a Software Bill of Materials (SBOM) in CycloneDX format:
- Enables vulnerability tracking
- Required for compliance (NIST, CISA)
- Available as build artifact

## Automated Dependency Updates

### Dependabot Configuration

Dependabot automatically creates PRs for dependency updates:

- **Schedule**: Weekly on Mondays at 3:00 AM
- **Python Dependencies**: Grouped by production/development/major
- **GitHub Actions**: All actions grouped together
- **Labels**: Auto-tagged with `dependencies`, `python`, or `github-actions`

### Reviewing Dependabot PRs

1. **Check CI Status**: All checks must pass
2. **Review Changes**: Check CHANGELOG/release notes
3. **Test Locally** (for major updates):
   ```bash
   gh pr checkout <PR-number>
   uv sync --frozen
   uv run pytest
   ```
4. **Merge**: Use "Squash and merge" for clean history

## GPU Testing

### Current Setup (CPU Fallback)

GPU tests run in CPU fallback mode in CI because GitHub-hosted runners don't have GPUs.

### Future: Self-Hosted GPU Runner

To enable real GPU testing:

1. **Setup Self-Hosted Runner** with CUDA 12+
   ```bash
   # On GPU machine (Linux only)
   nvidia-smi  # Verify CUDA 12+ is installed

   # Register runner with GitHub
   # Settings → Actions → Runners → New self-hosted runner
   ```

2. **Update Workflow**
   ```yaml
   test-gpu:
     runs-on: [self-hosted, linux, gpu, cuda12]
   ```

3. **Add GPU Tests**
   ```python
   @pytest.mark.gpu
   def test_gpu_acceleration():
       assert jax.default_backend() == "gpu"
   ```

## Caching Strategy

The pipeline uses aggressive caching to speed up builds:

- **uv cache**: Keyed on `uv.lock` hash
- **Shared across jobs**: All jobs use same cache
- **Auto-invalidation**: Cache updates when lock file changes

**Expected Speedup:**
- First run: ~2-3 minutes (no cache)
- Subsequent runs: ~30-60 seconds (cached)

## Branch Protection Rules

Recommended settings for `main` branch:

- ✅ Require pull request reviews (1 approval)
- ✅ Require status checks to pass:
  - `validate-dependencies`
  - `lint`
  - `test`
  - `security`
  - `build`
- ✅ Require branches to be up to date
- ✅ Require conversation resolution
- ✅ Require signed commits (optional)

## Monitoring and Alerts

### Failed Builds

GitHub automatically sends email notifications for failed workflows on `main`/`develop`.

### Security Alerts

- **Dependabot alerts**: Navigate to Security → Dependabot alerts
- **Code scanning**: Navigate to Security → Code scanning
- **Secret scanning**: Navigate to Security → Secret scanning

## Best Practices

1. **Always commit lock files**: `uv.lock` must be in version control
2. **Use frozen installs in CI**: `uv sync --frozen` ensures reproducibility
3. **Match Python versions**: Local and CI must use same Python version
4. **Run pre-commit locally**: Catch issues before pushing
5. **Review Dependabot PRs weekly**: Don't let security updates pile up
6. **Test major updates locally**: Don't blindly merge major version bumps
7. **Keep SBOM artifacts**: Required for security audits

## References

- [uv documentation](https://docs.astral.sh/uv/)
- [GitHub Actions documentation](https://docs.github.com/actions)
- [Dependabot configuration](https://docs.github.com/code-security/dependabot)
- [Pre-commit framework](https://pre-commit.com/)
