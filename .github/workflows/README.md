# CI/CD Pipeline Documentation

This directory contains GitHub Actions workflows for continuous integration, deployment, and security scanning.

## Overview

The CI/CD pipeline ensures code quality, security, and reliability through automated testing and validation.

### Workflows

1. **CI/CD Pipeline** (`.github/workflows/ci.yml`)
   - Runs on: Push to `main`/`develop`, Pull Requests
   - Duration: ~5-10 minutes
   - Key features:
     - Dependency lock file validation
     - Multi-version testing (Python 3.12, 3.13)
     - Cross-platform testing (Linux, macOS, Windows)
     - Code quality checks (Ruff, mypy)
     - Test coverage (95% minimum)
     - Documentation builds
     - Security scanning

2. **Security Scanning** (`.github/workflows/security.yml`)
   - Runs on: Weekly schedule (Mondays 3 AM), Push to `main`, Pull Requests
   - Duration: ~3-5 minutes
   - Key features:
     - Secret detection (Gitleaks)
     - Dependency vulnerability scanning (pip-audit)
     - Static analysis (Semgrep, CodeQL)
     - License compliance checking
     - SBOM generation

## Dependency Version Consistency

### Critical Principle

**The `uv.lock` file is the source of truth for all dependency versions.**

This ensures:
- ✅ Reproducible builds across all environments
- ✅ Identical dependencies in development, CI, and production
- ✅ No "works on my machine" issues
- ✅ Predictable security vulnerability management

### How It Works

1. **Local Development**
   ```bash
   # Install exact dependencies from lock file
   uv sync --frozen

   # Add new dependency (updates pyproject.toml and uv.lock)
   uv add package-name

   # Update dependencies (regenerates uv.lock)
   uv lock --upgrade
   ```

2. **Pre-commit Hook**
   - Automatically validates `uv.lock` is in sync before commits
   - Prevents committing outdated lock files
   - Run manually: `pre-commit run uv-lock-check --all-files`

3. **CI Pipeline**
   - First job validates lock file integrity
   - All subsequent jobs use `uv sync --frozen`
   - Fails fast if lock file is out of sync

### Common Scenarios

#### Adding a New Dependency

```bash
# Add the package (automatically updates lock file)
uv add numpy

# Commit both files
git add pyproject.toml uv.lock
git commit -m "feat: add numpy dependency"
```

#### Updating Dependencies

```bash
# Update all dependencies
uv lock --upgrade

# Or update specific package
uv lock --upgrade-package jax

# Review changes
git diff uv.lock

# Commit if everything looks good
git add uv.lock
git commit -m "chore(deps): update dependencies"
```

#### Lock File Out of Sync

If CI fails with lock file validation error:

```bash
# Regenerate lock file locally
uv lock

# Commit the updated lock file
git add uv.lock
git commit -m "chore: update lock file"
git push
```

## Workflow Jobs

### CI Pipeline (`ci.yml`)

#### 1. Validate Dependencies
- **Purpose**: Ensure reproducible builds
- **Checks**:
  - Lock file is in sync with `pyproject.toml`
  - No dependency drift
  - Security vulnerability scan
- **Failure**: Prevents all downstream jobs from running

#### 2. Lint
- **Tool**: Ruff
- **Checks**:
  - Code style (PEP 8)
  - Import sorting
  - Code complexity
  - Security patterns
- **Config**: `pyproject.toml` → `[tool.ruff]`

#### 3. Type Check
- **Tool**: mypy
- **Checks**:
  - Static type correctness
  - Type annotations
  - Generic type usage
- **Config**: `pyproject.toml` → `[tool.mypy]`

#### 4. Test
- **Tool**: pytest
- **Matrix**:
  - Python: 3.12, 3.13
  - OS: Ubuntu, macOS, Windows
- **Coverage**: Minimum 95%
- **Markers**:
  - `not slow`: Fast tests only in PRs
  - `not gpu`: Skip GPU tests in CI
- **Artifacts**:
  - Coverage report (Codecov)
  - HTML coverage report

#### 5. Slow Tests
- **Runs**: Only on `main`/`develop` branches
- **Purpose**: Expensive integration tests
- **Trigger**: After fast tests pass

#### 6. Build
- **Purpose**: Verify package builds correctly
- **Outputs**:
  - Wheel (`.whl`)
  - Source distribution (`.tar.gz`)
- **Artifacts**: Retained for 7 days

#### 7. Docs
- **Tool**: Sphinx
- **Output**: HTML documentation
- **Deployment**: Read the Docs (automatic)

#### 8. Security
- **Tools**:
  - Bandit (Python security linter)
  - Trivy (vulnerability scanner)
- **Output**: SARIF reports to GitHub Security

#### 9. CodeQL
- **Purpose**: Advanced security analysis
- **Language**: Python
- **Queries**: Security + Quality

### Security Workflow (`security.yml`)

#### Secret Scanning
- **Tool**: Gitleaks
- **Scans**: Full git history
- **Purpose**: Detect accidentally committed secrets

#### Dependency Audit
- **Tool**: pip-audit
- **Database**: OSV, PyPI Advisory Database
- **Action**: Fails on HIGH/CRITICAL vulnerabilities

#### SAST
- **Tool**: Semgrep
- **Rules**: Auto (security-focused)
- **Output**: SARIF for GitHub Security tab

#### License Compliance
- **Tool**: pip-licenses
- **Checks**: GPL/AGPL detection
- **Output**: Markdown license report

#### SBOM
- **Tool**: CycloneDX
- **Format**: JSON
- **Purpose**: Software supply chain transparency

## Branch Protection Rules

Recommended settings for `main` branch:

```yaml
Require a pull request before merging: ✅
  Required approvals: 1
  Dismiss stale reviews: ✅

Require status checks to pass: ✅
  Status checks required:
    - CI Status Check
    - Lint (Ruff)
    - Type Check (mypy)
    - Test (Python 3.12) (ubuntu-latest)
    - Test (Python 3.13) (ubuntu-latest)
    - Build Package
    - Security Scan

Require conversation resolution before merging: ✅
Require signed commits: ⚠️ (Optional but recommended)
Include administrators: ✅
```

## Secrets Configuration

Required GitHub Secrets:

| Secret | Purpose | Required |
|--------|---------|----------|
| `CODECOV_TOKEN` | Coverage reporting | Optional |
| `GITLEAKS_LICENSE` | Gitleaks Pro features | Optional |

To add secrets:
1. Go to repository Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Add name and value

## Automated Dependency Updates

### Dependabot Configuration

- **Location**: `.github/dependabot.yml`
- **Schedule**: Weekly (Mondays 3 AM)
- **Ecosystems**:
  - Python dependencies (pip)
  - GitHub Actions versions

#### Dependency Groups

- **jax-ecosystem**: JAX, JAXlib, NumPyro, Optax
- **testing**: pytest, hypothesis
- **docs**: Sphinx, numpydoc
- **dev-tools**: Ruff, mypy, pre-commit

### Reviewing Dependabot PRs

1. **Automated Checks**: All CI must pass
2. **Review Changes**: Check CHANGELOG in PR description
3. **Breaking Changes**: Review carefully, may require code updates
4. **Security Updates**: Merge quickly after validation

## Performance Optimization

### Caching Strategy

The pipeline uses aggressive caching:

```yaml
# uv cache (dependencies)
- uses: astral-sh/setup-uv@v4
  with:
    enable-cache: true

# Based on lock file hash
key: ${{ runner.os }}-uv-${{ hashFiles('**/uv.lock') }}
```

### Parallel Execution

Jobs run in parallel where possible:

```
validate-dependencies (30s)
├── lint (1m) ────────────┐
├── type-check (1.5m) ────┤
├── test (3m) ────────────┼─→ build (1m) ─→ ci-status
├── docs (2m) ────────────┤
└── security (2m) ────────┘
```

Total time: ~5-7 minutes (vs. 10+ sequential)

## Troubleshooting

### Common Issues

#### 1. Lock File Out of Sync

**Error:**
```
❌ ERROR: uv.lock is out of sync with pyproject.toml!
```

**Solution:**
```bash
uv lock
git add uv.lock
git commit -m "chore: sync lock file"
```

#### 2. Pre-commit Hook Fails

**Error:**
```
Verify uv.lock is up to date...Failed
```

**Solution:**
```bash
# Fix the lock file
uv lock

# Or bypass (NOT RECOMMENDED)
git commit --no-verify
```

#### 3. Coverage Below Threshold

**Error:**
```
FAILED coverage: total coverage is 94.5%, expected at least 95%
```

**Solution:**
```bash
# Find uncovered lines
pytest --cov=quantiq --cov-report=term-missing

# Add tests for uncovered code
# Or adjust threshold in pyproject.toml (if justified)
```

#### 4. Mypy Type Errors

**Error:**
```
error: Incompatible return value type
```

**Solution:**
```bash
# Run mypy locally
uv run mypy quantiq/

# Add type hints or use type: ignore comments
x: int = 5  # type: ignore[assignment]
```

#### 5. Security Vulnerabilities

**Error:**
```
pip-audit found 2 known vulnerabilities
```

**Solution:**
```bash
# Check vulnerabilities
uv run pip-audit

# Update vulnerable packages
uv lock --upgrade-package vulnerable-package

# Or wait for Dependabot PR
```

## Local Testing

Run CI checks locally before pushing:

```bash
# Full pre-commit checks
pre-commit run --all-files

# Lock file validation
uv lock --locked

# Linting
uv run ruff check .
uv run ruff format --check .

# Type checking
uv run mypy quantiq/

# Tests
uv run pytest --cov=quantiq

# Build
uv build

# Docs
cd docs && uv run make html
```

## Monitoring

### GitHub Actions Dashboard

View workflow status:
- Repository → Actions tab
- Click workflow name for details
- View logs for failed jobs

### Security Alerts

Review security findings:
- Repository → Security tab
- Code scanning alerts (CodeQL, Trivy, Semgrep)
- Dependabot alerts
- Secret scanning alerts

### Coverage Reports

- **Codecov**: Detailed coverage analytics (if configured)
- **Artifacts**: Download HTML report from workflow run

## Best Practices

### 1. Keep Lock File Updated

```bash
# Regularly update dependencies
uv lock --upgrade

# Or let Dependabot handle it weekly
```

### 2. Review Security Alerts Promptly

- Check Security tab weekly
- Merge security updates quickly
- Subscribe to notifications

### 3. Maintain High Test Coverage

- Target: 95%+ coverage
- Write tests before features (TDD)
- Cover edge cases

### 4. Use Semantic Commits

```bash
git commit -m "feat: add new feature"
git commit -m "fix: resolve bug"
git commit -m "chore(deps): update dependencies"
```

### 5. Run Pre-commit Hooks

```bash
# Install once
pre-commit install

# Runs automatically on git commit
# Or manually:
pre-commit run --all-files
```

## Additional Resources

- [uv documentation](https://github.com/astral-sh/uv)
- [GitHub Actions documentation](https://docs.github.com/actions)
- [Dependabot documentation](https://docs.github.com/code-security/dependabot)
- [Codecov documentation](https://docs.codecov.com)
- [Ruff documentation](https://docs.astral.sh/ruff/)
