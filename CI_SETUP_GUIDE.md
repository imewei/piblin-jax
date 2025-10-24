# CI/CD Setup Guide

This guide will help you set up and verify the CI/CD pipeline with dependency version consistency enforcement.

## 🚀 Quick Start

### 1. Create Lock File (First Time Setup)

If you don't have a `uv.lock` file yet:

```bash
# Create the lock file from pyproject.toml
uv lock

# Install dependencies with exact versions
uv sync --frozen

# Commit the lock file
git add uv.lock .python-version
git commit -m "chore: add dependency lock file and Python version"
```

### 2. Install Pre-commit Hooks

```bash
# Install pre-commit hooks (will run on every commit)
pre-commit install

# Test the hooks manually
pre-commit run --all-files
```

### 3. Verify CI Configuration

```bash
# Check that all required files are in place
ls -la .github/workflows/ci.yml
ls -la .github/dependabot.yml
ls -la .python-version
ls -la uv.lock

# Verify Python version matches
python --version  # Should match .python-version (3.13.9)
```

### 4. Push to GitHub

```bash
git push
```

The CI pipeline will automatically run on push!

---

## 📋 What Was Set Up

### Files Created

1. **`.python-version`** - Pins Python version (3.13.9)
   - Ensures CI uses the exact same Python version as local development
   - Read by `setup-python` action in CI

2. **`.github/workflows/ci.yml`** - Main CI/CD pipeline
   - ✅ Dependency version validation
   - ✅ Multi-platform testing (Ubuntu, macOS, Windows)
   - ✅ Security scanning (pip-audit, bandit, Trivy, Gitleaks)
   - ✅ Code quality checks (Ruff, MyPy)
   - ✅ Build and SBOM generation
   - ✅ GPU testing support (Linux only)

3. **`.github/dependabot.yml`** - Automated dependency updates
   - Weekly updates on Mondays at 3:00 AM
   - Groups dependencies by type (production/development/major)
   - Auto-labels and assigns PRs

4. **`.github/workflows/README.md`** - Comprehensive documentation
   - Workflow details and job descriptions
   - Troubleshooting guide
   - Best practices

5. **`.pre-commit-config.yaml`** (enhanced)
   - Added/updated `uv-lock-check` hook
   - Validates lock file on every commit

---

## 🔒 Dependency Version Consistency

### The Problem

Different dependency versions between local development and CI cause:
- ❌ "Works on my machine" but fails in CI
- ❌ Non-reproducible builds
- ❌ Hidden bugs from version differences
- ❌ Security vulnerabilities in CI but not local (or vice versa)

### The Solution

**Lock files + Exact Python version matching**

```
Local Development          CI/CD Pipeline
┌─────────────────┐       ┌─────────────────┐
│ .python-version │ ───▶  │ .python-version │
│ (3.13.9)        │       │ (3.13.9)        │
└─────────────────┘       └─────────────────┘
        │                         │
        ▼                         ▼
┌─────────────────┐       ┌─────────────────┐
│ uv.lock         │ ───▶  │ uv.lock         │
│ (exact versions)│       │ (exact versions)│
└─────────────────┘       └─────────────────┘
        │                         │
        ▼                         ▼
┌─────────────────┐       ┌─────────────────┐
│ uv sync --frozen│       │ uv sync --frozen│
└─────────────────┘       └─────────────────┘
```

### Key Commands

| Task | Command | Notes |
|------|---------|-------|
| **First install** | `uv sync --frozen` | Uses existing `uv.lock` |
| **Add dependency** | Edit `pyproject.toml` → `uv lock` → `uv sync --frozen` | Updates lock file |
| **Update deps** | `uv lock --upgrade` → `uv sync --frozen` | Updates all packages |
| **Update one package** | `uv lock --upgrade-package requests` | Updates specific package |
| **Verify lock** | `uv lock --check` | Fails if lock is out of sync |

---

## 🧪 Testing the CI Pipeline

### Local Testing (Before Push)

```bash
# 1. Verify lock file is current
uv lock --check

# 2. Run pre-commit hooks
pre-commit run --all-files

# 3. Run tests (same as CI)
uv run pytest -m "not gpu and not slow" --cov

# 4. Run linter (same as CI)
uv run ruff check .
uv run ruff format --check .

# 5. Run type checker (same as CI)
uv run mypy piblin_jax
```

### First Push to GitHub

```bash
# Add all CI configuration files
git add .python-version .github/workflows/ .github/dependabot.yml

# Commit
git commit -m "ci: setup CI/CD pipeline with dependency version consistency"

# Push and watch CI run
git push

# Monitor CI at: https://github.com/piblin/piblin-jax/actions
```

Expected workflow run time:
- **First run**: ~3-5 minutes (no cache)
- **Subsequent runs**: ~1-2 minutes (with cache)

---

## ✅ Verification Checklist

After pushing, verify the following:

### GitHub Actions

- [ ] Navigate to **Actions** tab
- [ ] See "CI/CD Pipeline" workflow running
- [ ] All jobs should pass:
  - [ ] `validate-dependencies`
  - [ ] `lint`
  - [ ] `test` (all matrix combinations)
  - [ ] `security`
  - [ ] `build`
  - [ ] `all-checks-passed`

### GitHub Security

- [ ] Navigate to **Security** tab → **Code scanning**
- [ ] See Trivy scan results
- [ ] No critical/high vulnerabilities

### Dependabot

- [ ] Navigate to **Security** tab → **Dependabot**
- [ ] See dependency graph
- [ ] Enable Dependabot alerts (if not already enabled)

---

## 🔧 Common Issues and Solutions

### Issue 1: "uv.lock is out of sync"

**Cause:** `pyproject.toml` was modified but `uv.lock` wasn't updated

**Solution:**
```bash
uv lock
git add uv.lock
git commit -m "chore: update lock file"
git push
```

### Issue 2: "Python version mismatch"

**Cause:** Local Python version doesn't match `.python-version`

**Solution:**
```bash
# Check your version
python --version

# If different, update .python-version
echo "$(python --version | cut -d' ' -f2)" > .python-version
git add .python-version
git commit -m "chore: update Python version to match local"
git push
```

### Issue 3: "uv.lock not found"

**Cause:** Lock file wasn't committed

**Solution:**
```bash
uv lock
git add uv.lock
git commit -m "chore: add uv.lock"
git push
```

### Issue 4: Pre-commit hook fails

**Cause:** Various (code style, security issues, etc.)

**Solution:**
```bash
# See what failed
pre-commit run --all-files

# Auto-fix what can be fixed
uv run ruff check --fix .

# Commit fixes
git add .
git commit -m "style: fix linting issues"
```

---

## 📊 Pipeline Jobs Explained

### 1. validate-dependencies
**Purpose:** Ensure reproducible builds

**What it does:**
- Verifies Python version matches `.python-version`
- Checks if `uv.lock` exists
- Validates lock file is up to date (`uv lock --check`)
- Installs dependencies with exact versions (`uv sync --frozen`)

**Why it matters:** Prevents "works on my machine" issues

### 2. lint
**Purpose:** Code quality enforcement

**What it does:**
- Ruff linter: Checks code style, common errors
- Ruff formatter: Validates formatting
- MyPy: Type checking (non-blocking)

**Why it matters:** Maintains consistent code quality

### 3. test (Matrix)
**Purpose:** Cross-platform compatibility

**What it does:**
- Tests on Ubuntu, macOS, Windows
- Tests Python 3.12 and 3.13
- Runs CPU tests only (excludes GPU and slow tests)
- Generates coverage report

**Why it matters:** Ensures package works everywhere

### 4. test-gpu
**Purpose:** GPU support verification

**What it does:**
- Runs GPU-marked tests in CPU fallback mode
- Only on main/develop branches
- Linux only (matches production GPU environment)

**Why it matters:** Validates GPU code paths (even in CPU mode)

### 5. security
**Purpose:** Vulnerability detection

**What it does:**
- `pip-audit`: Scans for vulnerable dependencies
- `bandit`: Security linting for code
- `Trivy`: Filesystem vulnerability scan
- `Gitleaks`: Secret detection

**Why it matters:** Prevents security issues from merging

### 6. build
**Purpose:** Package distribution

**What it does:**
- Builds wheel and source distribution
- Generates SBOM (Software Bill of Materials)
- Uploads artifacts

**Why it matters:** Verifies package can be built and distributed

---

## 🎯 Next Steps

### Enable Branch Protection

1. Go to **Settings** → **Branches**
2. Add rule for `main` branch:
   - ✅ Require pull request reviews (1 approval)
   - ✅ Require status checks:
     - `validate-dependencies`
     - `lint`
     - `test`
     - `security`
     - `build`
   - ✅ Require branches to be up to date

### Set Up Codecov (Optional)

1. Sign up at [codecov.io](https://codecov.io)
2. Add repository
3. Add `CODECOV_TOKEN` to GitHub Secrets
4. Coverage reports will be automatic

### Enable Dependabot Security Updates

1. Go to **Settings** → **Security & analysis**
2. Enable:
   - ✅ Dependency graph
   - ✅ Dependabot alerts
   - ✅ Dependabot security updates

### Add GPU Runner (Future)

For real GPU testing, set up self-hosted runner:

1. Provision Linux machine with CUDA 12+
2. **Settings** → **Actions** → **Runners** → **New self-hosted runner**
3. Update `test-gpu` job to use `runs-on: [self-hosted, linux, gpu]`

---

## 📚 References

- **CI/CD Documentation**: [.github/workflows/README.md](.github/workflows/README.md)
- **uv Documentation**: https://docs.astral.sh/uv/
- **Pre-commit**: https://pre-commit.com/
- **GitHub Actions**: https://docs.github.com/actions

---

## 🆘 Getting Help

If you encounter issues:

1. Check [.github/workflows/README.md](.github/workflows/README.md) for detailed troubleshooting
2. Review failed job logs in GitHub Actions tab
3. Run commands locally to reproduce issues
4. Check that Python version matches between local and CI

**Key Principle:** If it works locally with `uv sync --frozen`, it should work in CI with the same command.
