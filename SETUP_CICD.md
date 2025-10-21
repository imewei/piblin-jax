# CI/CD Setup Instructions

This document provides step-by-step instructions to complete the CI/CD pipeline setup for QuantiQ.

## Prerequisites

- Python 3.12 or higher
- Git
- GitHub account with admin access to the repository

## Step 1: Install uv Package Manager

uv is a fast Python package manager that ensures dependency version consistency.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version
```

## Step 2: Generate Lock File

**CRITICAL**: The CI/CD pipeline requires a `uv.lock` file for dependency version consistency.

```bash
# Generate lock file from pyproject.toml
uv lock

# This creates uv.lock with exact dependency versions
```

**What this does:**
- Resolves all dependencies and their versions
- Creates a lock file with exact versions and hashes
- Ensures reproducible builds across all environments (local, CI, production)

## Step 3: Install Dependencies

```bash
# Install all dependencies (including dev) from lock file
uv sync

# Or install only production dependencies
uv sync --no-dev

# Or install specific groups
uv sync --group test
uv sync --group docs
```

## Step 4: Install Pre-commit Hooks

```bash
# Install pre-commit package
uv pip install --system pre-commit

# Install git hooks
pre-commit install

# (Optional) Test hooks on all files
pre-commit run --all-files
```

This sets up local validation that runs before each commit:
- Ruff linting and formatting
- Bandit security checks
- Lock file validation
- YAML/TOML/JSON validation
- And more...

## Step 5: Commit Lock File and CI Configuration

```bash
# Stage all CI/CD files
git add uv.lock
git add .github/
git add .pre-commit-config.yaml
git add .dockerignore
git add Dockerfile
git add docker-compose.yml
git add SETUP_CICD.md

# Commit
git commit -m "feat: add comprehensive CI/CD pipeline with dependency locking

- Add GitHub Actions workflow with dependency validation
- Add pre-commit hooks for local validation
- Add Dependabot configuration
- Add Docker multi-stage builds
- Add comprehensive documentation
- Generate uv.lock for reproducible builds

🤖 Generated with Claude Code"

# Push to GitHub
git push origin main
```

## Step 6: Configure GitHub Repository Settings

### Enable Required Features

1. Go to **Settings → General**
   - [ ] Enable Issues (if not already enabled)
   - [ ] Enable Pull Requests (if not already enabled)

2. Go to **Settings → Code security and analysis**
   - [x] Enable Dependency graph
   - [x] Enable Dependabot alerts
   - [x] Enable Dependabot security updates
   - [x] Enable Dependabot version updates (will use .github/dependabot.yml)

### Set Up Branch Protection

Go to **Settings → Branches → Add branch protection rule**

#### For `main` branch:

**Branch name pattern:** `main`

Configure:
- [x] Require a pull request before merging
  - Required approvals: **1**
  - [x] Dismiss stale pull request approvals when new commits are pushed
  - [x] Require review from Code Owners

- [x] Require status checks to pass before merging
  - [x] Require branches to be up to date before merging
  - Search and add these required status checks:
    - `validate-dependencies`
    - `lint`
    - `test (3.12, ubuntu-latest)`
    - `security`
    - `build`
    - `all-checks-passed`

- [x] Require conversation resolution before merging

- [x] Require linear history

- [x] Do not allow bypassing the above settings

Click **Create** or **Save changes**

### Set Up GitHub Pages (for documentation)

1. Go to **Settings → Pages**
2. Source: **Deploy from a branch**
3. Branch: **gh-pages** / **root**
4. Click **Save**

Documentation will be available at: `https://imewei.github.io/quantiq` (after first deployment)

### Set Up PyPI Trusted Publishing

#### Test PyPI (for develop branch)

1. Go to [Test PyPI](https://test.pypi.org/)
2. Register/login to your account
3. Go to **Account settings → Publishing**
4. Click **Add a new pending publisher**
5. Fill in:
   - **PyPI Project Name:** `quantiq`
   - **Owner:** `imewei` (your GitHub username)
   - **Repository name:** `quantiq`
   - **Workflow name:** `ci.yml`
   - **Environment name:** `test-pypi`
6. Click **Add**

#### Production PyPI (for version tags)

1. Go to [PyPI](https://pypi.org/)
2. Register/login to your account
3. Go to **Account settings → Publishing**
4. Click **Add a new pending publisher**
5. Fill in:
   - **PyPI Project Name:** `quantiq`
   - **Owner:** `imewei` (your GitHub username)
   - **Repository name:** `quantiq`
   - **Workflow name:** `ci.yml`
   - **Environment name:** `pypi`
6. Click **Add**

### Create GitHub Environments

1. Go to **Settings → Environments**
2. Click **New environment**
3. Create **test-pypi**:
   - Name: `test-pypi`
   - Deployment branches: **Selected branches** → Add `develop`
   - Click **Save protection rules**

4. Click **New environment**
5. Create **pypi**:
   - Name: `pypi`
   - Deployment branches: **Protected branches only**
   - (Optional) Add yourself as required reviewer
   - (Optional) Set wait timer to 5 minutes
   - Click **Save protection rules**

### (Optional) Set Up Codecov

1. Go to [Codecov](https://codecov.io/)
2. Sign in with GitHub
3. Add your repository
4. Copy the repository upload token
5. Add to GitHub Secrets:
   - Go to **Settings → Secrets and variables → Actions**
   - Click **New repository secret**
   - Name: `CODECOV_TOKEN`
   - Value: Paste your token
   - Click **Add secret**

## Step 7: Test the Pipeline

### Create a Test Branch and PR

```bash
# Create a test branch
git checkout -b test/ci-pipeline

# Make a small change
echo "# Test" >> README.md

# Commit and push
git add README.md
git commit -m "test: verify CI pipeline"
git push origin test/ci-pipeline
```

### Create Pull Request

1. Go to your GitHub repository
2. Click **Pull requests → New pull request**
3. Base: `main` ← Compare: `test/ci-pipeline`
4. Click **Create pull request**
5. Wait for all checks to complete ✅

### Verify All Checks Pass

You should see these checks:
- ✅ validate-dependencies
- ✅ lint
- ✅ test (3.12, ubuntu-latest)
- ✅ test (3.12, macos-latest)
- ✅ test (3.12, windows-latest)
- ✅ test (3.13, ubuntu-latest)
- ✅ security
- ✅ build
- ✅ docs
- ✅ all-checks-passed

If any checks fail, click on them to see the logs and fix the issues.

## Step 8: Verify Dependabot

1. Go to **Insights → Dependency graph → Dependabot**
2. You should see Dependabot scheduled to run weekly on Mondays
3. Dependabot will create PRs for dependency updates automatically

## Common Issues and Solutions

### Issue: Lock file missing

**Error in CI:**
```
❌ No uv.lock found. Generating lock file...
❌ Lock file was missing! Please run 'uv lock' locally and commit uv.lock
```

**Solution:**
```bash
uv lock
git add uv.lock
git commit -m "chore: add uv.lock"
git push
```

### Issue: Lock file out of sync

**Error in CI:**
```
❌ Lock file is out of sync with pyproject.toml
```

**Solution:**
```bash
uv lock
git add uv.lock
git commit -m "chore: update uv.lock"
git push
```

### Issue: Tests failing

**Solution:**
```bash
# Run tests locally first
uv run pytest -v

# If tests pass locally but fail in CI, ensure dependencies match:
rm -rf .venv
uv sync --frozen
uv run pytest -v
```

### Issue: Pre-commit hook failures

**Solution:**
```bash
# Run hooks manually to see what's wrong
pre-commit run --all-files

# Fix issues and try again
git add .
git commit -m "fix: resolve pre-commit issues"
```

## Workflow Summary

### Daily Development

```bash
# 1. Create feature branch
git checkout -b feature/my-feature

# 2. Make changes
vim quantiq/my_module.py

# 3. Run tests locally
uv run pytest

# 4. Commit (pre-commit hooks run automatically)
git add .
git commit -m "feat: add my feature"

# 5. Push and create PR
git push origin feature/my-feature
```

### Adding Dependencies

```bash
# 1. Add to pyproject.toml
vim pyproject.toml

# 2. Update lock file
uv lock

# 3. Install
uv sync

# 4. Commit both files
git add pyproject.toml uv.lock
git commit -m "chore(deps): add new-package"
```

### Updating Dependencies

```bash
# Update all
uv lock --upgrade
uv sync

# Or update specific package
uv lock --upgrade-package numpy
uv sync

# Commit
git add uv.lock
git commit -m "chore(deps): update dependencies"
```

### Publishing to PyPI

```bash
# 1. Update version in pyproject.toml
vim pyproject.toml  # Change version = "0.1.0" to "0.2.0"

# 2. Commit version bump
git add pyproject.toml
git commit -m "chore: bump version to 0.2.0"
git push

# 3. Create and push tag
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin v0.2.0

# 4. GitHub Actions will automatically publish to PyPI
```

## Verification Checklist

After completing setup, verify:

- [ ] `uv.lock` exists and is committed
- [ ] Pre-commit hooks are installed (`pre-commit run --all-files` works)
- [ ] GitHub Actions workflow file exists (`.github/workflows/ci.yml`)
- [ ] Branch protection is enabled for `main`
- [ ] Dependabot is configured (`.github/dependabot.yml`)
- [ ] GitHub Pages is configured
- [ ] PyPI trusted publishing is set up (both Test and Production)
- [ ] GitHub environments created (`test-pypi`, `pypi`)
- [ ] Codecov token is set (optional)
- [ ] First PR shows all checks passing ✅

## Next Steps

1. Read the full documentation: [.github/CI_CD_SETUP.md](.github/CI_CD_SETUP.md)
2. Review the [GitHub Actions workflow](.github/workflows/ci.yml)
3. Customize the pipeline as needed
4. Set up additional integrations (Slack notifications, etc.)
5. Configure code coverage thresholds
6. Add custom deployment steps

## Support

For detailed information, see:
- [CI/CD Setup Documentation](.github/CI_CD_SETUP.md)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [uv Documentation](https://github.com/astral-sh/uv)

For issues, open a GitHub issue in this repository.

---

**Ready to get started?** Run these commands:

```bash
# Quick setup
uv lock
uv sync
pre-commit install
git add .
git commit -m "feat: complete CI/CD setup"
git push
```

Then follow the GitHub repository configuration steps above! 🚀
