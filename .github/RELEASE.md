# Release Process Guide

This document describes how to release a new version of quantiq to PyPI.

## Overview

The release process is fully automated via GitHub Actions. When you push a version tag (e.g., `v0.0.1`), the workflow:

1. Validates the release (tag format, version consistency)
2. Runs the full test suite
3. Builds distribution packages (wheel + sdist)
4. Publishes to TestPyPI (for testing)
5. Publishes to PyPI (production)
6. Creates a GitHub Release with artifacts

## Prerequisites

### 1. PyPI Trusted Publishing Setup

**Required**: Configure trusted publishing for secure, token-free releases.

#### For TestPyPI:

1. Go to https://test.pypi.org/manage/account/publishing/
2. Add a new publisher:
   - **PyPI Project Name**: `quantiq`
   - **Owner**: `imewei` (or your org)
   - **Repository name**: `quantiq` (or your repo)
   - **Workflow name**: `release.yml`
   - **Environment name**: `testpypi`

#### For Production PyPI:

1. Go to https://pypi.org/manage/account/publishing/
2. Add a new publisher:
   - **PyPI Project Name**: `quantiq`
   - **Owner**: `imewei` (or your org)
   - **Repository name**: `quantiq` (or your repo)
   - **Workflow name**: `release.yml`
   - **Environment name**: `pypi`

### 2. GitHub Environments

Create two protected environments in your repository settings:

#### TestPyPI Environment:
- Name: `testpypi`
- Deployment protection: None (optional: require reviewers)
- Environment secrets: None needed (uses OIDC)

#### PyPI Environment:
- Name: `pypi`
- Deployment protection: **Required reviewers** (recommended)
- Reviewers: Add yourself or team leads
- Environment secrets: None needed (uses OIDC)

**Why protected environments?**
- Prevents accidental releases
- Requires manual approval before publishing to production PyPI
- Provides audit trail of who approved releases

## Release Steps

### Step 1: Prepare the Release

1. **Update version numbers** consistently across:
   - `pyproject.toml` (line 7): `version = "0.0.1"`
   - `quantiq/__init__.py` (line 25): `__version__ = "0.0.1"`
   - `docs/source/conf.py` (lines 18-19): `version` and `release`
   - `tests/test_package.py` (line 10): version assertion

2. **Update CHANGELOG.md**:
   ```markdown
   ## [0.0.1] - YYYY-MM-DD

   ### Added
   - Feature 1
   - Feature 2

   ### Fixed
   - Bug fix 1
   ```

3. **Update CLAUDE.md** if needed (breaking changes section)

4. **Run local tests**:
   ```bash
   make test-cov
   make qa
   ```

5. **Build locally to verify**:
   ```bash
   uv build
   ls -lh dist/
   ```

### Step 2: Commit and Push

```bash
git add .
git commit -m "chore: prepare release v0.0.1"
git push origin main
```

Wait for CI to pass on the commit.

### Step 3: Create and Push Tag

```bash
# Create annotated tag
git tag -a v0.0.1 -m "Release v0.0.1 - First pre-release

Key features:
- Feature 1
- Feature 2
- 97.14% test coverage
- 100% mypy strict compliance"

# Push tag to trigger release
git push origin v0.0.1
```

### Step 4: Monitor Release Workflow

1. Go to: https://github.com/imewei/quantiq/actions
2. Watch the "Release to PyPI" workflow
3. The workflow will:
   - ✅ Validate version consistency
   - ✅ Run full test suite
   - ✅ Build packages
   - ✅ Publish to TestPyPI
   - ⏸️ **Wait for approval** (if environment protection enabled)
   - ✅ Publish to PyPI (after approval)
   - ✅ Create GitHub Release

### Step 5: Approve PyPI Deployment

If you set up environment protection (recommended):

1. GitHub will pause at "Publish to PyPI" job
2. You'll receive a notification to review the deployment
3. Click "Review deployments" → Approve
4. Workflow continues and publishes to PyPI

### Step 6: Verify Release

1. **Check TestPyPI**:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ quantiq==0.0.1
   python -c "import quantiq; print(quantiq.__version__)"
   ```

2. **Check PyPI** (after approval):
   ```bash
   pip install quantiq-jax==0.0.1
   python -c "import quantiq; print(quantiq.__version__)"
   ```

3. **Check GitHub Release**:
   - Go to: https://github.com/imewei/quantiq/releases
   - Verify release notes and artifacts

## Manual Release (Fallback)

If automated release fails, you can release manually:

### Using GitHub Actions Workflow Dispatch:

1. Go to Actions → "Release to PyPI" → "Run workflow"
2. Enter tag: `v0.0.1`
3. Click "Run workflow"

### Using uv Directly:

```bash
# Build
uv build

# Publish to TestPyPI
uv publish --repository testpypi

# Publish to PyPI
uv publish
```

## Release Checklist

- [ ] All tests passing on main branch
- [ ] Version updated in all files (pyproject.toml, __init__.py, docs, tests)
- [ ] CHANGELOG.md updated with release notes
- [ ] CLAUDE.md updated if breaking changes
- [ ] Committed and pushed to main
- [ ] CI passed on latest commit
- [ ] Tag created and pushed
- [ ] Release workflow triggered
- [ ] TestPyPI upload successful
- [ ] PyPI deployment approved
- [ ] PyPI upload successful
- [ ] GitHub Release created
- [ ] Verified installation from PyPI
- [ ] Announced release (if applicable)

## Troubleshooting

### Version Mismatch Error

**Error**: "Version mismatch: tag=0.0.1, pyproject.toml=0.1.0"

**Solution**: Ensure all version numbers match:
```bash
grep -r "0.1.0" pyproject.toml quantiq/__init__.py docs/source/conf.py tests/test_package.py
```

### Trusted Publishing Error

**Error**: "Trusted publishing exchange failure"

**Solution**:
1. Verify publisher configuration on PyPI
2. Ensure environment names match exactly
3. Check workflow file name is correct

### Build Artifacts Missing

**Error**: "Missing build artifacts"

**Solution**:
```bash
# Clean and rebuild
make clean-all
uv build
ls -lh dist/
```

### Tag Already Exists

**Error**: "tag 'v0.0.1' already exists"

**Solution**:
```bash
# Delete local and remote tag
git tag -d v0.0.1
git push origin :refs/tags/v0.0.1

# Recreate tag
git tag -a v0.0.1 -m "Release v0.0.1"
git push origin v0.0.1
```

## Versioning Guidelines

Follow [Semantic Versioning](https://semver.org/):

- **Major** (X.0.0): Breaking changes
- **Minor** (0.X.0): New features, backward compatible
- **Patch** (0.0.X): Bug fixes, backward compatible

### Pre-release Versions:

- `v0.0.1-alpha.1`: Alpha release
- `v0.0.1-beta.1`: Beta release
- `v0.0.1-rc.1`: Release candidate

Pre-releases are automatically detected and marked as "pre-release" on GitHub.

## Security

### Trusted Publishing (OIDC)

We use PyPI's trusted publishing with OpenID Connect (OIDC):

✅ **Advantages**:
- No API tokens to manage
- Tokens are short-lived (15 minutes)
- Automatically rotated
- Scoped to specific workflow + environment
- No secret storage in GitHub

❌ **Never**:
- Store PyPI tokens in repository secrets
- Commit API tokens to version control
- Share PyPI credentials

### Environment Protection

Production PyPI environment should require:
- Manual approval from repository maintainers
- Review of changes before deployment
- Audit trail of who approved releases

## Additional Resources

- PyPI Trusted Publishing: https://docs.pypi.org/trusted-publishers/
- GitHub Environments: https://docs.github.com/en/actions/deployment/targeting-different-environments
- Semantic Versioning: https://semver.org/
- Python Packaging Guide: https://packaging.python.org/
