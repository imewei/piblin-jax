#!/bin/bash
#
# CI/CD Setup Validation Script
# Validates that the CI/CD pipeline is properly configured
#
# Usage: bash .github/scripts/validate-setup.sh

set -e

echo "🔍 CI/CD Setup Validation"
echo "========================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

success_count=0
warning_count=0
error_count=0

check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $1 exists"
        ((success_count++))
        return 0
    else
        echo -e "${RED}✗${NC} $1 missing"
        ((error_count++))
        return 1
    fi
}

check_command() {
    if command -v "$1" &> /dev/null; then
        echo -e "${GREEN}✓${NC} $1 is installed"
        ((success_count++))
        return 0
    else
        echo -e "${YELLOW}⚠${NC} $1 not found (optional)"
        ((warning_count++))
        return 1
    fi
}

# Check required files
echo "📁 Checking Required Files"
echo "--------------------------"
check_file "uv.lock"
check_file "pyproject.toml"
check_file ".github/workflows/ci.yml"
check_file ".github/workflows/security.yml"
check_file ".github/workflows/publish.yml"
check_file ".github/dependabot.yml"
check_file ".github/PULL_REQUEST_TEMPLATE.md"
check_file ".github/workflows/README.md"
check_file ".pre-commit-config.yaml"
echo ""

# Check commands
echo "🔧 Checking Required Tools"
echo "--------------------------"
check_command "uv"
check_command "git"
check_command "python3"
check_command "pre-commit"
echo ""

# Validate uv.lock
echo "🔒 Validating Lock File"
echo "----------------------"
if [ -f "uv.lock" ]; then
    if uv lock --locked &> /dev/null; then
        echo -e "${GREEN}✓${NC} uv.lock is in sync with pyproject.toml"
        ((success_count++))
    else
        echo -e "${RED}✗${NC} uv.lock is out of sync! Run: uv lock"
        ((error_count++))
    fi
else
    echo -e "${RED}✗${NC} uv.lock not found"
    ((error_count++))
fi
echo ""

# Check workflow syntax
echo "📝 Validating Workflow Syntax"
echo "-----------------------------"
for workflow in .github/workflows/*.yml; do
    if [ -f "$workflow" ]; then
        # Basic YAML validation
        if python3 -c "import yaml; yaml.safe_load(open('$workflow'))" 2> /dev/null; then
            echo -e "${GREEN}✓${NC} $(basename $workflow) is valid YAML"
            ((success_count++))
        else
            echo -e "${RED}✗${NC} $(basename $workflow) has YAML errors"
            ((error_count++))
        fi
    fi
done
echo ""

# Check pre-commit hooks
echo "🪝 Checking Pre-commit Configuration"
echo "------------------------------------"
if [ -f ".pre-commit-config.yaml" ]; then
    if command -v pre-commit &> /dev/null; then
        if pre-commit validate-config &> /dev/null; then
            echo -e "${GREEN}✓${NC} Pre-commit config is valid"
            ((success_count++))
        else
            echo -e "${RED}✗${NC} Pre-commit config has errors"
            ((error_count++))
        fi

        # Check if hooks are installed
        if [ -f ".git/hooks/pre-commit" ]; then
            echo -e "${GREEN}✓${NC} Pre-commit hooks are installed"
            ((success_count++))
        else
            echo -e "${YELLOW}⚠${NC} Pre-commit hooks not installed. Run: pre-commit install"
            ((warning_count++))
        fi
    fi
fi
echo ""

# Check git configuration
echo "🔄 Checking Git Configuration"
echo "-----------------------------"
if git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Git repository detected"
    ((success_count++))

    # Check remote
    if git remote get-url origin &> /dev/null; then
        remote_url=$(git remote get-url origin)
        echo -e "${GREEN}✓${NC} Remote origin configured: $remote_url"
        ((success_count++))
    else
        echo -e "${YELLOW}⚠${NC} No remote origin configured"
        ((warning_count++))
    fi
else
    echo -e "${RED}✗${NC} Not a git repository"
    ((error_count++))
fi
echo ""

# Check dependencies
echo "📦 Checking Dependencies"
echo "-----------------------"
if command -v uv &> /dev/null; then
    dep_count=$(grep -c "^[[package]]" uv.lock 2>/dev/null || echo "0")
    echo -e "${GREEN}✓${NC} $dep_count packages in lock file"
    ((success_count++))
fi
echo ""

# Summary
echo "📊 Validation Summary"
echo "===================="
echo -e "${GREEN}✓${NC} Passed:   $success_count"
echo -e "${YELLOW}⚠${NC} Warnings: $warning_count"
echo -e "${RED}✗${NC} Failed:   $error_count"
echo ""

if [ $error_count -eq 0 ]; then
    echo -e "${GREEN}✅ CI/CD setup is valid!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Commit changes: git commit -m 'ci: add comprehensive CI/CD pipeline'"
    echo "2. Push to GitHub: git push origin main"
    echo "3. Enable branch protection in GitHub Settings"
    echo "4. Add secrets (CODECOV_TOKEN, etc.) in GitHub Settings"
    exit 0
else
    echo -e "${RED}❌ CI/CD setup has errors. Please fix them before proceeding.${NC}"
    exit 1
fi
