# Makefile for quantiq - Modern JAX-Powered Framework for Measurement Data Science
# This Makefile provides comprehensive automation for development, testing, building, and deployment.
#
# Usage:
#   make help          Show all available targets
#   make init          Setup development environment
#   make test          Run tests
#   make qa            Run full quality assurance
#

.PHONY: help init install install-dev install-test install-docs install-gpu-cuda \
        format format-check lint type-check check \
        test test-fast test-cov test-slow test-gpu test-visual test-bench test-all coverage-html \
        qa pre-commit-install pre-commit-run security \
        build dist clean clean-all clean-venv \
        docs docs-serve docs-clean \
        publish-test publish \
        info version quick

# Configuration
PYTHON := python3
VENV := .venv
VENV_BIN := $(VENV)/bin
PIP := $(VENV_BIN)/pip
PYTEST := $(VENV_BIN)/pytest
PYTHON_VENV := $(VENV_BIN)/python
PROJECT := quantiq
SRC_DIR := quantiq
TEST_DIR := tests
DOCS_DIR := docs

# Colors for output
BOLD := \033[1m
RESET := \033[0m
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
CYAN := \033[36m

# Default target
.DEFAULT_GOAL := help

## help: Show this help message
help:
	@echo "$(BOLD)$(BLUE)quantiq Makefile$(RESET)"
	@echo ""
	@echo "$(BOLD)Usage:$(RESET) make $(CYAN)<target>$(RESET)"
	@echo ""
	@echo "$(BOLD)$(GREEN)SETUP & INSTALLATION$(RESET)"
	@echo "  $(CYAN)init$(RESET)              Create venv and install all dev dependencies"
	@echo "  $(CYAN)install$(RESET)           Install package in editable mode"
	@echo "  $(CYAN)install-dev$(RESET)       Install with dev extras"
	@echo "  $(CYAN)install-test$(RESET)      Install with test extras"
	@echo "  $(CYAN)install-docs$(RESET)      Install with docs extras"
	@echo "  $(CYAN)install-gpu-cuda$(RESET)  Install with CUDA GPU support (Linux only)"
	@echo ""
	@echo "$(BOLD)$(GREEN)DEVELOPMENT$(RESET)"
	@echo "  $(CYAN)format$(RESET)            Auto-format code with ruff"
	@echo "  $(CYAN)format-check$(RESET)      Check code formatting without changes"
	@echo "  $(CYAN)lint$(RESET)              Run ruff linter"
	@echo "  $(CYAN)type-check$(RESET)        Run mypy type checker"
	@echo "  $(CYAN)check$(RESET)             Run all checks (format + lint + type)"
	@echo "  $(CYAN)quick$(RESET)             Fast iteration: format + test-fast"
	@echo ""
	@echo "$(BOLD)$(GREEN)TESTING$(RESET)"
	@echo "  $(CYAN)test$(RESET)              Run basic tests (fast, no GPU)"
	@echo "  $(CYAN)test-fast$(RESET)         Run only fast tests"
	@echo "  $(CYAN)test-cov$(RESET)          Run tests with coverage report"
	@echo "  $(CYAN)test-slow$(RESET)         Include slow tests"
	@echo "  $(CYAN)test-gpu$(RESET)          Run GPU tests only"
	@echo "  $(CYAN)test-visual$(RESET)       Run visual regression tests"
	@echo "  $(CYAN)test-bench$(RESET)        Run performance benchmarks"
	@echo "  $(CYAN)test-all$(RESET)          Run all tests (including slow, GPU, visual)"
	@echo "  $(CYAN)coverage-html$(RESET)     Open HTML coverage report in browser"
	@echo ""
	@echo "$(BOLD)$(GREEN)QUALITY ASSURANCE$(RESET)"
	@echo "  $(CYAN)qa$(RESET)                Full quality gate: check + test-cov"
	@echo "  $(CYAN)pre-commit-install$(RESET) Install pre-commit hooks"
	@echo "  $(CYAN)pre-commit-run$(RESET)    Run pre-commit on all files"
	@echo "  $(CYAN)security$(RESET)          Run security audit (requires pip-audit)"
	@echo ""
	@echo "$(BOLD)$(GREEN)BUILD & DISTRIBUTION$(RESET)"
	@echo "  $(CYAN)build$(RESET)             Build wheel and source distribution"
	@echo "  $(CYAN)dist$(RESET)              Alias for build"
	@echo "  $(CYAN)clean$(RESET)             Remove build artifacts and caches (preserves venv, .claude, .specify, agent-os)"
	@echo "  $(CYAN)clean-all$(RESET)         Deep clean of all caches (preserves venv, .claude, .specify, agent-os)"
	@echo "  $(CYAN)clean-venv$(RESET)        Remove virtual environment (use with caution)"
	@echo ""
	@echo "$(BOLD)$(GREEN)DOCUMENTATION$(RESET)"
	@echo "  $(CYAN)docs$(RESET)              Build HTML documentation with Sphinx"
	@echo "  $(CYAN)docs-serve$(RESET)        Serve documentation locally (port 8000)"
	@echo "  $(CYAN)docs-clean$(RESET)        Clean documentation build"
	@echo ""
	@echo "$(BOLD)$(GREEN)RELEASE$(RESET)"
	@echo "  $(CYAN)publish-test$(RESET)      Publish to TestPyPI"
	@echo "  $(CYAN)publish$(RESET)           Publish to PyPI (requires confirmation)"
	@echo ""
	@echo "$(BOLD)$(GREEN)UTILITY$(RESET)"
	@echo "  $(CYAN)info$(RESET)              Show project and environment info"
	@echo "  $(CYAN)version$(RESET)           Show package version"
	@echo "  $(CYAN)help$(RESET)              Show this help message"

# ============================================================================
# SETUP & INSTALLATION
# ============================================================================

## init: Create virtual environment and install all dev dependencies
init:
	@echo "$(BOLD)$(BLUE)Creating virtual environment...$(RESET)"
	uv venv $(VENV)
	@echo "$(BOLD)$(BLUE)Installing package with dev dependencies...$(RESET)"
	uv sync --extra dev --extra test --extra docs
	@echo "$(BOLD)$(GREEN)✓ Development environment ready!$(RESET)"
	@echo ""
	@echo "$(BOLD)Activate with:$(RESET) source $(VENV)/bin/activate"

## install: Install package in editable mode
install:
	@if [ ! -d "$(VENV)" ]; then \
		echo "$(BOLD)$(RED)Error: Virtual environment not found. Run 'make init' first.$(RESET)"; \
		exit 1; \
	fi
	@echo "$(BOLD)$(BLUE)Installing $(PROJECT) in editable mode...$(RESET)"
	uv sync
	@echo "$(BOLD)$(GREEN)✓ Package installed!$(RESET)"

## install-dev: Install with dev extras
install-dev:
	@if [ ! -d "$(VENV)" ]; then \
		echo "$(BOLD)$(RED)Error: Virtual environment not found. Run 'make init' first.$(RESET)"; \
		exit 1; \
	fi
	@echo "$(BOLD)$(BLUE)Installing dev dependencies...$(RESET)"
	uv sync --extra dev
	@echo "$(BOLD)$(GREEN)✓ Dev dependencies installed!$(RESET)"

## install-test: Install with test extras
install-test:
	@if [ ! -d "$(VENV)" ]; then \
		echo "$(BOLD)$(RED)Error: Virtual environment not found. Run 'make init' first.$(RESET)"; \
		exit 1; \
	fi
	@echo "$(BOLD)$(BLUE)Installing test dependencies...$(RESET)"
	uv sync --extra test
	@echo "$(BOLD)$(GREEN)✓ Test dependencies installed!$(RESET)"

## install-docs: Install with docs extras
install-docs:
	@if [ ! -d "$(VENV)" ]; then \
		echo "$(BOLD)$(RED)Error: Virtual environment not found. Run 'make init' first.$(RESET)"; \
		exit 1; \
	fi
	@echo "$(BOLD)$(BLUE)Installing docs dependencies...$(RESET)"
	uv sync --extra docs
	@echo "$(BOLD)$(GREEN)✓ Docs dependencies installed!$(RESET)"

## install-gpu-cuda: Install with CUDA GPU support (Linux only)
## This uninstalls CPU-only JAX, installs GPU-enabled JAX with CUDA 12, and verifies GPU detection
install-gpu-cuda:
	@# Validate platform (GPU support requires Linux)
	@if [ "$$(uname -s)" != "Linux" ]; then \
		echo "$(BOLD)$(RED)Error: GPU support requires Linux. Current platform: $$(uname -s)$(RESET)"; \
		echo "macOS and Windows only support CPU backend (5-10x speedup over piblin)."; \
		echo "For maximum performance (50-100x), use Linux with NVIDIA GPU."; \
		exit 1; \
	fi
	@# Validate virtual environment
	@if [ ! -d "$(VENV)" ]; then \
		echo "$(BOLD)$(RED)Error: Virtual environment not found. Run 'make init' first.$(RESET)"; \
		exit 1; \
	fi
	@echo "$(BOLD)$(BLUE)Installing CUDA GPU support...$(RESET)"
	@echo "  $(BOLD)1/3$(RESET) Uninstalling CPU-only JAX..."
	@uv pip uninstall -y jax jaxlib 2>/dev/null || true
	@echo "  $(BOLD)2/3$(RESET) Installing GPU-enabled JAX with CUDA 12 support..."
	@uv sync --extra gpu-cuda
	@echo "  $(BOLD)3/3$(RESET) Verifying GPU detection..."
	@uv run python -c "from quantiq.backend import get_device_info; info = get_device_info(); print(f'  Backend: {info[\"backend\"]}'); print(f'  Device: {info[\"device_type\"]}'); assert info['device_type'] == 'gpu', 'GPU not detected!'" && \
		echo "$(BOLD)$(GREEN)✓ GPU support verified!$(RESET)" || \
		(echo "$(BOLD)$(RED)✗ GPU not detected. Check your CUDA installation.$(RESET)" && \
		echo "  Requirements: Linux with CUDA 12+ and compatible NVIDIA GPU" && exit 1)

# ============================================================================
# DEVELOPMENT
# ============================================================================

## format: Auto-format code with ruff
format:
	@echo "$(BOLD)$(BLUE)Formatting code with ruff...$(RESET)"
	uv run ruff check --fix $(SRC_DIR) $(TEST_DIR) examples/
	uv run ruff format $(SRC_DIR) $(TEST_DIR) examples/
	@echo "$(BOLD)$(GREEN)✓ Code formatted!$(RESET)"

## format-check: Check code formatting without changes
format-check:
	@echo "$(BOLD)$(BLUE)Checking code formatting...$(RESET)"
	uv run ruff check $(SRC_DIR) $(TEST_DIR) examples/
	uv run ruff format --check $(SRC_DIR) $(TEST_DIR) examples/
	@echo "$(BOLD)$(GREEN)✓ Code formatting is correct!$(RESET)"

## lint: Run ruff linter
lint:
	@echo "$(BOLD)$(BLUE)Running ruff linter...$(RESET)"
	uv run ruff check $(SRC_DIR) $(TEST_DIR) examples/
	@echo "$(BOLD)$(GREEN)✓ No linting errors!$(RESET)"

## type-check: Run mypy type checker
type-check:
	@echo "$(BOLD)$(BLUE)Running mypy type checker...$(RESET)"
	uv run mypy $(SRC_DIR)
	@echo "$(BOLD)$(GREEN)✓ Type checking passed!$(RESET)"

## check: Run all checks (format + lint + type)
check: format-check lint type-check
	@echo "$(BOLD)$(GREEN)✓ All checks passed!$(RESET)"

## quick: Fast iteration - format and run fast tests
quick: format test-fast
	@echo "$(BOLD)$(GREEN)✓ Quick iteration complete!$(RESET)"

# ============================================================================
# TESTING
# ============================================================================

## test: Run basic tests (fast, no GPU)
test:
	@echo "$(BOLD)$(BLUE)Running tests...$(RESET)"
	uv run pytest -m "not slow and not gpu" --tb=short

## test-fast: Run only fast tests
test-fast:
	@echo "$(BOLD)$(BLUE)Running fast tests...$(RESET)"
	uv run pytest -m "not slow and not gpu and not benchmark" --tb=short -q

## test-cov: Run tests with coverage report
test-cov:
	@echo "$(BOLD)$(BLUE)Running tests with coverage...$(RESET)"
	uv run pytest -m "not slow and not gpu" --cov=$(SRC_DIR) --cov-report=term-missing --cov-report=html --cov-report=xml
	@echo "$(BOLD)$(GREEN)✓ Coverage report generated!$(RESET)"
	@echo "View HTML report: open htmlcov/index.html"

## test-slow: Include slow tests
test-slow:
	@echo "$(BOLD)$(BLUE)Running all tests (including slow)...$(RESET)"
	uv run pytest -m "not gpu" --tb=short

## test-gpu: Run GPU tests only
test-gpu:
	@echo "$(BOLD)$(BLUE)Running GPU tests...$(RESET)"
	uv run pytest -m "gpu" --tb=short

## test-visual: Run visual regression tests
test-visual:
	@echo "$(BOLD)$(BLUE)Running visual regression tests...$(RESET)"
	uv run pytest -m "visual" --tb=short

## test-bench: Run performance benchmarks
test-bench:
	@echo "$(BOLD)$(BLUE)Running benchmarks...$(RESET)"
	uv run pytest -m "benchmark" --benchmark-only

## test-all: Run all tests (including slow, GPU, visual)
test-all:
	@echo "$(BOLD)$(BLUE)Running ALL tests...$(RESET)"
	uv run pytest --tb=short

## coverage-html: Open HTML coverage report in browser
coverage-html:
	@if [ -f "htmlcov/index.html" ]; then \
		echo "$(BOLD)$(BLUE)Opening coverage report...$(RESET)"; \
		open htmlcov/index.html || xdg-open htmlcov/index.html || echo "Please open htmlcov/index.html manually"; \
	else \
		echo "$(BOLD)$(RED)Error: Coverage report not found. Run 'make test-cov' first.$(RESET)"; \
		exit 1; \
	fi

# ============================================================================
# QUALITY ASSURANCE
# ============================================================================

## qa: Full quality gate - check + test-cov
qa: check test-cov
	@echo "$(BOLD)$(GREEN)✓ Quality assurance complete!$(RESET)"

## pre-commit-install: Install pre-commit hooks
pre-commit-install:
	@echo "$(BOLD)$(BLUE)Installing pre-commit hooks...$(RESET)"
	uv run pre-commit install
	@echo "$(BOLD)$(GREEN)✓ Pre-commit hooks installed!$(RESET)"

## pre-commit-run: Run pre-commit on all files
pre-commit-run:
	@echo "$(BOLD)$(BLUE)Running pre-commit on all files...$(RESET)"
	uv run pre-commit run --all-files

## security: Run security audit (requires pip-audit)
security:
	@echo "$(BOLD)$(BLUE)Running security audit...$(RESET)"
	uv pip freeze > requirements-audit.txt
	uv run pip-audit -r requirements-audit.txt --desc || echo "$(BOLD)$(YELLOW)Security audit completed with findings$(RESET)"
	rm -f requirements-audit.txt

# ============================================================================
# BUILD & DISTRIBUTION
# ============================================================================

## build: Build wheel and source distribution
build: clean
	@echo "$(BOLD)$(BLUE)Building distributions...$(RESET)"
	uv build
	@echo "$(BOLD)$(GREEN)✓ Build complete!$(RESET)"
	@echo "Distributions in dist/"

## dist: Alias for build
dist: build

## clean: Remove build artifacts and caches (preserves venv, .claude, .specify, agent-os)
clean:
	@echo "$(BOLD)$(BLUE)Cleaning build artifacts...$(RESET)"
	rm -rf build/
	rm -rf dist/
	rm -rf .benchmarks/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf coverage.xml
	rm -rf coverage.json
	rm -rf .hypothesis
	rm -rf .nlsq_cache/
	rm -rf .ruff_cache/
	rm -f test_results.log
	@echo "$(BOLD)$(BLUE)Removing __pycache__ directories and .pyc files...$(RESET)"
	find . -type d -name __pycache__ \
		-not -path "./.venv/*" \
		-not -path "./.claude/*" \
		-not -path "./.specify/*" \
		-not -path "./agent-os/*" \
		-exec rm -rf {} + 2>/dev/null || true
	find . -type f \( -name "*.pyc" -o -name "*.pyo" \) \
		-not -path "./.venv/*" \
		-not -path "./.claude/*" \
		-not -path "./.specify/*" \
		-not -path "./agent-os/*" \
		-delete 2>/dev/null || true
	@echo "$(BOLD)$(GREEN)✓ Cleaned!$(RESET)"
	@echo "$(BOLD)Protected directories preserved:$(RESET) .venv/, .claude/, .specify/, agent-os/"

## clean-all: Deep clean of all caches (preserves .venv, .claude, .specify, agent-os)
clean-all: clean
	@echo "$(BOLD)$(BLUE)Performing deep clean of additional caches...$(RESET)"
	rm -rf .tox/ 2>/dev/null || true
	rm -rf .nox/ 2>/dev/null || true
	rm -rf .eggs/ 2>/dev/null || true
	rm -rf .cache/ 2>/dev/null || true
	rm -rf .ruff_cache/ 2>/dev/null || true
	rm -rf .pytest_cache/ 2>/dev/null || true
	rm -rf node_modules/ 2>/dev/null || true
	rm -rf .benchmarks/ 2>/dev/null || true
	rm -rf .nlsq_cache/ 2>/dev/null || true
	find . -type d -name "*.egg-info" \
		-not -path "./.venv/*" \
		-not -path "./.claude/*" \
		-not -path "./.specify/*" \
		-not -path "./agent-os/*" \
		-exec rm -rf {} + 2>/dev/null || true
	@echo "$(BOLD)$(GREEN)✓ Deep clean complete!$(RESET)"
	@echo "$(BOLD)Protected directories preserved:$(RESET) .venv/, .claude/, .specify/, agent-os/"

## clean-venv: Remove virtual environment (use with caution)
clean-venv:
	@echo "$(BOLD)$(YELLOW)WARNING: This will remove the virtual environment!$(RESET)"
	@echo "$(BOLD)You will need to run 'make init' to recreate it.$(RESET)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo "$(BOLD)$(BLUE)Removing virtual environment...$(RESET)"; \
		rm -rf $(VENV); \
		echo "$(BOLD)$(GREEN)✓ Virtual environment removed!$(RESET)"; \
		echo "$(BOLD)Run 'make init' to recreate the environment.$(RESET)"; \
	else \
		echo "Cancelled."; \
	fi

# ============================================================================
# DOCUMENTATION
# ============================================================================

## docs: Build HTML documentation with Sphinx
docs:
	@echo "$(BOLD)$(BLUE)Building documentation...$(RESET)"
	$(MAKE) -C $(DOCS_DIR) html
	@echo "$(BOLD)$(GREEN)✓ Documentation built!$(RESET)"
	@echo "Open: $(DOCS_DIR)/_build/html/index.html"

## docs-serve: Serve documentation locally (port 8000)
docs-serve: docs
	@echo "$(BOLD)$(BLUE)Serving documentation on http://localhost:8000$(RESET)"
	@echo "Press Ctrl+C to stop"
	cd $(DOCS_DIR)/_build/html && $(PYTHON) -m http.server 8000

## docs-clean: Clean documentation build
docs-clean:
	@echo "$(BOLD)$(BLUE)Cleaning documentation...$(RESET)"
	$(MAKE) -C $(DOCS_DIR) clean
	@echo "$(BOLD)$(GREEN)✓ Documentation cleaned!$(RESET)"

# ============================================================================
# RELEASE
# ============================================================================

## publish-test: Publish to TestPyPI
publish-test: build
	@echo "$(BOLD)$(BLUE)Publishing to TestPyPI...$(RESET)"
	uv run twine upload --repository testpypi dist/*
	@echo "$(BOLD)$(GREEN)✓ Published to TestPyPI!$(RESET)"

## publish: Publish to PyPI (requires confirmation)
publish: build
	@echo "$(BOLD)$(YELLOW)This will publish $(PROJECT) to PyPI!$(RESET)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo "$(BOLD)$(BLUE)Publishing to PyPI...$(RESET)"; \
		uv run twine upload dist/*; \
		echo "$(BOLD)$(GREEN)✓ Published to PyPI!$(RESET)"; \
	else \
		echo "Cancelled."; \
	fi

# ============================================================================
# UTILITY
# ============================================================================

## info: Show project and environment info
info:
	@echo "$(BOLD)$(BLUE)Project Information$(RESET)"
	@echo "===================="
	@echo "Project: $(PROJECT)"
	@echo "Python: $$($(PYTHON) --version 2>&1)"
	@if [ -d "$(VENV)" ]; then \
		echo "Venv: $(VENV) (active)"; \
		echo "Pip: $$($(PIP) --version)"; \
	else \
		echo "Venv: Not created"; \
	fi
	@echo ""
	@if [ -d "$(VENV)" ] && [ -f "$(PYTHON_VENV)" ]; then \
		echo "$(BOLD)$(BLUE)JAX Configuration$(RESET)"; \
		echo "=================="; \
		$(PYTHON_VENV) -c "import jax; print('JAX version:', jax.__version__); print('Default backend:', jax.default_backend())" 2>/dev/null || echo "JAX not installed"; \
	fi
	@echo ""
	@echo "$(BOLD)$(BLUE)Directory Structure$(RESET)"
	@echo "===================="
	@echo "Source: $(SRC_DIR)/"
	@echo "Tests: $(TEST_DIR)/"
	@echo "Docs: $(DOCS_DIR)/"

## version: Show package version
version:
	@if [ -d "$(VENV)" ]; then \
		$(PYTHON_VENV) -c "import $(PROJECT); print($(PROJECT).__version__)" 2>/dev/null || \
		echo "Version info not available (package not installed)"; \
	else \
		echo "$(BOLD)$(RED)Error: Virtual environment not found. Run 'make init' first.$(RESET)"; \
	fi
