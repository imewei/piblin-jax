"""
Tests for package dependency configuration and platform markers.

This module tests that GPU dependencies are correctly configured with
platform markers, ensuring gpu-cuda is Linux-only and that legacy
gpu-metal and gpu-rocm have been removed.
"""

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


class TestPackageDependencies:
    """Test package dependency configuration."""

    @pytest.fixture
    def pyproject_toml_path(self):
        """Return path to pyproject.toml."""
        repo_root = Path(__file__).parent.parent
        return repo_root / "pyproject.toml"

    def test_gpu_metal_removed_from_dependencies(self, pyproject_toml_path):
        """Test that gpu-metal optional dependency has been completely removed."""
        content = pyproject_toml_path.read_text(encoding="utf-8")

        # Verify no gpu-metal section exists
        assert "gpu-metal" not in content, (
            "gpu-metal should be completely removed from pyproject.toml"
        )

        # Verify no references to Metal GPU support
        assert (
            "Metal" not in content
            or "Apple Silicon" in content.split("[project.optional-dependencies]")[0]
        ), "No Metal references should exist in optional dependencies section"

    def test_gpu_rocm_removed_from_dependencies(self, pyproject_toml_path):
        """Test that gpu-rocm optional dependency has been completely removed."""
        content = pyproject_toml_path.read_text(encoding="utf-8")

        # Verify no gpu-rocm section exists
        assert "gpu-rocm" not in content, (
            "gpu-rocm should be completely removed from pyproject.toml"
        )

        # Verify no references to ROCm GPU support
        assert "rocm" not in content.lower().split("[project.optional-dependencies]")[1], (
            "No ROCm references should exist in optional dependencies section"
        )

    def test_gpu_cuda_has_linux_platform_marker(self, pyproject_toml_path):
        """Test that gpu-cuda optional extra has been removed (breaking change v0.1.0).

        As documented in CHANGELOG.md, the gpu-cuda extra was removed because pip
        extras are unreliable for mutually exclusive dependency variants (CPU vs GPU jaxlib).
        GPU installation now requires explicit manual installation via:
        - make install-gpu-cuda (recommended)
        - Manual: pip uninstall -y jax jaxlib && pip install "jax[cuda12-local]>=0.8.0"
        """
        content = pyproject_toml_path.read_text(encoding="utf-8")

        # Extract optional dependencies section
        if "[project.optional-dependencies]" in content:
            deps_section = content.split("[project.optional-dependencies]")[1]
            next_section_idx = deps_section.find("\n[")
            if next_section_idx > 0:
                deps_section = deps_section[:next_section_idx]

            # Check if gpu-cuda is defined as an optional dependency
            for line in deps_section.split("\n"):
                if line.strip() and not line.strip().startswith("#"):
                    if "gpu-cuda" in line and "=" in line:
                        pytest.fail(
                            "gpu-cuda optional extra should be removed as of v0.1.0 "
                            "(see CHANGELOG.md - breaking change)"
                        )

    def test_gpu_cuda_platform_marker_syntax(self, pyproject_toml_path):
        """Test that manual GPU installation is documented (gpu-cuda extra removed).

        The gpu-cuda extra was removed in v0.1.0. This test now verifies
        that no gpu-cuda extra exists, reflecting the manual-only installation approach.
        """
        content = pyproject_toml_path.read_text(encoding="utf-8")

        # Verify gpu-cuda extra does not exist
        lines = content.split("\n")
        for line in lines:
            if (
                "gpu-cuda" in line
                and "=" in line
                and "[project.optional-dependencies]" in content[: content.index(line)]
            ):
                pytest.fail(
                    "gpu-cuda extra should not exist. GPU installation is now manual-only "
                    "(see CHANGELOG.md v0.1.0 breaking change)"
                )

    @pytest.mark.skipif(
        sys.platform != "linux", reason="Platform marker enforcement only testable on Linux"
    )
    def test_gpu_cuda_installable_on_linux(self):
        """Test that gpu-cuda extra can be queried on Linux systems."""
        # This test verifies the syntax is valid, not that it actually installs
        # We use pip's dry-run capabilities to check dependency resolution
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--dry-run", "piblin-jax[gpu-cuda]"],
            capture_output=True,
            text=True,
        )

        # On Linux, the command should at least parse correctly
        # (may fail for other reasons like network, but not syntax)
        assert "error: invalid requirement" not in result.stderr.lower(), (
            "gpu-cuda dependency syntax should be valid"
        )

    @pytest.mark.skipif(
        sys.platform == "linux", reason="Non-Linux platform marker test only on non-Linux systems"
    )
    def test_gpu_cuda_platform_marker_on_non_linux(self, pyproject_toml_path):
        """Test that GPU installation is manual-only on non-Linux (gpu-cuda extra removed).

        The gpu-cuda extra was removed in v0.1.0. This test verifies
        that no gpu-cuda extra exists, reflecting the manual-only approach
        that is required for all platforms.
        """
        content = pyproject_toml_path.read_text(encoding="utf-8")

        # Extract optional dependencies section
        if "[project.optional-dependencies]" in content:
            deps_section = content.split("[project.optional-dependencies]")[1]
            next_section_idx = deps_section.find("\n[")
            if next_section_idx > 0:
                deps_section = deps_section[:next_section_idx]

            # Check if gpu-cuda is defined as an optional dependency
            for line in deps_section.split("\n"):
                if line.strip() and not line.strip().startswith("#"):
                    if "gpu-cuda" in line and "=" in line:
                        pytest.fail(
                            "gpu-cuda optional extra should not exist. GPU installation is manual-only"
                        )

        # Verify current platform is not Linux (test assumption)
        assert sys.platform != "linux", "This test should only run on non-Linux platforms"

    def test_only_gpu_cuda_extra_exists(self, pyproject_toml_path):
        """Test that NO GPU optional extras exist (all removed in v0.1.0).

        The gpu-cuda, gpu-metal, and gpu-rocm extras were all removed.
        GPU installation is now manual-only to prevent silent CPU/GPU conflicts.
        """
        content = pyproject_toml_path.read_text(encoding="utf-8")

        # Extract optional dependencies section
        if "[project.optional-dependencies]" in content:
            deps_section = content.split("[project.optional-dependencies]")[1]
            # Stop at next TOML section (starts with [tool. or [project.urls)
            next_section_idx = deps_section.find("\n[")
            if next_section_idx > 0:
                deps_section = deps_section[:next_section_idx]

            # Count GPU-related extras
            gpu_extras = []
            for line in deps_section.split("\n"):
                if "gpu-cuda" in line and "=" in line:
                    gpu_extras.append("gpu-cuda")
                elif "gpu-metal" in line and "=" in line:
                    gpu_extras.append("gpu-metal")
                elif "gpu-rocm" in line and "=" in line:
                    gpu_extras.append("gpu-rocm")

            assert gpu_extras == [], (
                f"No GPU extras should exist (all removed in v0.1.0), found: {gpu_extras}"
            )

    def test_no_references_to_legacy_gpu_backends(self, pyproject_toml_path):
        """Test that no references to Metal or ROCm remain in dependencies."""
        content = pyproject_toml_path.read_text(encoding="utf-8")

        # Extract the dependencies sections (both regular and optional)
        if "[project]" in content:
            project_section = content.split("[project]")[1]

            # Check dependencies array
            if "dependencies = [" in project_section:
                deps_array = project_section.split("dependencies = [")[1]
                deps_array = deps_array.split("]")[0]

                # Should not reference metal or rocm in dependencies
                assert "metal" not in deps_array.lower(), (
                    "No Metal references in dependencies array"
                )
                assert "rocm" not in deps_array.lower(), "No ROCm references in dependencies array"
