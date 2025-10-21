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
        content = pyproject_toml_path.read_text()

        # Verify no gpu-metal section exists
        assert "gpu-metal" not in content, (
            "gpu-metal should be completely removed from pyproject.toml"
        )

        # Verify no references to Metal GPU support
        assert "Metal" not in content or "Apple Silicon" in content.split("[project.optional-dependencies]")[0], (
            "No Metal references should exist in optional dependencies section"
        )

    def test_gpu_rocm_removed_from_dependencies(self, pyproject_toml_path):
        """Test that gpu-rocm optional dependency has been completely removed."""
        content = pyproject_toml_path.read_text()

        # Verify no gpu-rocm section exists
        assert "gpu-rocm" not in content, (
            "gpu-rocm should be completely removed from pyproject.toml"
        )

        # Verify no references to ROCm GPU support
        assert "rocm" not in content.lower().split("[project.optional-dependencies]")[1], (
            "No ROCm references should exist in optional dependencies section"
        )

    def test_gpu_cuda_has_linux_platform_marker(self, pyproject_toml_path):
        """Test that gpu-cuda dependency includes Linux platform marker."""
        content = pyproject_toml_path.read_text()

        # Verify gpu-cuda exists
        assert "gpu-cuda" in content, "gpu-cuda optional dependency should exist"

        # Verify platform marker is present
        assert "sys_platform == 'linux'" in content, (
            "gpu-cuda should include sys_platform == 'linux' marker"
        )

        # Verify jax[cuda12] is specified
        assert "jax[cuda12]" in content, (
            "gpu-cuda should specify jax[cuda12] dependency"
        )

    def test_gpu_cuda_platform_marker_syntax(self, pyproject_toml_path):
        """Test that platform marker syntax is correct for pip installation."""
        content = pyproject_toml_path.read_text()

        # Extract the gpu-cuda section
        lines = content.split("\n")
        gpu_cuda_section = []
        in_section = False

        for line in lines:
            if "gpu-cuda" in line and "=" in line:
                in_section = True
            elif in_section:
                if line.strip().startswith("[") and not line.strip().startswith('["'):
                    break
                if line.strip() and not line.strip().startswith("#"):
                    gpu_cuda_section.append(line.strip())

        # Verify at least one line has the platform marker
        assert any("sys_platform == 'linux'" in line for line in gpu_cuda_section), (
            "Platform marker should be in gpu-cuda dependency specification"
        )

    @pytest.mark.skipif(
        sys.platform != "linux",
        reason="Platform marker enforcement only testable on Linux"
    )
    def test_gpu_cuda_installable_on_linux(self):
        """Test that gpu-cuda extra can be queried on Linux systems."""
        # This test verifies the syntax is valid, not that it actually installs
        # We use pip's dry-run capabilities to check dependency resolution
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--dry-run", "quantiq[gpu-cuda]"],
            capture_output=True,
            text=True,
        )

        # On Linux, the command should at least parse correctly
        # (may fail for other reasons like network, but not syntax)
        assert "error: invalid requirement" not in result.stderr.lower(), (
            "gpu-cuda dependency syntax should be valid"
        )

    @pytest.mark.skipif(
        sys.platform == "linux",
        reason="Non-Linux platform marker test only on non-Linux systems"
    )
    def test_gpu_cuda_platform_marker_on_non_linux(self, pyproject_toml_path):
        """Test that platform marker correctly identifies non-Linux systems."""
        # This test verifies that the marker syntax is correct
        # by checking that sys_platform would evaluate correctly

        content = pyproject_toml_path.read_text()

        # The marker should be present
        assert "sys_platform == 'linux'" in content, (
            "Platform marker should restrict to Linux only"
        )

        # Verify current platform is not Linux (test assumption)
        assert sys.platform != "linux", (
            "This test should only run on non-Linux platforms"
        )

    def test_only_gpu_cuda_extra_exists(self, pyproject_toml_path):
        """Test that gpu-cuda is the only GPU optional dependency."""
        content = pyproject_toml_path.read_text()

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

            assert gpu_extras == ["gpu-cuda"], (
                f"Only gpu-cuda should exist as GPU extra, found: {gpu_extras}"
            )

    def test_no_references_to_legacy_gpu_backends(self, pyproject_toml_path):
        """Test that no references to Metal or ROCm remain in dependencies."""
        content = pyproject_toml_path.read_text()

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
                assert "rocm" not in deps_array.lower(), (
                    "No ROCm references in dependencies array"
                )
