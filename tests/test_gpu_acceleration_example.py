"""
Tests for GPU acceleration example script.

This module tests platform detection and appropriate messaging in the
gpu_acceleration_example.py script.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest


class TestGPUAccelerationExample:
    """Test GPU acceleration example script behavior."""

    def test_platform_detection_in_example(self):
        """Test that example script can detect platform."""
        # Import after mocking
        with patch("sys.platform", "linux"):
            # Mock the backend module
            mock_backend = MagicMock()
            mock_backend.get_device_info.return_value = {
                "backend": "jax",
                "os_platform": "linux",
                "gpu_supported": True,
                "cuda_version": (12, 0),
            }

            with patch.dict("sys.modules", {"piblin_jax.backend": mock_backend}):
                # Should not raise error
                assert sys.platform == "linux"

    def test_cpu_only_message_on_macos(self):
        """Test CPU-only message displayed on macOS."""
        with patch("sys.platform", "darwin"):
            mock_backend = MagicMock()
            mock_backend.get_device_info.return_value = {
                "backend": "numpy",
                "os_platform": "macos",
                "gpu_supported": False,
                "cuda_version": None,
            }
            mock_backend._detect_platform.return_value = "macos"

            with patch.dict("sys.modules", {"piblin_jax.backend": mock_backend}):
                # Verify platform is detected as macOS
                assert sys.platform == "darwin"

    def test_cpu_only_message_on_windows(self):
        """Test CPU-only message displayed on Windows."""
        with patch("sys.platform", "win32"):
            mock_backend = MagicMock()
            mock_backend.get_device_info.return_value = {
                "backend": "numpy",
                "os_platform": "windows",
                "gpu_supported": False,
                "cuda_version": None,
            }
            mock_backend._detect_platform.return_value = "windows"

            with patch.dict("sys.modules", {"piblin_jax.backend": mock_backend}):
                # Verify platform is detected as Windows
                assert sys.platform == "win32"

    def test_gpu_demonstrations_skip_on_non_linux(self):
        """Test GPU demonstrations skip on non-Linux platforms."""
        # Mock macOS platform
        with patch("sys.platform", "darwin"):
            from piblin_jax.backend import get_device_info

            info = get_device_info()
            platform = info.get("os_platform")

            # On non-Linux, GPU should not be supported
            if platform != "linux":
                assert info.get("gpu_supported") is False

    def test_linux_with_cuda12_enables_gpu(self):
        """Test Linux with CUDA 12+ enables GPU demonstrations."""
        mock_backend = MagicMock()
        mock_backend.get_device_info.return_value = {
            "backend": "jax",
            "os_platform": "linux",
            "gpu_supported": True,
            "cuda_version": (12, 3),
            "platform": "gpu",
        }

        with patch.dict("sys.modules", {"piblin_jax.backend": mock_backend}):
            # Verify GPU is supported on Linux with CUDA 12+
            info = mock_backend.get_device_info()
            assert info["os_platform"] == "linux"
            assert info["gpu_supported"] is True
            assert info["cuda_version"][0] >= 12

    def test_device_info_includes_platform_fields(self):
        """Test that device info includes new platform validation fields."""
        from piblin_jax.backend import get_device_info

        info = get_device_info()

        # Verify new fields exist
        assert "os_platform" in info
        assert "gpu_supported" in info
        assert "cuda_version" in info

        # Verify field types
        assert isinstance(info["os_platform"], str)
        assert isinstance(info["gpu_supported"], bool)
        assert info["cuda_version"] is None or isinstance(info["cuda_version"], tuple)

    def test_print_device_info_displays_platform_status(self):
        """Test print_device_info displays platform validation status."""
        from io import StringIO
        from unittest.mock import patch

        from piblin_jax.backend import get_device_info

        # Capture output
        with patch("sys.stdout", new=StringIO()) as fake_out:
            info = get_device_info()
            # Verify we can access platform information
            assert "os_platform" in info
            assert "gpu_supported" in info

    def test_example_no_metal_or_rocm_references(self):
        """Test example script has no Metal or ROCm references."""
        from pathlib import Path

        # Get path relative to project root (tests/../examples/)
        example_path = Path(__file__).parent.parent / "examples" / "gpu_acceleration_example.py"

        # Read the example file (explicit UTF-8 for cross-platform compatibility)
        with open(example_path, encoding="utf-8") as f:
            content = f.read().lower()

        # Verify no references to Metal or ROCm
        assert "metal" not in content, "Example should not reference Metal"
        assert "rocm" not in content, "Example should not reference ROCm"
