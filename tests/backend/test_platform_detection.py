"""
Tests for platform detection and CUDA validation logic.

This module tests the platform detection, CUDA version detection,
and GPU availability validation for the backend abstraction layer.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest


class TestPlatformDetection:
    """Test platform detection and GPU availability logic."""

    def test_platform_detection_helper(self):
        """Test _detect_platform() helper function."""
        from quantiq.backend import _detect_platform

        # Test with different sys.platform values
        test_cases = [
            ("linux", "linux"),
            ("linux2", "linux"),
            ("darwin", "macos"),
            ("win32", "windows"),
            ("win64", "windows"),
        ]

        for platform_value, expected in test_cases:
            with patch("sys.platform", platform_value):
                result = _detect_platform()
                assert result == expected, (
                    f"Expected '{expected}' for platform '{platform_value}', got '{result}'"
                )

    def test_cuda_version_validation_valid(self):
        """Test CUDA version validation for valid versions."""
        from quantiq.backend import _validate_cuda_version

        # Test valid CUDA versions (>= 12.0)
        assert _validate_cuda_version((12, 0)) is True
        assert _validate_cuda_version((12, 1)) is True
        assert _validate_cuda_version((13, 0)) is True
        assert _validate_cuda_version((15, 2)) is True

    def test_cuda_version_validation_invalid(self):
        """Test CUDA version validation for invalid versions."""
        from quantiq.backend import _validate_cuda_version

        # Test invalid CUDA versions (< 12.0)
        assert _validate_cuda_version((11, 8)) is False
        assert _validate_cuda_version((11, 0)) is False
        assert _validate_cuda_version((10, 2)) is False
        assert _validate_cuda_version(None) is False

    def test_cuda_version_detection_with_mock(self):
        """Test CUDA version detection with mocked JAX."""
        from quantiq.backend import _get_cuda_version

        # Mock JAX with CUDA version
        mock_jax = MagicMock()
        mock_backend = MagicMock()
        mock_backend.platform_version = "12.3"
        mock_jax.lib.xla_bridge.get_backend.return_value = mock_backend

        with (
            patch.dict("sys.modules", {"jax": mock_jax}),
            patch("quantiq.backend.jax", mock_jax),
        ):
            result = _get_cuda_version()
            assert result == (12, 3), f"Expected (12, 3), got {result}"

    def test_cuda_version_detection_exception(self):
        """Test CUDA version detection when exception occurs."""
        from quantiq.backend import _get_cuda_version

        # Mock JAX that raises exception
        mock_jax = MagicMock()
        mock_jax.lib.xla_bridge.get_backend.side_effect = Exception("No CUDA")

        with patch("quantiq.backend.jax", mock_jax):
            result = _get_cuda_version()
            assert result is None, "Should return None when CUDA unavailable"

    def test_device_info_includes_platform_fields(self):
        """Test that get_device_info() includes new platform validation fields."""
        from quantiq.backend import get_device_info

        info = get_device_info()

        # Verify new fields are present
        assert "os_platform" in info, "Device info should include os_platform"
        assert "gpu_supported" in info, "Device info should include gpu_supported"
        assert "cuda_version" in info, "Device info should include cuda_version"

        # Verify types
        assert isinstance(info["os_platform"], str), "os_platform should be string"
        assert isinstance(info["gpu_supported"], bool), "gpu_supported should be boolean"
        assert info["cuda_version"] is None or isinstance(info["cuda_version"], tuple), (
            "cuda_version should be None or tuple"
        )

        # Verify os_platform value is valid
        assert info["os_platform"] in ["linux", "macos", "windows"], (
            f"os_platform should be 'linux', 'macos', or 'windows', got '{info['os_platform']}'"
        )

    def test_device_info_backward_compatibility(self):
        """Test that get_device_info() maintains backward compatibility."""
        from quantiq.backend import get_device_info

        info = get_device_info()

        # Verify original fields are still present
        assert "backend" in info, "Device info should include backend"
        assert "devices" in info, "Device info should include devices"
        assert "default_device" in info, "Device info should include default_device"

        # Verify types of original fields
        assert isinstance(info["backend"], str)
        assert isinstance(info["devices"], list)
        assert isinstance(info["default_device"], str)

    def test_current_platform_detection(self):
        """Test that platform is correctly detected for current system."""
        from quantiq.backend import get_device_info

        info = get_device_info()

        # Verify platform matches current system
        if sys.platform.startswith("linux"):
            assert info["os_platform"] == "linux"
        elif sys.platform == "darwin":
            assert info["os_platform"] == "macos"
        elif sys.platform.startswith("win"):
            assert info["os_platform"] == "windows"
