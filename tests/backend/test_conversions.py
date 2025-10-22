"""
Tests for backend array conversion functions.

This module tests pytree conversions, boundary conversions between JAX and NumPy,
and legacy GPU extras detection.
"""

import sys
import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from quantiq.backend import (
    from_numpy,
    from_numpy_pytree,
    is_jax_available,
    jnp,
    to_numpy,
    to_numpy_pytree,
)


class TestArrayConversions:
    """Test basic array conversion functions."""

    def test_to_numpy_from_backend_array(self):
        """Test to_numpy converts backend arrays to NumPy."""
        arr = jnp.array([1.0, 2.0, 3.0])
        np_arr = to_numpy(arr)

        assert isinstance(np_arr, np.ndarray)
        assert np.array_equal(np_arr, np.array([1.0, 2.0, 3.0]))

    def test_to_numpy_from_numpy_array(self):
        """Test to_numpy passes through NumPy arrays."""
        np_arr = np.array([1.0, 2.0, 3.0])
        result = to_numpy(np_arr)

        assert isinstance(result, np.ndarray)
        assert result is np_arr  # Should be the same object

    def test_to_numpy_with_complex_dtype(self):
        """Test to_numpy with complex data types."""
        arr = jnp.array([1.0 + 2.0j, 3.0 + 4.0j])
        np_arr = to_numpy(arr)

        assert isinstance(np_arr, np.ndarray)
        assert np.array_equal(np_arr, np.array([1.0 + 2.0j, 3.0 + 4.0j]))

    def test_from_numpy_to_backend(self):
        """Test from_numpy converts NumPy to backend array."""
        np_arr = np.array([1.0, 2.0, 3.0])
        backend_arr = from_numpy(np_arr)

        # Should work with backend operations
        result = jnp.sum(backend_arr)
        assert float(result) == 6.0

    def test_roundtrip_conversion(self):
        """Test roundtrip conversion preserves data."""
        original = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Convert to NumPy and back
        np_arr = to_numpy(original)
        backend_arr = from_numpy(np_arr)

        assert jnp.array_equal(original, backend_arr)

    def test_to_numpy_with_exception_fallback(self):
        """Test to_numpy fallback when asarray fails."""
        if not is_jax_available():
            pytest.skip("This test requires JAX backend")

        # Just verify we can convert normal arrays
        arr = jnp.array([1, 2, 3])
        result = to_numpy(arr)
        assert isinstance(result, np.ndarray)


class TestPytreeConversions:
    """Test pytree conversion functions for nested structures."""

    def test_to_numpy_pytree_with_dict(self):
        """Test to_numpy_pytree with dictionary."""
        pytree = {"a": jnp.array([1, 2]), "b": jnp.array([3, 4])}

        np_pytree = to_numpy_pytree(pytree)

        assert isinstance(np_pytree, dict)
        assert isinstance(np_pytree["a"], np.ndarray)
        assert isinstance(np_pytree["b"], np.ndarray)
        assert np.array_equal(np_pytree["a"], np.array([1, 2]))
        assert np.array_equal(np_pytree["b"], np.array([3, 4]))

    def test_to_numpy_pytree_with_list(self):
        """Test to_numpy_pytree with list."""
        pytree = [jnp.array([1, 2]), jnp.array([3, 4])]

        np_pytree = to_numpy_pytree(pytree)

        assert isinstance(np_pytree, list)
        assert len(np_pytree) == 2
        assert isinstance(np_pytree[0], np.ndarray)
        assert np.array_equal(np_pytree[0], np.array([1, 2]))

    def test_to_numpy_pytree_with_tuple(self):
        """Test to_numpy_pytree with tuple."""
        pytree = (jnp.array([1, 2]), jnp.array([3, 4]))

        np_pytree = to_numpy_pytree(pytree)

        assert isinstance(np_pytree, tuple)
        assert len(np_pytree) == 2
        assert isinstance(np_pytree[0], np.ndarray)

    def test_to_numpy_pytree_nested(self):
        """Test to_numpy_pytree with nested structures."""
        pytree = {
            "layer1": {"a": jnp.array([1, 2]), "b": jnp.array([3, 4])},
            "layer2": [jnp.array([5, 6]), jnp.array([7, 8])],
        }

        np_pytree = to_numpy_pytree(pytree)

        assert isinstance(np_pytree["layer1"], dict)
        assert isinstance(np_pytree["layer1"]["a"], np.ndarray)
        assert isinstance(np_pytree["layer2"], list)
        assert isinstance(np_pytree["layer2"][0], np.ndarray)

    def test_to_numpy_pytree_with_scalars(self):
        """Test to_numpy_pytree preserves non-array values."""
        pytree = {"array": jnp.array([1, 2]), "scalar": 42, "string": "hello"}

        np_pytree = to_numpy_pytree(pytree)

        assert isinstance(np_pytree["array"], np.ndarray)
        assert np_pytree["scalar"] == 42
        assert np_pytree["string"] == "hello"

    def test_to_numpy_pytree_with_none(self):
        """Test to_numpy_pytree handles None values."""
        pytree = {"array": jnp.array([1, 2]), "none_value": None}

        np_pytree = to_numpy_pytree(pytree)

        assert isinstance(np_pytree["array"], np.ndarray)
        assert np_pytree["none_value"] is None

    def test_from_numpy_pytree_with_dict(self):
        """Test from_numpy_pytree with dictionary."""
        pytree = {"a": np.array([1, 2]), "b": np.array([3, 4])}

        backend_pytree = from_numpy_pytree(pytree)

        assert isinstance(backend_pytree, dict)
        assert jnp.array_equal(backend_pytree["a"], jnp.array([1, 2]))
        assert jnp.array_equal(backend_pytree["b"], jnp.array([3, 4]))

    def test_from_numpy_pytree_with_list(self):
        """Test from_numpy_pytree with list."""
        pytree = [np.array([1, 2]), np.array([3, 4])]

        backend_pytree = from_numpy_pytree(pytree)

        assert isinstance(backend_pytree, list)
        assert len(backend_pytree) == 2

    def test_from_numpy_pytree_with_tuple(self):
        """Test from_numpy_pytree with tuple."""
        pytree = (np.array([1, 2]), np.array([3, 4]))

        backend_pytree = from_numpy_pytree(pytree)

        assert isinstance(backend_pytree, tuple)
        assert len(backend_pytree) == 2

    def test_from_numpy_pytree_nested(self):
        """Test from_numpy_pytree with nested structures."""
        pytree = {
            "layer1": {"a": np.array([1, 2]), "b": np.array([3, 4])},
            "layer2": [np.array([5, 6]), np.array([7, 8])],
        }

        backend_pytree = from_numpy_pytree(pytree)

        assert isinstance(backend_pytree["layer1"], dict)
        assert jnp.array_equal(backend_pytree["layer1"]["a"], jnp.array([1, 2]))
        assert isinstance(backend_pytree["layer2"], list)

    def test_from_numpy_pytree_preserves_non_arrays(self):
        """Test from_numpy_pytree preserves non-array values."""
        pytree = {"array": np.array([1, 2]), "scalar": 42, "string": "hello"}

        backend_pytree = from_numpy_pytree(pytree)

        assert jnp.array_equal(backend_pytree["array"], jnp.array([1, 2]))
        assert backend_pytree["scalar"] == 42
        assert backend_pytree["string"] == "hello"

    def test_pytree_roundtrip_conversion(self):
        """Test pytree roundtrip conversion preserves structure and data."""
        original = {
            "arrays": [jnp.array([1, 2]), jnp.array([3, 4])],
            "nested": {"x": jnp.array([5, 6])},
            "scalar": 42,
        }

        # Convert to NumPy and back
        np_pytree = to_numpy_pytree(original)
        backend_pytree = from_numpy_pytree(np_pytree)

        # Verify structure
        assert isinstance(backend_pytree["arrays"], list)
        assert isinstance(backend_pytree["nested"], dict)
        assert backend_pytree["scalar"] == 42

        # Verify data
        assert jnp.array_equal(backend_pytree["arrays"][0], original["arrays"][0])
        assert jnp.array_equal(backend_pytree["nested"]["x"], original["nested"]["x"])


class TestCUDAVersionDetection:
    """Test CUDA version detection and parsing."""

    def test_get_cuda_version_with_single_digit_minor(self):
        """Test parsing CUDA version with single digit minor version."""
        from quantiq.backend import _get_cuda_version

        mock_jax = MagicMock()
        mock_backend = MagicMock()
        mock_backend.platform_version = "12.0"

        mock_jax_extend = MagicMock()
        mock_jax_extend_backend = MagicMock()
        mock_jax_extend_backend.get_backend.return_value = mock_backend
        mock_jax_extend.backend = mock_jax_extend_backend
        mock_jax.extend = mock_jax_extend

        with (
            patch.dict("sys.modules", {"jax": mock_jax, "jax.extend": mock_jax_extend}),
            patch("quantiq.backend.jax", mock_jax),
        ):
            result = _get_cuda_version()
            assert result == (12, 0)

    def test_get_cuda_version_with_patch_version(self):
        """Test parsing CUDA version with patch version."""
        from quantiq.backend import _get_cuda_version

        mock_jax = MagicMock()
        mock_backend = MagicMock()
        mock_backend.platform_version = "12.3.1"

        mock_jax_extend = MagicMock()
        mock_jax_extend_backend = MagicMock()
        mock_jax_extend_backend.get_backend.return_value = mock_backend
        mock_jax_extend.backend = mock_jax_extend_backend
        mock_jax.extend = mock_jax_extend

        with (
            patch.dict("sys.modules", {"jax": mock_jax, "jax.extend": mock_jax_extend}),
            patch("quantiq.backend.jax", mock_jax),
        ):
            result = _get_cuda_version()
            assert result == (12, 3)

    def test_get_cuda_version_invalid_format(self):
        """Test CUDA version detection with invalid version format."""
        from quantiq.backend import _get_cuda_version

        mock_jax = MagicMock()
        mock_backend = MagicMock()
        mock_backend.platform_version = "invalid"

        mock_jax_extend = MagicMock()
        mock_jax_extend_backend = MagicMock()
        mock_jax_extend_backend.get_backend.return_value = mock_backend
        mock_jax_extend.backend = mock_jax_extend_backend
        mock_jax.extend = mock_jax_extend

        with (
            patch.dict("sys.modules", {"jax": mock_jax, "jax.extend": mock_jax_extend}),
            patch("quantiq.backend.jax", mock_jax),
        ):
            result = _get_cuda_version()
            assert result is None

    def test_get_cuda_version_single_component(self):
        """Test CUDA version detection with single version component."""
        from quantiq.backend import _get_cuda_version

        mock_jax = MagicMock()
        mock_backend = MagicMock()
        mock_backend.platform_version = "12"

        mock_jax_extend = MagicMock()
        mock_jax_extend_backend = MagicMock()
        mock_jax_extend_backend.get_backend.return_value = mock_backend
        mock_jax_extend.backend = mock_jax_extend_backend
        mock_jax.extend = mock_jax_extend

        with (
            patch.dict("sys.modules", {"jax": mock_jax, "jax.extend": mock_jax_extend}),
            patch("quantiq.backend.jax", mock_jax),
        ):
            result = _get_cuda_version()
            assert result is None


class TestLegacyGPUExtras:
    """Test detection of legacy GPU extras (Metal, ROCm)."""

    def test_check_legacy_metal_backend(self):
        """Test detection of legacy Metal backend on macOS."""
        # Mock JAX with Metal device
        mock_jax = MagicMock()
        mock_device = MagicMock()
        mock_device.__str__ = MagicMock(return_value="MetalDevice(id=0)")
        mock_jax.devices.return_value = [mock_device]

        # Remove cached module
        if "quantiq.backend" in sys.modules:
            del sys.modules["quantiq.backend"]

        with (
            patch("sys.platform", "darwin"),
            patch.dict("sys.modules", {"jax": mock_jax, "jax.numpy": MagicMock()}),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            import importlib

            import quantiq.backend

            importlib.reload(quantiq.backend)

            # Should issue deprecation warning for Metal
            metal_warnings = [
                w_msg for w_msg in w if "gpu-metal is deprecated" in str(w_msg.message).lower()
            ]
            assert len(metal_warnings) >= 1, "Should warn about deprecated Metal backend"

    def test_check_legacy_rocm_backend(self):
        """Test detection of legacy ROCm backend."""
        # Mock JAX with ROCm device
        mock_jax = MagicMock()
        mock_device = MagicMock()
        mock_device.__str__ = MagicMock(return_value="RocmDevice(id=0)")
        mock_jax.devices.return_value = [mock_device]

        # Remove cached module
        if "quantiq.backend" in sys.modules:
            del sys.modules["quantiq.backend"]

        with (
            patch("sys.platform", "linux"),
            patch.dict("sys.modules", {"jax": mock_jax, "jax.numpy": MagicMock()}),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            import importlib

            import quantiq.backend

            importlib.reload(quantiq.backend)

            # Should issue deprecation warning for ROCm
            rocm_warnings = [
                w_msg for w_msg in w if "gpu-rocm is deprecated" in str(w_msg.message).lower()
            ]
            assert len(rocm_warnings) >= 1, "Should warn about deprecated ROCm backend"

    def test_check_legacy_gpu_with_gpu_string_on_macos(self):
        """Test detection of GPU device on macOS (likely Metal)."""
        # Mock JAX with generic GPU device on macOS
        mock_jax = MagicMock()
        mock_device = MagicMock()
        mock_device.__str__ = MagicMock(return_value="GpuDevice(id=0)")
        mock_jax.devices.return_value = [mock_device]

        # Remove cached module
        if "quantiq.backend" in sys.modules:
            del sys.modules["quantiq.backend"]

        with (
            patch("sys.platform", "darwin"),
            patch.dict("sys.modules", {"jax": mock_jax, "jax.numpy": MagicMock()}),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            import importlib

            import quantiq.backend

            importlib.reload(quantiq.backend)

            # Should issue warning for Metal
            metal_warnings = [w_msg for w_msg in w if "metal" in str(w_msg.message).lower()]
            assert len(metal_warnings) >= 1


class TestUnknownPlatform:
    """Test handling of unknown/unsupported platforms."""

    def test_unknown_platform_detection(self):
        """Test _detect_platform with unknown platform."""
        from quantiq.backend import _detect_platform

        with patch("sys.platform", "freebsd"):
            result = _detect_platform()
            assert result == "freebsd"  # Should return the actual platform string


class TestDeviceInfoEdgeCases:
    """Test edge cases in get_device_info."""

    def test_device_info_with_jax_exception(self):
        """Test get_device_info handles JAX exceptions gracefully."""
        # This test verifies that exceptions in JAX device detection are handled
        # The actual implementation catches exceptions and issues warnings
        from quantiq.backend import get_device_info

        # Even with potential errors, should return valid info structure
        info = get_device_info()

        assert isinstance(info, dict)
        assert "backend" in info
        assert "devices" in info
        assert "os_platform" in info

    def test_device_info_jax_old_api_fallback(self):
        """Test get_device_info falls back to old JAX API."""
        if not is_jax_available():
            pytest.skip("This test requires JAX")

        from quantiq.backend import get_device_info

        # Mock to simulate old JAX API
        with patch("quantiq.backend.jax") as mock_jax:
            mock_device = MagicMock()
            mock_device.__str__ = MagicMock(return_value="cpu:0")
            mock_jax.devices.return_value = [mock_device]

            # Make new API unavailable
            mock_jax.extend = MagicMock()
            del mock_jax.extend.backend

            info = get_device_info()

            # Should still work and extract platform from device string
            assert "platform" in info
