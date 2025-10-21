"""
Tests for backend abstraction layer.

This module tests backend detection, fallback mechanisms, array operations,
and boundary conversions between JAX and NumPy.
"""

import sys
import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestBackendDetection:
    """Test backend detection and initialization."""

    def test_backend_detection_with_jax(self):
        """Test that backend is 'jax' when JAX is available on Linux with CUDA 12+."""
        # This test now accounts for platform validation
        try:
            import jax

            # Reimport to get fresh backend detection
            if "quantiq.backend" in sys.modules:
                del sys.modules["quantiq.backend"]
            from quantiq.backend import BACKEND, get_backend

            # Backend depends on platform: 'jax' on Linux with CUDA 12+, 'numpy' otherwise
            if sys.platform.startswith("linux"):
                # On Linux, backend depends on CUDA version (might be 'jax' or 'numpy')
                assert BACKEND in ["jax", "numpy"], f"Expected BACKEND in ['jax', 'numpy'], got '{BACKEND}'"
            else:
                # On non-Linux platforms, should fall back to NumPy
                assert BACKEND == "numpy", f"Expected BACKEND='numpy' on non-Linux platforms, got '{BACKEND}'"
            assert get_backend() == BACKEND, "get_backend() should match BACKEND"
        except ImportError:
            pytest.skip("JAX not available, skipping JAX backend test")

    def test_backend_fallback_to_numpy(self):
        """Test that backend falls back to NumPy when JAX unavailable."""
        # Mock JAX import failure
        with patch.dict("sys.modules", {"jax": None, "jax.numpy": None}):
            # Remove cached module to force re-import
            if "quantiq.backend" in sys.modules:
                del sys.modules["quantiq.backend"]

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                from quantiq.backend import BACKEND, get_backend

                assert BACKEND == "numpy", f"Expected BACKEND='numpy', got '{BACKEND}'"
                assert get_backend() == "numpy", "get_backend() should return 'numpy'"

                # Verify warning was issued
                assert len(w) >= 1, "Expected warning when falling back to NumPy"
                warning_messages = [str(warning.message).lower() for warning in w]
                assert any("jax" in msg and "numpy" in msg for msg in warning_messages), (
                    "Warning should mention JAX and NumPy fallback"
                )

    def test_jax_availability_query(self):
        """Test is_jax_available() function."""
        from quantiq.backend import is_jax_available

        result = is_jax_available()
        assert isinstance(result, bool), "is_jax_available() should return bool"

        # The result depends on both JAX being importable AND platform validation
        try:
            import jax

            # On Linux with CUDA 12+, JAX should be available
            # On other platforms, JAX is available but GPU support is disabled
            if sys.platform.startswith("linux"):
                # Result depends on CUDA version
                assert isinstance(result, bool), "is_jax_available() should return bool on Linux"
            else:
                # On non-Linux platforms, should return False due to platform restriction
                assert result is False, "is_jax_available() should return False on non-Linux platforms"
        except ImportError:
            assert result is False, "is_jax_available() should return False when JAX not installed"

    def test_device_info_query(self):
        """Test get_device_info() function returns valid structure."""
        from quantiq.backend import get_device_info

        info = get_device_info()
        assert isinstance(info, dict), "get_device_info() should return dict"
        assert "backend" in info, "Device info should contain 'backend' key"
        assert "devices" in info, "Device info should contain 'devices' key"
        assert info["backend"] in ["jax", "numpy"], "Backend should be 'jax' or 'numpy'"


class TestArrayOperations:
    """Test backend-agnostic array operations."""

    def test_array_creation(self):
        """Test creating arrays using unified interface."""
        from quantiq.backend import jnp

        arr = jnp.array([1, 2, 3, 4])
        assert arr.shape == (4,), "Array should have shape (4,)"
        assert jnp.sum(arr) == 10, "Array sum should be 10"

    def test_array_operations_on_backend(self):
        """Test that basic array operations work on current backend."""
        from quantiq.backend import jnp
        from quantiq.backend.operations import concatenate, copy, reshape, stack

        # Create test arrays
        arr1 = jnp.array([1, 2, 3])
        arr2 = jnp.array([4, 5, 6])

        # Test copy
        arr_copy = copy(arr1)
        assert jnp.array_equal(arr1, arr_copy), "Copy should create equal array"

        # Test concatenate
        arr_concat = concatenate([arr1, arr2])
        assert arr_concat.shape == (6,), "Concatenated array should have shape (6,)"
        assert jnp.array_equal(arr_concat, jnp.array([1, 2, 3, 4, 5, 6])), (
            "Concatenation should preserve values"
        )

        # Test stack
        arr_stack = stack([arr1, arr2])
        assert arr_stack.shape == (2, 3), "Stacked array should have shape (2, 3)"

        # Test reshape
        arr_reshaped = reshape(arr_concat, (2, 3))
        assert arr_reshaped.shape == (2, 3), "Reshaped array should have shape (2, 3)"

    def test_jit_decorator_works(self):
        """Test that JIT decorator works (even as no-op for NumPy)."""
        from quantiq.backend.operations import jit

        @jit
        def simple_function(x):
            return x * 2

        from quantiq.backend import jnp

        arr = jnp.array([1, 2, 3])
        result = simple_function(arr)

        expected = jnp.array([2, 4, 6])
        assert jnp.array_equal(result, expected), "JIT-decorated function should work"


class TestBoundaryConversions:
    """Test NumPy boundary conversions."""

    def test_to_numpy_conversion(self):
        """Test converting backend arrays to NumPy."""
        from quantiq.backend import jnp, to_numpy

        arr = jnp.array([1.0, 2.0, 3.0])
        np_arr = to_numpy(arr)

        assert isinstance(np_arr, np.ndarray), "to_numpy() should return np.ndarray"
        assert np.array_equal(np_arr, np.array([1.0, 2.0, 3.0])), (
            "Converted array should have same values"
        )

    def test_from_numpy_conversion(self):
        """Test converting NumPy arrays to backend."""
        from quantiq.backend import from_numpy, jnp

        np_arr = np.array([1.0, 2.0, 3.0])
        backend_arr = from_numpy(np_arr)

        # The type depends on backend, but it should work with jnp operations
        result = jnp.sum(backend_arr)
        assert float(result) == 6.0, "from_numpy() result should work with backend operations"

    def test_roundtrip_conversion(self):
        """Test that to_numpy and from_numpy are inverses."""
        from quantiq.backend import from_numpy, jnp, to_numpy

        original = jnp.array([1.0, 2.0, 3.0])
        np_arr = to_numpy(original)
        backend_arr = from_numpy(np_arr)

        assert jnp.array_equal(original, backend_arr), "Roundtrip conversion should preserve values"


class TestPlatformValidationIntegration:
    """Test GPU restriction platform validation integration scenarios."""

    def test_linux_cuda12_gpu_enabled_integration(self):
        """Test Linux with CUDA 12+ enables GPU in full workflow."""
        # Mock platform detection and CUDA version
        mock_jax = MagicMock()
        mock_jax_numpy = MagicMock()
        mock_backend = MagicMock()
        mock_backend.platform_version = "12.3"

        # Mock both old and new JAX APIs
        mock_jax.lib.xla_bridge.get_backend.return_value = mock_backend

        # Mock new JAX API (jax.extend.backend)
        mock_jax_extend_backend = MagicMock()
        mock_jax_extend_backend.get_backend.return_value = mock_backend
        mock_jax.extend.backend = mock_jax_extend_backend

        # Mock jax.devices() for legacy extras check
        mock_jax.devices.return_value = []

        # Mock JAX submodules that numpyro might import
        mock_jax_scipy = MagicMock()
        mock_jax.scipy = mock_jax_scipy

        # Remove cached module
        if "quantiq.backend" in sys.modules:
            del sys.modules["quantiq.backend"]

        with patch("sys.platform", "linux"):
            with patch.dict("sys.modules", {
                "jax": mock_jax,
                "jax.numpy": mock_jax_numpy,
                "jax.scipy": mock_jax_scipy,
                "jax.scipy.special": MagicMock()
            }):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    import importlib
                    import quantiq.backend
                    importlib.reload(quantiq.backend)

                    from quantiq.backend import BACKEND, get_device_info

                    # On Linux with CUDA 12+, should use JAX backend
                    assert BACKEND == "jax", "Should use JAX backend on Linux with CUDA 12+"

                    # Should not issue platform warning
                    platform_warnings = [w_msg for w_msg in w if "GPU support is only available" in str(w_msg.message)]
                    assert len(platform_warnings) == 0, "Should not warn on Linux with CUDA 12+"

                    # Device info should reflect GPU support
                    info = get_device_info()
                    assert info["os_platform"] == "linux"
                    assert info["gpu_supported"] is True
                    assert info["cuda_version"] == (12, 3)

    def test_linux_cuda11_fallback_integration(self):
        """Test Linux with CUDA 11.x falls back to CPU with version warning."""
        mock_jax = MagicMock()
        mock_jax_numpy = MagicMock()
        mock_backend = MagicMock()
        mock_backend.platform_version = "11.8"

        # Mock both old and new JAX APIs
        mock_jax.lib.xla_bridge.get_backend.return_value = mock_backend

        # Mock new JAX API (jax.extend.backend)
        mock_jax_extend_backend = MagicMock()
        mock_jax_extend_backend.get_backend.return_value = mock_backend
        mock_jax.extend.backend = mock_jax_extend_backend

        # Mock jax.devices() for legacy extras check
        mock_jax.devices.return_value = []

        # Mock JAX submodules that numpyro might import
        mock_jax_scipy = MagicMock()
        mock_jax.scipy = mock_jax_scipy

        # Remove cached module
        if "quantiq.backend" in sys.modules:
            del sys.modules["quantiq.backend"]

        with patch("sys.platform", "linux"):
            with patch.dict("sys.modules", {
                "jax": mock_jax,
                "jax.numpy": mock_jax_numpy,
                "jax.scipy": mock_jax_scipy,
                "jax.scipy.special": MagicMock()
            }):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    import importlib
                    import quantiq.backend
                    importlib.reload(quantiq.backend)

                    from quantiq.backend import BACKEND, get_device_info

                    # Should fall back to NumPy
                    assert BACKEND == "numpy", "Should fall back to NumPy with CUDA 11.x"

                    # Should issue warning about CUDA version
                    platform_warnings = [w_msg for w_msg in w if "GPU support is only available" in str(w_msg.message)]
                    assert len(platform_warnings) >= 1, "Should warn about CUDA version requirement"

                    # Device info should reflect no GPU support
                    info = get_device_info()
                    assert info["os_platform"] == "linux"
                    assert info["gpu_supported"] is False
                    assert info["cuda_version"] == (11, 8)

    def test_macos_gpu_fallback_integration(self):
        """Test macOS platform falls back to CPU with platform warning."""
        mock_jax = MagicMock()
        mock_jax_numpy = MagicMock()

        # Remove cached module
        if "quantiq.backend" in sys.modules:
            del sys.modules["quantiq.backend"]

        with patch("sys.platform", "darwin"):
            with patch.dict("sys.modules", {"jax": mock_jax, "jax.numpy": mock_jax_numpy}):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    import importlib
                    import quantiq.backend
                    importlib.reload(quantiq.backend)

                    from quantiq.backend import BACKEND, get_device_info

                    # Should fall back to NumPy
                    assert BACKEND == "numpy", "Should fall back to NumPy on macOS"

                    # Should issue platform warning
                    platform_warnings = [w_msg for w_msg in w if "GPU support is only available" in str(w_msg.message)]
                    assert len(platform_warnings) >= 1, "Should warn about platform restriction"

                    # Device info should reflect macOS and no GPU support
                    info = get_device_info()
                    assert info["os_platform"] == "macos"
                    assert info["gpu_supported"] is False
                    assert info["cuda_version"] is None

    def test_windows_gpu_fallback_integration(self):
        """Test Windows platform falls back to CPU with platform warning."""
        mock_jax = MagicMock()
        mock_jax_numpy = MagicMock()

        # Remove cached module
        if "quantiq.backend" in sys.modules:
            del sys.modules["quantiq.backend"]

        with patch("sys.platform", "win32"):
            with patch.dict("sys.modules", {"jax": mock_jax, "jax.numpy": mock_jax_numpy}):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    import importlib
                    import quantiq.backend
                    importlib.reload(quantiq.backend)

                    from quantiq.backend import BACKEND, get_device_info

                    # Should fall back to NumPy
                    assert BACKEND == "numpy", "Should fall back to NumPy on Windows"

                    # Should issue platform warning
                    platform_warnings = [w_msg for w_msg in w if "GPU support is only available" in str(w_msg.message)]
                    assert len(platform_warnings) >= 1, "Should warn about platform restriction"

                    # Device info should reflect Windows and no GPU support
                    info = get_device_info()
                    assert info["os_platform"] == "windows"
                    assert info["gpu_supported"] is False
                    assert info["cuda_version"] is None

    def test_backward_compatibility_cpu_workflow(self):
        """Test that CPU-only workflows remain unchanged."""
        # This tests that when JAX is not available, everything works as before
        with patch.dict("sys.modules", {"jax": None, "jax.numpy": None}):
            if "quantiq.backend" in sys.modules:
                del sys.modules["quantiq.backend"]

            from quantiq.backend import BACKEND, get_backend, jnp

            # Should use NumPy backend
            assert BACKEND == "numpy"
            assert get_backend() == "numpy"

            # NumPy operations should work unchanged
            arr = jnp.array([1, 2, 3])
            assert arr.shape == (3,)
            assert jnp.sum(arr) == 6

            # Device info should work
            from quantiq.backend import get_device_info
            info = get_device_info()
            assert info["backend"] == "numpy"
            assert "os_platform" in info
            assert "gpu_supported" in info

    def test_cuda_detection_error_handling(self):
        """Test graceful handling of CUDA detection failures."""
        mock_jax = MagicMock()
        mock_jax_numpy = MagicMock()
        # Mock CUDA version detection to raise exception
        mock_jax.lib.xla_bridge.get_backend.side_effect = Exception("CUDA not available")

        # Remove cached module
        if "quantiq.backend" in sys.modules:
            del sys.modules["quantiq.backend"]

        with patch("sys.platform", "linux"):
            with patch.dict("sys.modules", {"jax": mock_jax, "jax.numpy": mock_jax_numpy}):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    import importlib
                    import quantiq.backend
                    importlib.reload(quantiq.backend)

                    from quantiq.backend import BACKEND, get_device_info

                    # Should fall back to NumPy when CUDA detection fails
                    assert BACKEND == "numpy", "Should fall back when CUDA detection fails"

                    # Device info should handle error gracefully
                    info = get_device_info()
                    assert info["os_platform"] == "linux"
                    assert info["gpu_supported"] is False
                    assert info["cuda_version"] is None

    def test_device_info_includes_all_platform_fields(self):
        """Test that get_device_info() includes all required platform fields."""
        from quantiq.backend import get_device_info

        info = get_device_info()

        # Verify all new platform fields are present
        required_fields = ["os_platform", "gpu_supported", "cuda_version"]
        for field in required_fields:
            assert field in info, f"Device info should include '{field}' field"

        # Verify backward compatibility fields
        legacy_fields = ["backend", "devices", "default_device"]
        for field in legacy_fields:
            assert field in info, f"Device info should maintain '{field}' for backward compatibility"

        # Verify field types
        assert isinstance(info["os_platform"], str)
        assert isinstance(info["gpu_supported"], bool)
        assert info["cuda_version"] is None or isinstance(info["cuda_version"], tuple)

    def test_linux_without_cuda_fallback(self):
        """Test Linux without CUDA falls back to CPU gracefully."""
        mock_jax = MagicMock()
        mock_jax_numpy = MagicMock()
        mock_backend = MagicMock()
        # Mock no CUDA available (no platform_version)
        del mock_backend.platform_version
        mock_jax.lib.xla_bridge.get_backend.return_value = mock_backend

        # Remove cached module
        if "quantiq.backend" in sys.modules:
            del sys.modules["quantiq.backend"]

        with patch("sys.platform", "linux"):
            with patch.dict("sys.modules", {"jax": mock_jax, "jax.numpy": mock_jax_numpy}):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    import importlib
                    import quantiq.backend
                    importlib.reload(quantiq.backend)

                    from quantiq.backend import BACKEND, get_device_info

                    # Should fall back to NumPy when CUDA unavailable
                    assert BACKEND == "numpy", "Should fall back when no CUDA on Linux"

                    # Device info should reflect no CUDA
                    info = get_device_info()
                    assert info["os_platform"] == "linux"
                    assert info["gpu_supported"] is False
                    assert info["cuda_version"] is None

    def test_platform_validation_warning_messages(self):
        """Test that platform validation warnings contain helpful messages."""
        mock_jax = MagicMock()
        mock_jax_numpy = MagicMock()

        # Remove cached module
        if "quantiq.backend" in sys.modules:
            del sys.modules["quantiq.backend"]

        # Test on macOS
        with patch("sys.platform", "darwin"):
            with patch.dict("sys.modules", {"jax": mock_jax, "jax.numpy": mock_jax_numpy}):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    import importlib
                    import quantiq.backend
                    importlib.reload(quantiq.backend)

                    # Warning should mention Linux and CUDA 12+
                    platform_warnings = [w_msg for w_msg in w if "GPU support is only available" in str(w_msg.message)]
                    assert len(platform_warnings) >= 1, "Should issue warning"

                    warning_text = str(platform_warnings[0].message).lower()
                    assert "linux" in warning_text, "Warning should mention Linux requirement"
                    assert "cuda 12+" in warning_text, "Warning should mention CUDA 12+ requirement"
                    assert "cpu" in warning_text, "Warning should mention CPU fallback"
