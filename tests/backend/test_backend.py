"""
Tests for backend abstraction layer.

This module tests backend detection, fallback mechanisms, array operations,
and boundary conversions between JAX and NumPy.
"""

import pytest
import warnings
import numpy as np
import sys
from unittest.mock import patch


class TestBackendDetection:
    """Test backend detection and initialization."""

    def test_backend_detection_with_jax(self):
        """Test that backend is 'jax' when JAX is available."""
        # This test assumes JAX is installed in the test environment
        try:
            import jax
            # Reimport to get fresh backend detection
            if 'quantiq.backend' in sys.modules:
                del sys.modules['quantiq.backend']
            from quantiq.backend import BACKEND, get_backend

            assert BACKEND == 'jax', f"Expected BACKEND='jax', got '{BACKEND}'"
            assert get_backend() == 'jax', "get_backend() should return 'jax'"
        except ImportError:
            pytest.skip("JAX not available, skipping JAX backend test")

    def test_backend_fallback_to_numpy(self):
        """Test that backend falls back to NumPy when JAX unavailable."""
        # Mock JAX import failure
        with patch.dict('sys.modules', {'jax': None, 'jax.numpy': None}):
            # Remove cached module to force re-import
            if 'quantiq.backend' in sys.modules:
                del sys.modules['quantiq.backend']

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                from quantiq.backend import BACKEND, get_backend

                assert BACKEND == 'numpy', f"Expected BACKEND='numpy', got '{BACKEND}'"
                assert get_backend() == 'numpy', "get_backend() should return 'numpy'"

                # Verify warning was issued
                assert len(w) >= 1, "Expected warning when falling back to NumPy"
                warning_messages = [str(warning.message).lower() for warning in w]
                assert any('jax' in msg and 'numpy' in msg for msg in warning_messages), \
                    "Warning should mention JAX and NumPy fallback"

    def test_jax_availability_query(self):
        """Test is_jax_available() function."""
        from quantiq.backend import is_jax_available

        result = is_jax_available()
        assert isinstance(result, bool), "is_jax_available() should return bool"

        # The result should match whether JAX can be imported
        try:
            import jax
            assert result is True, "is_jax_available() should return True when JAX installed"
        except ImportError:
            assert result is False, "is_jax_available() should return False when JAX not installed"

    def test_device_info_query(self):
        """Test get_device_info() function returns valid structure."""
        from quantiq.backend import get_device_info

        info = get_device_info()
        assert isinstance(info, dict), "get_device_info() should return dict"
        assert 'backend' in info, "Device info should contain 'backend' key"
        assert 'devices' in info, "Device info should contain 'devices' key"
        assert info['backend'] in ['jax', 'numpy'], "Backend should be 'jax' or 'numpy'"


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
        from quantiq.backend.operations import copy, concatenate, stack, reshape

        # Create test arrays
        arr1 = jnp.array([1, 2, 3])
        arr2 = jnp.array([4, 5, 6])

        # Test copy
        arr_copy = copy(arr1)
        assert jnp.array_equal(arr1, arr_copy), "Copy should create equal array"

        # Test concatenate
        arr_concat = concatenate([arr1, arr2])
        assert arr_concat.shape == (6,), "Concatenated array should have shape (6,)"
        assert jnp.array_equal(arr_concat, jnp.array([1, 2, 3, 4, 5, 6])), \
            "Concatenation should preserve values"

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
        assert np.array_equal(np_arr, np.array([1.0, 2.0, 3.0])), \
            "Converted array should have same values"

    def test_from_numpy_conversion(self):
        """Test converting NumPy arrays to backend."""
        from quantiq.backend import jnp, from_numpy

        np_arr = np.array([1.0, 2.0, 3.0])
        backend_arr = from_numpy(np_arr)

        # The type depends on backend, but it should work with jnp operations
        result = jnp.sum(backend_arr)
        assert float(result) == 6.0, "from_numpy() result should work with backend operations"

    def test_roundtrip_conversion(self):
        """Test that to_numpy and from_numpy are inverses."""
        from quantiq.backend import jnp, to_numpy, from_numpy

        original = jnp.array([1.0, 2.0, 3.0])
        np_arr = to_numpy(original)
        backend_arr = from_numpy(np_arr)

        assert jnp.array_equal(original, backend_arr), \
            "Roundtrip conversion should preserve values"
