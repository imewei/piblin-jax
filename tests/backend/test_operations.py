"""
Tests for backend operations module.

This module tests backend-agnostic array operations, JIT compilation,
vectorization, and device management.
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from quantiq.backend import is_jax_available, jnp
from quantiq.backend.operations import (
    astype,
    concatenate,
    copy,
    device_get,
    device_put,
    ensure_array,
    grad,
    jit,
    reshape,
    stack,
    vmap,
)


class TestArrayOperations:
    """Test backend-agnostic array operations."""

    def test_copy_with_jax_backend(self):
        """Test copy operation with JAX backend."""
        arr = jnp.array([1, 2, 3, 4])
        arr_copy = copy(arr)

        # Arrays should be equal but potentially different objects
        assert jnp.array_equal(arr, arr_copy)
        assert jnp.sum(arr_copy) == 10

    def test_copy_with_numpy_backend(self):
        """Test copy operation covers both JAX and NumPy paths."""
        # When JAX is available, even NumPy arrays get converted via jnp.array()
        # When JAX is not available, np.copy() is used
        with patch("quantiq.backend.operations._JAX_AVAILABLE", False):
            from quantiq.backend.operations import copy as copy_numpy

            np_arr = np.array([1.0, 2.0, 3.0])
            np_copy = copy_numpy(np_arr)

            assert np.array_equal(np_arr, np_copy)
            assert isinstance(np_copy, np.ndarray)

            # Modify copy and verify original unchanged
            np_copy[0] = 999
            assert np_arr[0] == 1.0

    def test_concatenate_basic(self):
        """Test concatenate operation with default axis."""
        arr1 = jnp.array([1, 2, 3])
        arr2 = jnp.array([4, 5, 6])

        result = concatenate([arr1, arr2])

        assert result.shape == (6,)
        expected = jnp.array([1, 2, 3, 4, 5, 6])
        assert jnp.array_equal(result, expected)

    def test_concatenate_with_axis(self):
        """Test concatenate operation with specified axis."""
        arr1 = jnp.array([[1, 2], [3, 4]])
        arr2 = jnp.array([[5, 6], [7, 8]])

        # Concatenate along axis 0
        result_axis0 = concatenate([arr1, arr2], axis=0)
        assert result_axis0.shape == (4, 2)

        # Concatenate along axis 1
        result_axis1 = concatenate([arr1, arr2], axis=1)
        assert result_axis1.shape == (2, 4)

    def test_stack_basic(self):
        """Test stack operation with default axis."""
        arr1 = jnp.array([1, 2, 3])
        arr2 = jnp.array([4, 5, 6])

        result = stack([arr1, arr2])

        assert result.shape == (2, 3)
        assert jnp.array_equal(result[0], arr1)
        assert jnp.array_equal(result[1], arr2)

    def test_stack_with_axis(self):
        """Test stack operation with specified axis."""
        arr1 = jnp.array([1, 2, 3])
        arr2 = jnp.array([4, 5, 6])

        # Stack along axis 1
        result = stack([arr1, arr2], axis=1)
        assert result.shape == (3, 2)

    def test_reshape_basic(self):
        """Test reshape operation."""
        arr = jnp.array([1, 2, 3, 4, 5, 6])

        result = reshape(arr, (2, 3))
        assert result.shape == (2, 3)

        result_flat = reshape(result, 6)
        assert result_flat.shape == (6,)

    def test_reshape_with_inferred_dimension(self):
        """Test reshape with -1 for inferred dimension."""
        arr = jnp.array([1, 2, 3, 4, 5, 6, 7, 8])

        result = reshape(arr, (2, -1))
        assert result.shape == (2, 4)

        result2 = reshape(arr, (-1, 2))
        assert result2.shape == (4, 2)


class TestJITCompilation:
    """Test JIT compilation decorator."""

    def test_jit_decorator_basic(self):
        """Test JIT decorator with basic function."""

        @jit
        def square(x):
            return x**2

        arr = jnp.array([1.0, 2.0, 3.0])
        result = square(arr)

        expected = jnp.array([1.0, 4.0, 9.0])
        assert jnp.allclose(result, expected)

    def test_jit_decorator_with_kwargs(self):
        """Test JIT decorator with keyword arguments."""

        @jit(static_argnums=0)
        def power(n, x):
            return x**n

        arr = jnp.array([2.0, 3.0, 4.0])
        result = power(2, arr)

        expected = jnp.array([4.0, 9.0, 16.0])
        assert jnp.allclose(result, expected)

    def test_jit_with_parentheses_no_args(self):
        """Test JIT decorator with parentheses but no arguments."""

        @jit()
        def add_one(x):
            return x + 1

        arr = jnp.array([1, 2, 3])
        result = add_one(arr)

        expected = jnp.array([2, 3, 4])
        assert jnp.array_equal(result, expected)

    def test_jit_preserves_function_metadata(self):
        """Test that JIT decorator preserves function metadata."""

        @jit
        def documented_function(x):
            """This is a documented function."""
            return x * 2

        assert documented_function.__doc__ == "This is a documented function."

    def test_jit_numpy_fallback(self):
        """Test JIT decorator with NumPy backend (no-op)."""
        # Mock JAX as unavailable
        with patch("quantiq.backend.operations._JAX_AVAILABLE", False):
            # Re-import to get NumPy version of decorator
            from quantiq.backend.operations import jit as jit_numpy

            @jit_numpy
            def multiply(x, y):
                return x * y

            arr1 = np.array([1, 2, 3])
            arr2 = np.array([4, 5, 6])
            result = multiply(arr1, arr2)

            expected = np.array([4, 10, 18])
            assert np.array_equal(result, expected)


class TestVectorization:
    """Test vectorization decorator."""

    def test_vmap_basic(self):
        """Test vmap with simple function."""

        def add_one(x):
            return x + 1

        batched_add = vmap(add_one)
        arr = jnp.array([1.0, 2.0, 3.0])
        result = batched_add(arr)

        expected = jnp.array([2.0, 3.0, 4.0])
        assert jnp.allclose(result, expected)

    def test_vmap_with_in_axes(self):
        """Test vmap with specified input axes."""
        if not is_jax_available():
            pytest.skip("vmap with in_axes requires JAX backend")

        def multiply(x, y):
            return x * y

        batched_multiply = vmap(multiply, in_axes=(0, 0))
        arr1 = jnp.array([1.0, 2.0, 3.0])
        arr2 = jnp.array([4.0, 5.0, 6.0])
        result = batched_multiply(arr1, arr2)

        expected = jnp.array([4.0, 10.0, 18.0])
        assert jnp.allclose(result, expected)

    def test_vmap_with_out_axes(self):
        """Test vmap with specified output axis."""
        if not is_jax_available():
            pytest.skip("vmap with out_axes requires JAX backend")

        def identity(x):
            return x

        batched_identity = vmap(identity, out_axes=0)
        arr = jnp.array([1.0, 2.0, 3.0])
        result = batched_identity(arr)

        assert jnp.allclose(result, arr)

    def test_vmap_numpy_fallback_single_arg(self):
        """Test vmap with NumPy backend fallback (single argument)."""
        with patch("quantiq.backend.operations._JAX_AVAILABLE", False):
            from quantiq.backend.operations import vmap as vmap_numpy

            def square(x):
                return x * x

            batched_square = vmap_numpy(square)
            arr = np.array([1.0, 2.0, 3.0])
            result = batched_square(arr)

            expected = np.array([1.0, 4.0, 9.0])
            assert np.allclose(result, expected)

    def test_vmap_numpy_fallback_multiple_args(self):
        """Test vmap with NumPy backend raises NotImplementedError for multiple args."""
        with patch("quantiq.backend.operations._JAX_AVAILABLE", False):
            from quantiq.backend.operations import vmap as vmap_numpy

            def add(x, y):
                return x + y

            batched_add = vmap_numpy(add)

            arr1 = np.array([1.0, 2.0, 3.0])
            arr2 = np.array([4.0, 5.0, 6.0])

            with pytest.raises(NotImplementedError) as exc_info:
                batched_add(arr1, arr2)

            assert "NumPy backend vmap with multiple inputs" in str(exc_info.value)

    def test_vmap_numpy_fallback_no_args(self):
        """Test vmap with NumPy backend and no arguments."""
        with patch("quantiq.backend.operations._JAX_AVAILABLE", False):
            from quantiq.backend.operations import vmap as vmap_numpy

            def constant():
                return 42

            batched_constant = vmap_numpy(constant)
            result = batched_constant()

            assert result == 42


class TestGradient:
    """Test gradient computation decorator."""

    def test_grad_basic(self):
        """Test grad with simple function."""
        if not is_jax_available():
            pytest.skip("grad requires JAX backend")

        def loss(x):
            return jnp.sum(x**2)

        grad_loss = grad(loss)
        arr = jnp.array([1.0, 2.0, 3.0])
        gradient = grad_loss(arr)

        # Gradient of sum(x^2) is 2*x
        expected = jnp.array([2.0, 4.0, 6.0])
        assert jnp.allclose(gradient, expected)

    def test_grad_with_argnums(self):
        """Test grad with specified argument indices."""
        if not is_jax_available():
            pytest.skip("grad with argnums requires JAX backend")

        def loss(x, y):
            return jnp.sum(x**2) + jnp.sum(y**2)

        # Gradient with respect to first argument
        grad_loss_x = grad(loss, argnums=0)
        arr_x = jnp.array([1.0, 2.0])
        arr_y = jnp.array([3.0, 4.0])
        gradient = grad_loss_x(arr_x, arr_y)

        expected = jnp.array([2.0, 4.0])
        assert jnp.allclose(gradient, expected)

    def test_grad_with_kwargs(self):
        """Test grad with keyword arguments."""
        if not is_jax_available():
            pytest.skip("grad with kwargs requires JAX backend")

        def loss(x):
            return jnp.sum(x**2)

        grad_loss = grad(loss, has_aux=False)
        arr = jnp.array([1.0, 2.0, 3.0])
        gradient = grad_loss(arr)

        expected = jnp.array([2.0, 4.0, 6.0])
        assert jnp.allclose(gradient, expected)

    def test_grad_numpy_fallback(self):
        """Test grad with NumPy backend raises NotImplementedError."""
        with patch("quantiq.backend.operations._JAX_AVAILABLE", False):
            from quantiq.backend.operations import grad as grad_numpy

            def loss(x):
                return np.sum(x**2)

            grad_loss = grad_numpy(loss)
            arr = np.array([1.0, 2.0, 3.0])

            with pytest.raises(NotImplementedError) as exc_info:
                grad_loss(arr)

            assert "Automatic differentiation requires JAX backend" in str(exc_info.value)


class TestDeviceManagement:
    """Test device management functions."""

    def test_device_put_basic(self):
        """Test device_put without specifying device."""
        arr = jnp.array([1, 2, 3])
        arr_on_device = device_put(arr)

        # Result should be usable in operations
        result = jnp.sum(arr_on_device)
        assert float(result) == 6.0

    def test_device_put_with_device(self):
        """Test device_put with specified device."""
        if not is_jax_available():
            pytest.skip("device_put with device requires JAX backend")

        import jax

        arr = jnp.array([1.0, 2.0, 3.0])
        devices = jax.devices()

        if len(devices) > 0:
            arr_on_device = device_put(arr, devices[0])
            result = jnp.sum(arr_on_device)
            assert jnp.allclose(result, 6.0)

    def test_device_put_numpy_fallback(self):
        """Test device_put with NumPy backend (no-op)."""
        with patch("quantiq.backend.operations._JAX_AVAILABLE", False):
            from quantiq.backend.operations import device_put as device_put_numpy

            arr = np.array([1, 2, 3])
            result = device_put_numpy(arr)

            # Should return same array
            assert np.array_equal(result, arr)

    def test_device_put_numpy_fallback_with_device(self):
        """Test device_put with NumPy backend ignores device parameter."""
        with patch("quantiq.backend.operations._JAX_AVAILABLE", False):
            from quantiq.backend.operations import device_put as device_put_numpy

            arr = np.array([1, 2, 3])
            result = device_put_numpy(arr, device="fake_device")

            # Should return same array, ignoring device parameter
            assert np.array_equal(result, arr)

    def test_device_get_basic(self):
        """Test device_get converts to NumPy array."""
        arr = jnp.array([1.0, 2.0, 3.0])
        np_arr = device_get(arr)

        assert isinstance(np_arr, np.ndarray)
        assert np.array_equal(np_arr, np.array([1.0, 2.0, 3.0]))

    def test_device_get_with_numpy_input(self):
        """Test device_get with NumPy array input."""
        np_arr = np.array([1.0, 2.0, 3.0])
        result = device_get(np_arr)

        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np_arr)


class TestTypeConversions:
    """Test type conversion utilities."""

    def test_ensure_array_basic(self):
        """Test ensure_array with various inputs."""
        # From list
        arr_from_list = ensure_array([1, 2, 3])
        assert arr_from_list.shape == (3,)
        assert jnp.array_equal(arr_from_list, jnp.array([1, 2, 3]))

        # From scalar
        arr_from_scalar = ensure_array(5.0)
        assert arr_from_scalar.shape == ()
        assert float(arr_from_scalar) == 5.0

        # From NumPy array
        np_arr = np.array([1, 2, 3])
        arr_from_numpy = ensure_array(np_arr)
        assert jnp.array_equal(arr_from_numpy, jnp.array([1, 2, 3]))

    def test_ensure_array_with_dtype(self):
        """Test ensure_array with dtype conversion."""
        # Integer to float
        arr_int = ensure_array([1, 2, 3], dtype=jnp.float32)
        assert arr_int.dtype == jnp.float32

        # Float to integer
        arr_float = ensure_array([1.5, 2.5, 3.5], dtype=jnp.int32)
        assert arr_float.dtype == jnp.int32

    def test_astype_basic(self):
        """Test astype for dtype conversion."""
        arr = jnp.array([1, 2, 3])

        # Convert to float
        arr_float = astype(arr, jnp.float32)
        assert arr_float.dtype == jnp.float32
        assert jnp.array_equal(arr_float, jnp.array([1.0, 2.0, 3.0]))

        # Convert to int32 (int64 may not be available in JAX without x64 mode)
        arr_int32 = astype(arr_float, jnp.int32)
        assert arr_int32.dtype == jnp.int32

    def test_astype_with_complex_types(self):
        """Test astype with complex data types."""
        arr = jnp.array([1.0, 2.0, 3.0])

        # Convert to complex
        arr_complex = astype(arr, jnp.complex64)
        assert arr_complex.dtype == jnp.complex64
        assert jnp.allclose(arr_complex.real, arr)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_array_operations(self):
        """Test operations with empty arrays."""
        empty = jnp.array([])

        # Test copy
        empty_copy = copy(empty)
        assert empty_copy.shape == (0,)

        # Test reshape
        reshaped = reshape(empty, (0, 5))
        assert reshaped.shape == (0, 5)

    def test_concatenate_empty_list(self):
        """Test concatenate with empty list raises error."""
        with pytest.raises((ValueError, TypeError)):
            concatenate([])

    def test_stack_single_array(self):
        """Test stack with single array."""
        arr = jnp.array([1, 2, 3])
        result = stack([arr])
        assert result.shape == (1, 3)

    def test_operations_preserve_dtype(self):
        """Test that operations preserve dtype when appropriate."""
        arr_int = jnp.array([1, 2, 3], dtype=jnp.int32)
        # Use float32 instead of float64 to avoid JAX x64 mode requirement
        arr_float = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)

        # Copy should preserve dtype
        assert copy(arr_int).dtype == jnp.int32
        assert copy(arr_float).dtype == jnp.float32

        # Reshape should preserve dtype
        assert reshape(arr_int, (3, 1)).dtype == jnp.int32


class TestIntegration:
    """Test integration scenarios combining multiple operations."""

    def test_jit_with_array_operations(self):
        """Test JIT compilation with array operations."""

        @jit
        def process_array(x):
            y = copy(x)
            z = reshape(y, (-1, 1))
            return jnp.sum(z)

        arr = jnp.array([1.0, 2.0, 3.0, 4.0])
        result = process_array(arr)
        assert jnp.allclose(result, 10.0)

    def test_vmap_with_jit(self):
        """Test combining vmap with jit."""
        if not is_jax_available():
            pytest.skip("vmap + jit requires JAX backend")

        @jit
        def square(x):
            return x**2

        batched_square = vmap(square)
        arr = jnp.array([1.0, 2.0, 3.0])
        result = batched_square(arr)

        expected = jnp.array([1.0, 4.0, 9.0])
        assert jnp.allclose(result, expected)

    def test_pipeline_with_type_conversions(self):
        """Test pipeline with type conversions."""
        # Create array
        arr = ensure_array([1, 2, 3, 4], dtype=jnp.float32)

        # Process with operations
        arr_reshaped = reshape(arr, (2, 2))
        arr_stacked = stack([arr_reshaped, arr_reshaped])

        # Convert back to NumPy
        np_result = device_get(arr_stacked)

        assert isinstance(np_result, np.ndarray)
        assert np_result.shape == (2, 2, 2)
