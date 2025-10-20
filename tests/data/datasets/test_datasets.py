"""
Tests for quantiq dataset classes.

Tests cover all 7 dataset types with focus on:
- Dataset ABC interface
- Internal JAX/NumPy backend arrays
- External NumPy returns from properties
- Metadata system (conditions, details)
- Validation and error handling
"""

import numpy as np
import pytest

from quantiq.backend import get_backend, is_jax_available, jnp
from quantiq.data.datasets import (
    Dataset,
    Distribution,
    Histogram,
    OneDimensionalCompositeDataset,
    OneDimensionalDataset,
    ThreeDimensionalDataset,
    TwoDimensionalDataset,
    ZeroDimensionalDataset,
)


class TestDatasetABC:
    """Test Dataset abstract base class."""

    def test_dataset_is_abc(self):
        """Dataset is an ABC and can be instantiated with metadata."""
        from abc import ABC

        # Dataset is an ABC
        assert issubclass(Dataset, ABC)

        # Dataset can be instantiated since it has no abstract methods
        # (it's a concrete ABC that provides metadata functionality)
        dataset = Dataset(conditions={"temp": 25.0}, details={"operator": "test"})
        assert dataset.conditions == {"temp": 25.0}
        assert dataset.details == {"operator": "test"}


class TestZeroDimensionalDataset:
    """Test 0D dataset (single scalar value)."""

    def test_creation_and_value_access(self):
        """Test creating 0D dataset and accessing value."""
        dataset = ZeroDimensionalDataset(value=42.5)

        assert dataset.value == 42.5
        assert isinstance(dataset.value, (float, np.floating))

    def test_with_metadata(self, sample_metadata):
        """Test 0D dataset with conditions and details."""
        conditions, details = sample_metadata
        dataset = ZeroDimensionalDataset(value=100.0, conditions=conditions, details=details)

        assert dataset.value == 100.0
        assert dataset.conditions == conditions
        assert dataset.details == details
        assert "temperature" in dataset.conditions
        assert dataset.conditions["temperature"] == 25.0


class TestOneDimensionalDataset:
    """Test 1D dataset (most common type)."""

    def test_creation_from_numpy(self, sample_1d_data):
        """Test creating 1D dataset from NumPy arrays."""
        x, y = sample_1d_data
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        # Properties should return NumPy arrays
        assert isinstance(dataset.independent_variable_data, np.ndarray)
        assert isinstance(dataset.dependent_variable_data, np.ndarray)

        # Data should match input (within floating point tolerance)
        np.testing.assert_allclose(dataset.independent_variable_data, x)
        np.testing.assert_allclose(dataset.dependent_variable_data, y)

    def test_internal_backend_arrays(self):
        """Test that internal storage uses backend arrays."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])

        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        # Internal arrays should be backend type
        backend = get_backend()
        if backend == "jax":
            # Check internal storage is JAX array
            import jax.numpy as jnp_jax

            assert isinstance(dataset._independent_variable_data, jnp_jax.ndarray)
            assert isinstance(dataset._dependent_variable_data, jnp_jax.ndarray)
        else:
            # NumPy backend
            assert isinstance(dataset._independent_variable_data, np.ndarray)
            assert isinstance(dataset._dependent_variable_data, np.ndarray)

    def test_dimension_validation(self):
        """Test that dimension mismatch raises ValueError."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0])  # Wrong length

        with pytest.raises(ValueError, match="same shape"):
            OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)


class TestTwoDimensionalDataset:
    """Test 2D dataset."""

    def test_creation_with_2d_dependent(self, sample_2d_data):
        """Test creating 2D dataset with 2D dependent variable."""
        x, y, Z = sample_2d_data

        dataset = TwoDimensionalDataset(
            independent_variable_data_1=x, independent_variable_data_2=y, dependent_variable_data=Z
        )

        # All should return NumPy arrays
        assert isinstance(dataset.independent_variable_data_1, np.ndarray)
        assert isinstance(dataset.independent_variable_data_2, np.ndarray)
        assert isinstance(dataset.dependent_variable_data, np.ndarray)

        # Check shapes
        assert dataset.independent_variable_data_1.shape == (50,)
        assert dataset.independent_variable_data_2.shape == (50,)
        assert dataset.dependent_variable_data.shape == (50, 50)

    def test_dimension_validation(self):
        """Test dimension validation for 2D dataset."""
        x = np.linspace(0, 5, 10)
        y = np.linspace(0, 5, 20)
        Z = np.random.randn(10, 15)  # Wrong shape

        with pytest.raises(ValueError, match="dimension"):
            TwoDimensionalDataset(
                independent_variable_data_1=x,
                independent_variable_data_2=y,
                dependent_variable_data=Z,
            )


class TestThreeDimensionalDataset:
    """Test 3D dataset."""

    def test_creation_with_3d_dependent(self):
        """Test creating 3D dataset."""
        x = np.linspace(0, 5, 10)
        y = np.linspace(0, 5, 20)
        z = np.linspace(0, 5, 15)
        data = np.random.randn(10, 20, 15)

        dataset = ThreeDimensionalDataset(
            independent_variable_data_1=x,
            independent_variable_data_2=y,
            independent_variable_data_3=z,
            dependent_variable_data=data,
        )

        # Check all properties return NumPy
        assert isinstance(dataset.independent_variable_data_1, np.ndarray)
        assert isinstance(dataset.independent_variable_data_2, np.ndarray)
        assert isinstance(dataset.independent_variable_data_3, np.ndarray)
        assert isinstance(dataset.dependent_variable_data, np.ndarray)

        # Check shapes
        assert dataset.dependent_variable_data.shape == (10, 20, 15)


class TestHistogram:
    """Test Histogram dataset type."""

    def test_creation_with_bin_edges_and_counts(self):
        """Test histogram with bin edges and counts."""
        # Variable-width bins
        bin_edges = np.array([0, 1, 3, 6, 10])  # 4 bins
        counts = np.array([5, 12, 8, 3])  # 4 counts

        hist = Histogram(bin_edges=bin_edges, counts=counts)

        assert isinstance(hist.bin_edges, np.ndarray)
        assert isinstance(hist.counts, np.ndarray)
        np.testing.assert_array_equal(hist.bin_edges, bin_edges)
        np.testing.assert_array_equal(hist.counts, counts)

    def test_bin_count_validation(self):
        """Test that bin edges and counts must be compatible."""
        bin_edges = np.array([0, 1, 2, 3])  # 3 bins
        counts = np.array([5, 10])  # Wrong: only 2 counts

        with pytest.raises(ValueError, match="bins"):
            Histogram(bin_edges=bin_edges, counts=counts)


class TestDistribution:
    """Test Distribution dataset type."""

    def test_creation_with_variable_and_pdf(self):
        """Test distribution with variable data and probability density."""
        # Molecular weight distribution
        molecular_weights = np.linspace(1000, 10000, 100)
        pdf = np.exp(-((molecular_weights - 5000) ** 2) / 1000000)
        pdf = pdf / np.trapezoid(pdf, molecular_weights)  # Normalize

        dist = Distribution(variable_data=molecular_weights, probability_density=pdf)

        assert isinstance(dist.variable_data, np.ndarray)
        assert isinstance(dist.probability_density, np.ndarray)
        assert len(dist.variable_data) == len(dist.probability_density)

    def test_dimension_validation(self):
        """Test that variable and pdf must have same length."""
        variable = np.linspace(0, 10, 100)
        pdf = np.ones(50)  # Wrong length

        with pytest.raises(ValueError, match="same shape"):
            Distribution(variable_data=variable, probability_density=pdf)


class TestOneDimensionalCompositeDataset:
    """Test composite dataset with multiple dependent variables."""

    def test_creation_with_multiple_channels(self):
        """Test composite dataset with multiple dependent variables."""
        # Multi-channel instrument data
        time = np.linspace(0, 10, 100)
        channel1 = np.sin(time)
        channel2 = np.cos(time)
        channel3 = np.sin(2 * time)

        dataset = OneDimensionalCompositeDataset(
            independent_variable_data=time,
            dependent_variable_data_list=[channel1, channel2, channel3],
        )

        assert isinstance(dataset.independent_variable_data, np.ndarray)
        assert isinstance(dataset.dependent_variable_data_list, list)
        assert len(dataset.dependent_variable_data_list) == 3

        # Each channel should be NumPy array
        for channel in dataset.dependent_variable_data_list:
            assert isinstance(channel, np.ndarray)
            assert len(channel) == len(time)

    def test_validation_all_same_length(self):
        """Test that all channels must match independent variable length."""
        time = np.linspace(0, 10, 100)
        channel1 = np.sin(time)
        channel2 = np.cos(time)[:50]  # Wrong length

        with pytest.raises(ValueError, match="same length"):
            OneDimensionalCompositeDataset(
                independent_variable_data=time, dependent_variable_data_list=[channel1, channel2]
            )

    def test_empty_channels_validation(self):
        """Test that at least one dependent variable is required."""
        time = np.linspace(0, 10, 100)

        with pytest.raises(ValueError, match="at least one"):
            OneDimensionalCompositeDataset(
                independent_variable_data=time, dependent_variable_data_list=[]
            )
