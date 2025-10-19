"""
Tests for region-based transforms.

This module tests RegionTransform and RegionMultiplyTransform classes.
"""

import pytest
import numpy as np
from quantiq.data.datasets import OneDimensionalDataset, TwoDimensionalDataset
from quantiq.data.roi import LinearRegion, CompoundRegion
from quantiq.transform.region import RegionTransform, RegionMultiplyTransform


class TestRegionTransform:
    """Test base RegionTransform class."""

    def test_init(self):
        """Test RegionTransform initialization."""
        region = LinearRegion(x_min=1.0, x_max=5.0)
        transform = RegionMultiplyTransform(region, factor=2.0)
        assert transform.region == region
        assert transform.factor == 2.0

    def test_apply_with_linear_region(self):
        """Test applying transform with LinearRegion."""
        # Create dataset
        x_data = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        y_data = np.ones(6)
        dataset = OneDimensionalDataset(
            independent_variable_data=x_data,
            dependent_variable_data=y_data
        )

        # Create transform
        region = LinearRegion(x_min=2.0, x_max=4.0)
        transform = RegionMultiplyTransform(region, factor=2.0)

        # Apply transform
        result = transform.apply_to(dataset, make_copy=True)

        # Check result
        expected = np.array([1.0, 1.0, 2.0, 2.0, 2.0, 1.0])
        np.testing.assert_array_almost_equal(
            result.dependent_variable_data, expected
        )

    def test_apply_with_compound_region(self):
        """Test applying transform with CompoundRegion."""
        # Create dataset
        x_data = np.linspace(0, 10, 11)
        y_data = np.ones(11)
        dataset = OneDimensionalDataset(
            independent_variable_data=x_data,
            dependent_variable_data=y_data
        )

        # Create compound region (two disjoint regions)
        region1 = LinearRegion(x_min=1.0, x_max=2.0)
        region2 = LinearRegion(x_min=8.0, x_max=9.0)
        compound = CompoundRegion([region1, region2])
        transform = RegionMultiplyTransform(compound, factor=0.5)

        # Apply transform
        result = transform.apply_to(dataset, make_copy=True)

        # Check result - points in [1, 2] OR [8, 9] should be multiplied by 0.5
        expected = np.ones(11)
        expected[1:3] = 0.5  # indices 1, 2
        expected[8:10] = 0.5  # indices 8, 9
        np.testing.assert_array_almost_equal(
            result.dependent_variable_data, expected
        )

    def test_apply_preserves_outside_region(self):
        """Test that data outside region is preserved."""
        x_data = np.linspace(0, 10, 101)
        y_data = np.sin(x_data)
        dataset = OneDimensionalDataset(
            independent_variable_data=x_data,
            dependent_variable_data=y_data
        )

        # Transform only middle region
        region = LinearRegion(x_min=4.0, x_max=6.0)
        transform = RegionMultiplyTransform(region, factor=3.0)
        result = transform.apply_to(dataset, make_copy=True)

        # Check data outside region is preserved
        mask = region.get_mask(x_data)
        outside_mask = ~mask
        np.testing.assert_array_almost_equal(
            result.dependent_variable_data[outside_mask],
            dataset.dependent_variable_data[outside_mask]
        )

    def test_apply_to_wrong_dataset_type(self):
        """Test that RegionTransform raises TypeError for wrong dataset type."""
        dataset = TwoDimensionalDataset(
            independent_variable_data_1=np.array([1, 2]),
            independent_variable_data_2=np.array([3, 4]),
            dependent_variable_data=np.array([[1, 2], [3, 4]])
        )
        region = LinearRegion(x_min=1.0, x_max=5.0)
        transform = RegionMultiplyTransform(region, factor=2.0)

        with pytest.raises(TypeError, match="only works with OneDimensionalDataset"):
            transform.apply_to(dataset)

    def test_region_transform_not_implemented(self):
        """Test that base RegionTransform._apply_to_region raises NotImplementedError."""
        region = LinearRegion(x_min=1.0, x_max=5.0)
        transform = RegionTransform(region)

        x_region = np.array([1, 2, 3])
        y_region = np.array([1, 1, 1])

        with pytest.raises(NotImplementedError, match="must implement _apply_to_region"):
            transform._apply_to_region(x_region, y_region)


class TestRegionMultiplyTransform:
    """Test RegionMultiplyTransform class."""

    def test_multiply_factor_positive(self):
        """Test multiplication with positive factor."""
        x_data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dataset = OneDimensionalDataset(
            independent_variable_data=x_data,
            dependent_variable_data=y_data
        )

        region = LinearRegion(x_min=1.0, x_max=3.0)
        transform = RegionMultiplyTransform(region, factor=10.0)
        result = transform.apply_to(dataset, make_copy=True)

        expected = np.array([1.0, 20.0, 30.0, 40.0, 5.0])
        np.testing.assert_array_almost_equal(
            result.dependent_variable_data, expected
        )

    def test_multiply_factor_negative(self):
        """Test multiplication with negative factor."""
        x_data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y_data = np.ones(5)
        dataset = OneDimensionalDataset(
            independent_variable_data=x_data,
            dependent_variable_data=y_data
        )

        region = LinearRegion(x_min=1.0, x_max=3.0)
        transform = RegionMultiplyTransform(region, factor=-1.0)
        result = transform.apply_to(dataset, make_copy=True)

        expected = np.array([1.0, -1.0, -1.0, -1.0, 1.0])
        np.testing.assert_array_almost_equal(
            result.dependent_variable_data, expected
        )

    def test_multiply_factor_zero(self):
        """Test multiplication with zero factor."""
        x_data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y_data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        dataset = OneDimensionalDataset(
            independent_variable_data=x_data,
            dependent_variable_data=y_data
        )

        region = LinearRegion(x_min=1.0, x_max=3.0)
        transform = RegionMultiplyTransform(region, factor=0.0)
        result = transform.apply_to(dataset, make_copy=True)

        expected = np.array([5.0, 0.0, 0.0, 0.0, 5.0])
        np.testing.assert_array_almost_equal(
            result.dependent_variable_data, expected
        )

    def test_multiply_empty_region(self):
        """Test multiplication when region is empty."""
        x_data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dataset = OneDimensionalDataset(
            independent_variable_data=x_data,
            dependent_variable_data=y_data
        )

        # Region outside data range
        region = LinearRegion(x_min=10.0, x_max=20.0)
        transform = RegionMultiplyTransform(region, factor=2.0)
        result = transform.apply_to(dataset, make_copy=True)

        # Nothing should change
        np.testing.assert_array_almost_equal(
            result.dependent_variable_data,
            dataset.dependent_variable_data
        )

    def test_multiply_make_copy_false(self):
        """Test in-place multiplication."""
        x_data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dataset = OneDimensionalDataset(
            independent_variable_data=x_data,
            dependent_variable_data=y_data
        )

        region = LinearRegion(x_min=1.0, x_max=3.0)
        transform = RegionMultiplyTransform(region, factor=2.0)
        result = transform.apply_to(dataset, make_copy=False)

        # Result should be same object as input
        assert result is dataset

        expected = np.array([1.0, 4.0, 6.0, 8.0, 5.0])
        np.testing.assert_array_almost_equal(
            result.dependent_variable_data, expected
        )
