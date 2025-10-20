"""
Tests for region-based transform infrastructure.

This module tests:
- LinearRegion creation and bounds validation
- LinearRegion mask generation
- CompoundRegion with multiple sub-regions
- RegionTransform application to 1D datasets
- Preservation of untransformed regions
"""

import numpy as np
import pytest

from quantiq.data.datasets import OneDimensionalDataset
from quantiq.data.roi import CompoundRegion, LinearRegion
from quantiq.transform.region import RegionMultiplyTransform, RegionTransform


class TestLinearRegion:
    """Tests for LinearRegion class."""

    def test_linear_region_creation_valid(self):
        """Test LinearRegion creation with valid bounds."""
        region = LinearRegion(x_min=2.0, x_max=5.0)
        assert region.x_min == 2.0
        assert region.x_max == 5.0

    def test_linear_region_validation(self):
        """Test LinearRegion validation (x_min < x_max)."""
        # Should raise ValueError if x_min >= x_max
        with pytest.raises(ValueError, match="x_min.*must be less than.*x_max"):
            LinearRegion(x_min=5.0, x_max=2.0)

        with pytest.raises(ValueError, match="x_min.*must be less than.*x_max"):
            LinearRegion(x_min=3.0, x_max=3.0)

    def test_linear_region_mask_generation(self):
        """Test LinearRegion mask generation."""
        region = LinearRegion(x_min=2.0, x_max=5.0)
        x_data = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

        mask = region.get_mask(x_data)
        expected_mask = np.array([False, False, True, True, True, True, False, False])

        np.testing.assert_array_equal(mask, expected_mask)

    def test_linear_region_repr(self):
        """Test LinearRegion string representation."""
        region = LinearRegion(x_min=2.0, x_max=5.0)
        repr_str = repr(region)
        assert "LinearRegion" in repr_str
        assert "2.0" in repr_str
        assert "5.0" in repr_str


class TestCompoundRegion:
    """Tests for CompoundRegion class."""

    def test_compound_region_creation(self):
        """Test CompoundRegion with multiple sub-regions."""
        region1 = LinearRegion(x_min=1.0, x_max=2.0)
        region2 = LinearRegion(x_min=4.0, x_max=5.0)

        compound = CompoundRegion([region1, region2])
        assert len(compound) == 2
        assert compound[0] is region1
        assert compound[1] is region2

    def test_compound_region_empty_list(self):
        """Test CompoundRegion requires at least one region."""
        with pytest.raises(ValueError, match="requires at least one region"):
            CompoundRegion([])

    def test_compound_region_invalid_types(self):
        """Test CompoundRegion validates region types."""
        with pytest.raises(TypeError, match="All regions must be LinearRegion"):
            CompoundRegion([LinearRegion(1.0, 2.0), "not a region"])

    def test_compound_region_mask_generation(self):
        """Test CompoundRegion combined mask (union)."""
        region1 = LinearRegion(x_min=1.0, x_max=2.0)
        region2 = LinearRegion(x_min=4.0, x_max=5.0)
        compound = CompoundRegion([region1, region2])

        x_data = np.array([0.0, 1.0, 1.5, 2.0, 3.0, 4.0, 4.5, 5.0, 6.0])
        mask = compound.get_mask(x_data)

        # Should be True for points in [1.0, 2.0] OR [4.0, 5.0]
        expected_mask = np.array([False, True, True, True, False, True, True, True, False])
        np.testing.assert_array_equal(mask, expected_mask)

    def test_compound_region_repr(self):
        """Test CompoundRegion string representation."""
        region1 = LinearRegion(x_min=1.0, x_max=2.0)
        region2 = LinearRegion(x_min=4.0, x_max=5.0)
        compound = CompoundRegion([region1, region2])

        repr_str = repr(compound)
        assert "CompoundRegion" in repr_str
        assert "2" in repr_str  # Number of regions


class TestRegionTransform:
    """Tests for RegionTransform and concrete implementations."""

    def test_region_multiply_transform_single_region(self):
        """Test RegionMultiplyTransform application to single region."""
        # Create dataset: x from 0 to 10, y all ones
        x_data = np.linspace(0, 10, 11)
        y_data = np.ones(11)
        dataset = OneDimensionalDataset(
            independent_variable_data=x_data, dependent_variable_data=y_data
        )

        # Define region [3, 7] and multiply by 2.0
        region = LinearRegion(x_min=3.0, x_max=7.0)
        transform = RegionMultiplyTransform(region, factor=2.0)

        # Apply transform
        result = transform.apply_to(dataset, make_copy=True)

        # Check that region [3, 7] is multiplied by 2.0
        expected_y = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(result.dependent_variable_data, expected_y)

        # Verify original dataset is unchanged (make_copy=True)
        np.testing.assert_array_equal(dataset.dependent_variable_data, np.ones(11))

    def test_region_transform_preserves_untransformed_data(self):
        """Test that untransformed regions are preserved."""
        # Create dataset with varying values
        x_data = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        y_data = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        dataset = OneDimensionalDataset(
            independent_variable_data=x_data, dependent_variable_data=y_data
        )

        # Transform only region [2.0, 4.0] by multiplying by 0.5
        region = LinearRegion(x_min=2.0, x_max=4.0)
        transform = RegionMultiplyTransform(region, factor=0.5)

        result = transform.apply_to(dataset, make_copy=True)

        # Expected: only indices 2, 3, 4 (x=2, 3, 4) are transformed
        expected_y = np.array([10.0, 20.0, 15.0, 20.0, 25.0, 60.0])
        np.testing.assert_array_almost_equal(result.dependent_variable_data, expected_y)

    def test_region_transform_compound_region(self):
        """Test RegionTransform with CompoundRegion (multiple disjoint regions)."""
        # Create dataset
        x_data = np.linspace(0, 10, 21)  # 0, 0.5, 1.0, ..., 10.0
        y_data = np.ones(21)
        dataset = OneDimensionalDataset(
            independent_variable_data=x_data, dependent_variable_data=y_data
        )

        # Define two disjoint regions: [1, 2] and [7, 8]
        region1 = LinearRegion(x_min=1.0, x_max=2.0)
        region2 = LinearRegion(x_min=7.0, x_max=8.0)
        compound = CompoundRegion([region1, region2])

        # Multiply both regions by 3.0
        transform = RegionMultiplyTransform(compound, factor=3.0)
        result = transform.apply_to(dataset, make_copy=True)

        # Check that only points in [1, 2] and [7, 8] are multiplied
        result_y = result.dependent_variable_data

        # Points at x=1.0, 1.5, 2.0 should be 3.0
        assert result_y[2] == 3.0  # x=1.0
        assert result_y[3] == 3.0  # x=1.5
        assert result_y[4] == 3.0  # x=2.0

        # Points at x=7.0, 7.5, 8.0 should be 3.0
        assert result_y[14] == 3.0  # x=7.0
        assert result_y[15] == 3.0  # x=7.5
        assert result_y[16] == 3.0  # x=8.0

        # Points outside regions should remain 1.0
        assert result_y[0] == 1.0  # x=0.0
        assert result_y[10] == 1.0  # x=5.0
        assert result_y[20] == 1.0  # x=10.0

    def test_region_transform_type_validation(self):
        """Test RegionTransform validates dataset type."""
        region = LinearRegion(x_min=1.0, x_max=2.0)
        transform = RegionMultiplyTransform(region, factor=2.0)

        # Should raise TypeError if not a Dataset at all (caught by DatasetTransform)
        with pytest.raises(TypeError, match="DatasetTransform requires Dataset"):
            transform.apply_to("not a dataset")

    def test_region_transform_inplace(self):
        """Test RegionTransform can work in-place (make_copy=False)."""
        x_data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y_data = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        dataset = OneDimensionalDataset(
            independent_variable_data=x_data, dependent_variable_data=y_data
        )

        region = LinearRegion(x_min=1.0, x_max=3.0)
        transform = RegionMultiplyTransform(region, factor=2.0)

        # Apply in-place
        result = transform.apply_to(dataset, make_copy=False)

        # Result and dataset should be the same object
        assert result is dataset

        # Check transformation applied
        expected_y = np.array([1.0, 2.0, 2.0, 2.0, 1.0])
        np.testing.assert_array_almost_equal(dataset.dependent_variable_data, expected_y)
