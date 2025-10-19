"""Tests for ROI (Region of Interest) classes.

This module tests:
- LinearRegion creation and masking
- CompoundRegion union of multiple regions
- Edge cases and error handling
- Boundary conditions (inclusive bounds)
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from quantiq.data.roi import LinearRegion, CompoundRegion


class TestLinearRegion:
    """Test LinearRegion class for 1D contiguous regions."""

    def test_creation_with_valid_bounds(self):
        """Test creating LinearRegion with valid bounds."""
        region = LinearRegion(x_min=2.0, x_max=5.0)

        assert region.x_min == 2.0
        assert region.x_max == 5.0

    def test_creation_with_equal_bounds_raises(self):
        """Test that equal x_min and x_max raises ValueError."""
        with pytest.raises(ValueError, match="x_min .* must be less than x_max"):
            LinearRegion(x_min=3.0, x_max=3.0)

    def test_creation_with_inverted_bounds_raises(self):
        """Test that x_min > x_max raises ValueError."""
        with pytest.raises(ValueError, match="x_min .* must be less than x_max"):
            LinearRegion(x_min=5.0, x_max=2.0)

    def test_get_mask_basic(self):
        """Test basic mask generation for data within region."""
        region = LinearRegion(x_min=2.0, x_max=5.0)
        x_data = np.array([0, 1, 2, 3, 4, 5, 6, 7])

        mask = region.get_mask(x_data)
        expected = np.array([False, False, True, True, True, True, False, False])

        assert_array_equal(mask, expected)

    def test_get_mask_inclusive_bounds(self):
        """Test that bounds are inclusive (x_min and x_max included)."""
        region = LinearRegion(x_min=2.0, x_max=5.0)
        x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        mask = region.get_mask(x_data)
        expected = np.array([False, True, True, True, True, False])

        assert_array_equal(mask, expected)
        # Verify boundary values are included
        assert mask[1] == True  # x_min = 2.0
        assert mask[4] == True  # x_max = 5.0

    def test_get_mask_no_overlap(self):
        """Test mask when no data falls within region."""
        region = LinearRegion(x_min=10.0, x_max=20.0)
        x_data = np.array([0, 1, 2, 3, 4, 5])

        mask = region.get_mask(x_data)
        expected = np.array([False, False, False, False, False, False])

        assert_array_equal(mask, expected)

    def test_get_mask_all_overlap(self):
        """Test mask when all data falls within region."""
        region = LinearRegion(x_min=0.0, x_max=10.0)
        x_data = np.array([1, 2, 3, 4, 5])

        mask = region.get_mask(x_data)
        expected = np.array([True, True, True, True, True])

        assert_array_equal(mask, expected)

    def test_get_mask_with_floats(self):
        """Test mask generation with floating point data."""
        region = LinearRegion(x_min=1.5, x_max=3.7)
        x_data = np.array([1.0, 1.5, 2.3, 3.7, 4.2])

        mask = region.get_mask(x_data)
        expected = np.array([False, True, True, True, False])

        assert_array_equal(mask, expected)

    def test_get_mask_with_negative_bounds(self):
        """Test region with negative bounds."""
        region = LinearRegion(x_min=-5.0, x_max=-2.0)
        x_data = np.array([-6, -5, -4, -3, -2, -1, 0])

        mask = region.get_mask(x_data)
        expected = np.array([False, True, True, True, True, False, False])

        assert_array_equal(mask, expected)

    def test_repr(self):
        """Test string representation of LinearRegion."""
        region = LinearRegion(x_min=2.5, x_max=7.3)

        repr_str = repr(region)

        assert "LinearRegion" in repr_str
        assert "2.5" in repr_str
        assert "7.3" in repr_str


class TestCompoundRegion:
    """Test CompoundRegion class for unions of multiple regions."""

    def test_creation_with_single_region(self):
        """Test creating CompoundRegion with one region."""
        region = LinearRegion(x_min=1.0, x_max=2.0)
        compound = CompoundRegion([region])

        assert len(compound) == 1
        assert compound[0] is region

    def test_creation_with_multiple_regions(self):
        """Test creating CompoundRegion with multiple regions."""
        region1 = LinearRegion(x_min=1.0, x_max=2.0)
        region2 = LinearRegion(x_min=4.0, x_max=5.0)
        region3 = LinearRegion(x_min=7.0, x_max=8.0)

        compound = CompoundRegion([region1, region2, region3])

        assert len(compound) == 3
        assert compound[0] is region1
        assert compound[1] is region2
        assert compound[2] is region3

    def test_creation_with_empty_list_raises(self):
        """Test that empty region list raises ValueError."""
        with pytest.raises(ValueError, match="at least one region"):
            CompoundRegion([])

    def test_creation_with_invalid_type_raises(self):
        """Test that non-LinearRegion objects raise TypeError."""
        region = LinearRegion(x_min=1.0, x_max=2.0)

        with pytest.raises(TypeError, match="must be LinearRegion"):
            CompoundRegion([region, "not a region"])

    def test_get_mask_disjoint_regions(self):
        """Test mask generation with disjoint (non-overlapping) regions."""
        region1 = LinearRegion(x_min=1.0, x_max=2.0)
        region2 = LinearRegion(x_min=4.0, x_max=5.0)
        compound = CompoundRegion([region1, region2])

        x_data = np.array([0, 1, 2, 3, 4, 5, 6])
        mask = compound.get_mask(x_data)
        expected = np.array([False, True, True, False, True, True, False])

        assert_array_equal(mask, expected)

    def test_get_mask_overlapping_regions(self):
        """Test mask generation with overlapping regions."""
        region1 = LinearRegion(x_min=1.0, x_max=4.0)
        region2 = LinearRegion(x_min=3.0, x_max=6.0)
        compound = CompoundRegion([region1, region2])

        x_data = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        mask = compound.get_mask(x_data)
        # Union of [1-4] and [3-6] = [1-6]
        expected = np.array([False, True, True, True, True, True, True, False])

        assert_array_equal(mask, expected)

    def test_get_mask_adjacent_regions(self):
        """Test mask generation with adjacent regions."""
        region1 = LinearRegion(x_min=1.0, x_max=3.0)
        region2 = LinearRegion(x_min=3.0, x_max=5.0)  # Shares boundary at 3.0
        compound = CompoundRegion([region1, region2])

        x_data = np.array([0, 1, 2, 3, 4, 5, 6])
        mask = compound.get_mask(x_data)
        # Both include 3.0, but OR operation handles it correctly
        expected = np.array([False, True, True, True, True, True, False])

        assert_array_equal(mask, expected)

    def test_get_mask_three_regions(self):
        """Test mask with three disjoint regions."""
        region1 = LinearRegion(x_min=1.0, x_max=2.0)
        region2 = LinearRegion(x_min=4.0, x_max=5.0)
        region3 = LinearRegion(x_min=7.0, x_max=8.0)
        compound = CompoundRegion([region1, region2, region3])

        x_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        mask = compound.get_mask(x_data)
        expected = np.array([False, True, True, False, True, True, False, True, True, False])

        assert_array_equal(mask, expected)

    def test_get_mask_no_data_in_any_region(self):
        """Test mask when no data falls within any region."""
        region1 = LinearRegion(x_min=10.0, x_max=20.0)
        region2 = LinearRegion(x_min=30.0, x_max=40.0)
        compound = CompoundRegion([region1, region2])

        x_data = np.array([0, 1, 2, 3, 4, 5])
        mask = compound.get_mask(x_data)
        expected = np.array([False, False, False, False, False, False])

        assert_array_equal(mask, expected)

    def test_indexing(self):
        """Test accessing individual regions by index."""
        region1 = LinearRegion(x_min=1.0, x_max=2.0)
        region2 = LinearRegion(x_min=4.0, x_max=5.0)
        compound = CompoundRegion([region1, region2])

        assert compound[0] is region1
        assert compound[1] is region2

        # Test negative indexing
        assert compound[-1] is region2
        assert compound[-2] is region1

    def test_indexing_out_of_range_raises(self):
        """Test that out-of-range indexing raises IndexError."""
        region = LinearRegion(x_min=1.0, x_max=2.0)
        compound = CompoundRegion([region])

        with pytest.raises(IndexError):
            _ = compound[1]  # Only one region (index 0)

    def test_len(self):
        """Test length of CompoundRegion."""
        region1 = LinearRegion(x_min=1.0, x_max=2.0)
        region2 = LinearRegion(x_min=4.0, x_max=5.0)
        region3 = LinearRegion(x_min=7.0, x_max=8.0)

        compound = CompoundRegion([region1, region2, region3])

        assert len(compound) == 3

    def test_repr(self):
        """Test string representation of CompoundRegion."""
        region1 = LinearRegion(x_min=1.0, x_max=2.0)
        region2 = LinearRegion(x_min=4.0, x_max=5.0)
        compound = CompoundRegion([region1, region2])

        repr_str = repr(compound)

        assert "CompoundRegion" in repr_str
        assert "2" in repr_str  # Number of regions


class TestROIIntegration:
    """Integration tests for ROI classes."""

    def test_extract_data_with_linear_region(self):
        """Test extracting data points within a LinearRegion."""
        region = LinearRegion(x_min=2.0, x_max=5.0)
        x_data = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        y_data = np.array([0, 10, 20, 30, 40, 50, 60, 70])

        mask = region.get_mask(x_data)
        x_in_region = x_data[mask]
        y_in_region = y_data[mask]

        assert_array_equal(x_in_region, np.array([2, 3, 4, 5]))
        assert_array_equal(y_in_region, np.array([20, 30, 40, 50]))

    def test_extract_data_with_compound_region(self):
        """Test extracting data from multiple regions."""
        region1 = LinearRegion(x_min=1.0, x_max=2.0)
        region2 = LinearRegion(x_min=5.0, x_max=6.0)
        compound = CompoundRegion([region1, region2])

        x_data = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        y_data = np.array([0, 10, 20, 30, 40, 50, 60, 70])

        mask = compound.get_mask(x_data)
        x_in_regions = x_data[mask]
        y_in_regions = y_data[mask]

        assert_array_equal(x_in_regions, np.array([1, 2, 5, 6]))
        assert_array_equal(y_in_regions, np.array([10, 20, 50, 60]))

    def test_region_with_unsorted_data(self):
        """Test that regions work correctly with unsorted data."""
        region = LinearRegion(x_min=2.0, x_max=5.0)
        x_data = np.array([7, 3, 1, 5, 0, 4, 2, 6])

        mask = region.get_mask(x_data)
        # Should select indices where x is in [2, 5]
        expected = np.array([False, True, False, True, False, True, True, False])

        assert_array_equal(mask, expected)
        assert_array_equal(x_data[mask], np.array([3, 5, 4, 2]))
