"""Tests for metadata utilities.

Test the metadata system for managing, validating, extracting, and merging
metadata (conditions and details) across the data hierarchy.
"""

import pytest
from quantiq.data import metadata


class TestMetadataMerging:
    """Test metadata merging with different strategies."""

    def test_merge_override_strategy(self):
        """Test that later values override earlier ones (default)."""
        meta1 = {"temp": 20, "sample": "A1"}
        meta2 = {"temp": 25, "pressure": 1.0}
        meta3 = {"temp": 30}

        result = metadata.merge_metadata([meta1, meta2, meta3])

        assert result["temp"] == 30  # Last value wins
        assert result["sample"] == "A1"
        assert result["pressure"] == 1.0

    def test_merge_keep_first_strategy(self):
        """Test that first value is kept on conflicts."""
        meta1 = {"temp": 20, "sample": "A1"}
        meta2 = {"temp": 25, "pressure": 1.0}

        result = metadata.merge_metadata([meta1, meta2], strategy="keep_first")

        assert result["temp"] == 20  # First value kept
        assert result["sample"] == "A1"
        assert result["pressure"] == 1.0

    def test_merge_raise_strategy(self):
        """Test that conflicts raise errors with 'raise' strategy."""
        meta1 = {"temp": 20, "sample": "A1"}
        meta2 = {"temp": 20, "sample": "A1"}  # Same values OK

        # Same values should not raise
        result = metadata.merge_metadata([meta1, meta2], strategy="raise")
        assert result["temp"] == 20

        # Different values should raise
        meta3 = {"temp": 30}
        with pytest.raises(ValueError, match="Metadata conflict"):
            metadata.merge_metadata([meta1, meta3], strategy="raise")

    def test_merge_list_strategy(self):
        """Test that conflicts are collected in lists."""
        meta1 = {"temp": 20, "sample": "A1"}
        meta2 = {"temp": 25, "pressure": 1.0}
        meta3 = {"temp": 20}  # Duplicate value

        result = metadata.merge_metadata([meta1, meta2, meta3], strategy="list")

        assert result["temp"] == [20, 25]  # Unique values collected
        assert result["sample"] == "A1"
        assert result["pressure"] == 1.0

    def test_merge_empty_list(self):
        """Test merging empty metadata list."""
        result = metadata.merge_metadata([])
        assert result == {}


class TestConditionsDetailsSeparation:
    """Test separation of conditions from details."""

    def test_separate_with_explicit_keys(self):
        """Test separation with explicitly provided condition keys."""
        combined = {
            "temperature": 25,
            "pressure": 1.0,
            "operator": "John",
            "date": "2025-01-15"
        }

        conditions, details = metadata.separate_conditions_details(
            combined,
            condition_keys=["temperature", "pressure"]
        )

        assert conditions == {"temperature": 25, "pressure": 1.0}
        assert details == {"operator": "John", "date": "2025-01-15"}

    def test_separate_with_heuristics(self):
        """Test automatic separation using heuristics."""
        combined = {
            "temp": 25,
            "pressure": 1.0,
            "strain": 0.1,
            "operator": "John",
            "notes": "First trial"
        }

        conditions, details = metadata.separate_conditions_details(combined)

        # Heuristics should identify temp, pressure, strain as conditions
        assert "temp" in conditions
        assert "pressure" in conditions
        assert "strain" in conditions
        assert "operator" in details
        assert "notes" in details


class TestMetadataValidation:
    """Test metadata validation."""

    def test_validate_types(self):
        """Test type checking validation."""
        meta = {"temp": 25.0, "sample": "A1", "count": 10}
        schema = {"temp": float, "sample": str, "count": int}

        # Should pass
        assert metadata.validate_metadata(meta, schema=schema)

    def test_validate_type_mismatch(self):
        """Test validation fails on type mismatch."""
        meta = {"temp": "25", "sample": "A1"}  # temp should be float
        schema = {"temp": float, "sample": str}

        with pytest.raises(ValueError, match="incorrect type"):
            metadata.validate_metadata(meta, schema=schema)

    def test_validate_required_keys(self):
        """Test required keys enforcement."""
        meta = {"temp": 25.0, "sample": "A1"}

        # Should pass
        assert metadata.validate_metadata(
            meta,
            required_keys=["temp", "sample"]
        )

        # Should fail
        with pytest.raises(ValueError, match="Missing required"):
            metadata.validate_metadata(
                meta,
                required_keys=["temp", "sample", "pressure"]
            )

    def test_validate_custom_function(self):
        """Test validation with custom function."""
        meta = {"temp": 25.0, "ph": 7.2}
        schema = {
            "temp": float,
            "ph": lambda x: 0 <= x <= 14  # pH range validator
        }

        # Valid pH
        assert metadata.validate_metadata(meta, schema=schema)

        # Invalid pH
        meta_invalid = {"temp": 25.0, "ph": 15.0}
        with pytest.raises(ValueError, match="failed validation"):
            metadata.validate_metadata(meta_invalid, schema=schema)


class TestMetadataExtraction:
    """Test metadata extraction from various sources."""

    def test_parse_key_value_string(self):
        """Test parsing key-value pairs from strings."""
        text = "temp=25,pressure=1.0,sample=A1"
        result = metadata.parse_key_value_string(text)

        assert result == {"temp": "25", "pressure": "1.0", "sample": "A1"}

    def test_parse_key_value_custom_separators(self):
        """Test parsing with custom separators."""
        text = "temp:25;pressure:1.0;sample:A1"
        result = metadata.parse_key_value_string(
            text,
            separator=":",
            delimiter=";"
        )

        assert result == {"temp": "25", "pressure": "1.0", "sample": "A1"}

    def test_extract_from_filename_heuristic(self):
        """Test filename extraction with heuristics."""
        filename = "sample_A1_temp_25C_001.csv"
        result = metadata.extract_from_filename(filename)

        assert result["sample"] == "A1"
        assert result["temp"] == "25"
        assert result["replicate"] == "001"

    def test_extract_from_filename_pattern(self):
        """Test filename extraction with regex pattern."""
        filename = "data_A1_25C_1atm.csv"
        pattern = r"data_(?P<sample>\w+)_(?P<temp>\d+)C_(?P<pressure>\d+)atm"
        result = metadata.extract_from_filename(filename, pattern=pattern)

        assert result == {"sample": "A1", "temp": "25", "pressure": "1"}

    def test_extract_from_path(self):
        """Test extraction from directory structure."""
        filepath = "/data/ProjectA/ExpB/SampleC/data.csv"
        result = metadata.extract_from_path(
            filepath,
            level_names=["sample", "experiment", "project"]
        )

        assert result == {
            "sample": "SampleC",
            "experiment": "ExpB",
            "project": "ProjectA"
        }

    def test_parse_header_metadata(self):
        """Test parsing metadata from file headers."""
        header_lines = [
            "# Temperature: 25",
            "# Pressure: 1.0",
            "# Sample: A1",
            "# Operator: John Doe"
        ]
        result = metadata.parse_header_metadata(header_lines)

        assert result == {
            "Temperature": "25",
            "Pressure": "1.0",
            "Sample": "A1",
            "Operator": "John Doe"
        }


class TestMetadataIntegration:
    """Test integrated metadata workflows."""

    def test_full_workflow(self):
        """Test complete metadata workflow: extract, merge, separate, validate."""
        # Extract from multiple sources
        file_meta = metadata.extract_from_filename("sample_A1_temp_25C_001.csv")
        path_meta = {"project": "ProjectX", "experiment": "ExpY"}
        header_meta = {"operator": "John", "date": "2025-01-15"}

        # Merge
        combined = metadata.merge_metadata([path_meta, file_meta, header_meta])

        # Should have all metadata
        assert "sample" in combined
        assert "temp" in combined
        assert "project" in combined
        assert "operator" in combined

        # Separate into conditions and details
        conditions, details = metadata.separate_conditions_details(
            combined,
            condition_keys=["temp"]
        )

        assert "temp" in conditions
        assert "operator" in details
        assert "project" in details

        # Validate conditions
        # Convert temp to float for validation
        conditions_typed = {k: float(v) if k == "temp" else v
                          for k, v in conditions.items()}
        schema = {"temp": float}
        assert metadata.validate_metadata(conditions_typed, schema=schema)
