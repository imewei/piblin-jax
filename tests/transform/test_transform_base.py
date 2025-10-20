"""
Tests for transform base classes and pipeline composition.

This module tests the foundation of the transform system including:
- Transform ABC interface
- Pipeline composition
- make_copy parameter behavior
- Transform application to different collection levels
- Lazy evaluation
"""

import numpy as np
import pytest

from quantiq.backend import jnp
from quantiq.data.collections import Experiment, ExperimentSet, Measurement, MeasurementSet
from quantiq.data.datasets import Dataset, OneDimensionalDataset
from quantiq.transform.base import (
    DatasetTransform,
    ExperimentSetTransform,
    ExperimentTransform,
    MeasurementSetTransform,
    MeasurementTransform,
    Transform,
)
from quantiq.transform.pipeline import Pipeline


class ConcreteDatasetTransform(DatasetTransform):
    """Concrete implementation for testing."""

    def __init__(self, multiplier: float = 2.0):
        super().__init__()
        self.multiplier = multiplier

    def _apply(self, target: Dataset) -> Dataset:
        """Multiply dataset values by a factor."""
        # Simple transform: multiply dependent variable values
        if hasattr(target, "_dependent_variable_data"):
            target._dependent_variable_data = target._dependent_variable_data * self.multiplier
        return target


class ConcreteMeasurementTransform(MeasurementTransform):
    """Concrete implementation for testing."""

    def __init__(self, offset: float = 1.0):
        super().__init__()
        self.offset = offset

    def _apply(self, target: Measurement) -> Measurement:
        """Add offset to all datasets in measurement."""
        for dataset in target.datasets.values():
            if hasattr(dataset, "_dependent_variable_data"):
                dataset._dependent_variable_data = dataset._dependent_variable_data + self.offset
        return target


class TestTransformABC:
    """Test Transform abstract base class interface."""

    def test_cannot_instantiate_abstract_transform(self):
        """Transform ABC cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Transform()

    def test_concrete_transform_can_be_instantiated(self):
        """Concrete transform implementations can be instantiated."""
        transform = ConcreteDatasetTransform()
        assert isinstance(transform, Transform)
        assert isinstance(transform, DatasetTransform)


class TestTransformApplication:
    """Test transform application to different hierarchy levels."""

    def test_dataset_transform_applies_correctly(self):
        """DatasetTransform correctly transforms Dataset objects."""
        # Create test dataset
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([2.0, 4.0, 6.0])
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        # Apply transform
        transform = ConcreteDatasetTransform(multiplier=3.0)
        result = transform.apply_to(dataset, make_copy=True)

        # Check result
        expected = y * 3.0
        np.testing.assert_allclose(result.dependent_variable_data, expected)

    def test_dataset_transform_validates_type(self):
        """DatasetTransform raises TypeError for wrong target type."""
        transform = ConcreteDatasetTransform()

        with pytest.raises(TypeError, match="DatasetTransform requires Dataset"):
            transform.apply_to("not a dataset")

    def test_measurement_transform_validates_type(self):
        """MeasurementTransform raises TypeError for wrong target type."""
        transform = ConcreteMeasurementTransform()

        with pytest.raises(TypeError, match="MeasurementTransform requires Measurement"):
            transform.apply_to("not a measurement")

    def test_transform_callable_shorthand(self):
        """Transform can be called directly as shorthand for apply_to."""
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([2.0, 4.0, 6.0])
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        transform = ConcreteDatasetTransform(multiplier=2.0)

        # Both should work
        result1 = transform.apply_to(dataset, make_copy=True)
        result2 = transform(dataset, make_copy=True)

        np.testing.assert_allclose(result1.dependent_variable_data, result2.dependent_variable_data)


class TestMakeCopyParameter:
    """Test make_copy parameter behavior."""

    def test_make_copy_true_creates_new_object(self):
        """make_copy=True creates a new object, leaving original unchanged."""
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([2.0, 4.0, 6.0])
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        original_y = dataset.dependent_variable_data.copy()

        transform = ConcreteDatasetTransform(multiplier=3.0)
        result = transform.apply_to(dataset, make_copy=True)

        # Original should be unchanged
        np.testing.assert_allclose(dataset.dependent_variable_data, original_y)

        # Result should be different
        np.testing.assert_allclose(result.dependent_variable_data, original_y * 3.0)

        # Should be different objects
        assert result is not dataset

    def test_make_copy_false_modifies_in_place(self):
        """make_copy=False modifies the original object."""
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([2.0, 4.0, 6.0])
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        original_y = y.copy()

        transform = ConcreteDatasetTransform(multiplier=3.0)
        result = transform.apply_to(dataset, make_copy=False)

        # Result should be the same object
        assert result is dataset

        # Values should be modified
        np.testing.assert_allclose(result.dependent_variable_data, original_y * 3.0)


class TestPipelineComposition:
    """Test Pipeline composition and sequential application."""

    def test_pipeline_creation(self):
        """Pipeline can be created with list of transforms."""
        t1 = ConcreteDatasetTransform(multiplier=2.0)
        t2 = ConcreteDatasetTransform(multiplier=3.0)

        pipeline = Pipeline([t1, t2])

        assert len(pipeline) == 2
        assert pipeline[0] is t1
        assert pipeline[1] is t2

    def test_pipeline_empty_creation(self):
        """Pipeline can be created empty."""
        pipeline = Pipeline()
        assert len(pipeline) == 0

    def test_pipeline_append(self):
        """Transforms can be appended to pipeline."""
        pipeline = Pipeline()
        t1 = ConcreteDatasetTransform(multiplier=2.0)

        pipeline.append(t1)

        assert len(pipeline) == 1
        assert pipeline[0] is t1

    def test_pipeline_sequential_application(self):
        """Pipeline applies transforms sequentially."""
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([2.0, 4.0, 6.0])
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        # Create pipeline: multiply by 2, then by 3 (net: 6x)
        t1 = ConcreteDatasetTransform(multiplier=2.0)
        t2 = ConcreteDatasetTransform(multiplier=3.0)
        pipeline = Pipeline([t1, t2])

        result = pipeline.apply_to(dataset, make_copy=True)

        # Should be multiplied by 6 total
        expected = y * 6.0
        np.testing.assert_allclose(result.dependent_variable_data, expected)

    def test_pipeline_single_copy_at_entry(self):
        """Pipeline makes only one copy at entry, then transforms in-place."""
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([2.0, 4.0, 6.0])
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        original_y = dataset.dependent_variable_data.copy()

        # Create pipeline
        t1 = ConcreteDatasetTransform(multiplier=2.0)
        t2 = ConcreteDatasetTransform(multiplier=3.0)
        pipeline = Pipeline([t1, t2])

        result = pipeline.apply_to(dataset, make_copy=True)

        # Original should be unchanged (copied at entry)
        np.testing.assert_allclose(dataset.dependent_variable_data, original_y)

        # Result should have both transforms applied
        np.testing.assert_allclose(result.dependent_variable_data, original_y * 6.0)

    def test_pipeline_mutable_sequence_interface(self):
        """Pipeline supports list-like operations."""
        t1 = ConcreteDatasetTransform(multiplier=2.0)
        t2 = ConcreteDatasetTransform(multiplier=3.0)
        t3 = ConcreteDatasetTransform(multiplier=4.0)

        pipeline = Pipeline([t1, t2])

        # Insert
        pipeline.insert(1, t3)
        assert len(pipeline) == 3
        assert pipeline[1] is t3

        # Delete
        del pipeline[1]
        assert len(pipeline) == 2

        # Set item
        t4 = ConcreteDatasetTransform(multiplier=5.0)
        pipeline[0] = t4
        assert pipeline[0] is t4

    def test_pipeline_only_accepts_transforms(self):
        """Pipeline validates that only Transform objects are added."""
        pipeline = Pipeline()

        with pytest.raises(TypeError, match="Pipeline can only contain Transform objects"):
            pipeline.append("not a transform")

        with pytest.raises(TypeError, match="Pipeline can only contain Transform objects"):
            pipeline.insert(0, 123)


class TestLazyEvaluation:
    """Test lazy evaluation and computation deferral."""

    def test_lazy_pipeline_defers_computation(self):
        """Lazy evaluation defers computation until results accessed."""
        # This is a placeholder test for lazy evaluation
        # Full implementation will be done in task 7.6

        # For now, test that pipeline can be created and stores transforms
        t1 = ConcreteDatasetTransform(multiplier=2.0)
        t2 = ConcreteDatasetTransform(multiplier=3.0)

        pipeline = Pipeline([t1, t2])

        # Pipeline should store transforms
        assert len(pipeline) == 2

        # When we implement lazy evaluation, this will verify
        # that computation is deferred until property access
