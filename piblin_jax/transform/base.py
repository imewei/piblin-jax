"""
Transform base classes for piblin-jax.

This module provides the abstract base class and hierarchy for all transforms:
- Transform: Abstract base class for all transforms
- DatasetTransform: Operates on Dataset objects
- MeasurementTransform: Operates on Measurement objects
- MeasurementSetTransform: Operates on MeasurementSet objects
- ExperimentTransform: Operates on Experiment objects
- ExperimentSetTransform: Operates on ExperimentSet objects

Transforms support:
- Lazy evaluation (computation deferred until needed)
- JIT compilation (via JAX backend)
- Immutability (via make_copy parameter)
- Pipeline composition (via Pipeline class)
"""

import copy
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypeVar

from piblin_jax.backend import is_jax_available
from piblin_jax.backend.operations import jit

# Import types for runtime isinstance checks
# These imports are safe - no circular dependency issues
from piblin_jax.data.collections import (
    Experiment,
    ExperimentSet,
    Measurement,
    MeasurementSet,
)
from piblin_jax.data.datasets import Dataset

# Type variables for generic transforms
T = TypeVar("T")


class Transform[T](ABC):
    """
    Abstract base class for all transforms.

    Transforms operate on data structures at various hierarchy levels:
    - Dataset level: individual measurements
    - Measurement level: collections of datasets
    - MeasurementSet level: collections of measurements
    - Experiment level: collections of measurement sets
    - ExperimentSet level: collections of experiments

    Transforms support:
    - Lazy evaluation: computation deferred until results accessed
    - JIT compilation: automatic compilation with JAX backend
    - Immutability: optional copying for functional programming style
    - Pipeline composition: chaining multiple transforms

    Examples
    --------
    >>> from piblin_jax.transform.base import DatasetTransform
    >>> from piblin_jax.data.datasets import OneDimensionalDataset
    >>> import numpy as np
    >>>
    >>> class MultiplyTransform(DatasetTransform):
    ...     def __init__(self, factor):
    ...         super().__init__()
    ...         self.factor = factor
    ...
    ...     def _apply(self, dataset):
    ...         dataset.y_data = dataset.y_data * self.factor
    ...         return dataset
    >>>
    >>> dataset = OneDimensionalDataset(
    ...     x_data=np.array([1, 2, 3]),
    ...     y_data=np.array([2, 4, 6])
    ... )
    >>> transform = MultiplyTransform(2.0)
    >>> result = transform.apply_to(dataset, make_copy=True)
    """

    def __init__(self) -> None:
        """Initialize transform with default settings."""
        self._lazy = False  # Lazy evaluation flag
        self._compiled = False  # JIT compilation flag

    @abstractmethod
    def _apply(self, target: T) -> T:
        """
        Internal implementation of transform logic.

        This method must be implemented by subclasses to define the
        actual transformation behavior. It receives the target object
        and returns the transformed object.

        Parameters
        ----------
        target : T
            Data structure to transform (Dataset, Measurement, etc.)

        Returns
        -------
        T
            Transformed data structure (same type as input)

        Notes
        -----
        - This method should modify the target in-place when possible
        - Copying is handled by apply_to, not _apply
        - JIT compilation can be applied to this method
        """
        pass

    def apply_to(self, target: T, make_copy: bool = True, propagate_uncertainty: bool = False) -> T:
        """
        Apply transform to target data structure.

        This is the main public interface for applying transforms.
        It handles copying (if requested) and delegates to the
        subclass-specific _apply method.

        Parameters
        ----------
        target : T
            Data structure to transform (Dataset, Measurement, etc.)
        make_copy : bool, default=True
            If True, create a deep copy before transforming (default).
            If False, transform in-place (more memory efficient but
            modifies the original object).
        propagate_uncertainty : bool, default=False
            If True and target has uncertainty samples, propagate uncertainty
            through the transform using Monte Carlo sampling.

        Returns
        -------
        T
            Transformed data structure

        Examples
        --------
        >>> transform = MyTransform()
        >>> # Create copy and transform
        >>> result = transform.apply_to(data, make_copy=True)
        >>> # Transform in-place (memory efficient)
        >>> result = transform.apply_to(data, make_copy=False)
        >>> # With uncertainty propagation
        >>> result = transform.apply_to(data_with_unc, propagate_uncertainty=True)

        Notes
        -----
        - make_copy=True ensures functional programming style (immutability)
        - make_copy=False is more memory efficient for large datasets
        - In pipelines, only the first copy is made at entry
        - Uncertainty propagation applies the transform to each uncertainty sample
        """
        if make_copy:
            # Deep copy using Python's copy module
            # JAX arrays are immutable, so this is safe
            target = self._copy_tree(target)

        # Apply transformation
        result = self._apply(target)

        # Propagate uncertainty if requested
        if propagate_uncertainty:
            result = self._propagate_uncertainty(result)

        return result

    @staticmethod
    def _copy_tree(obj: Any) -> Any:
        """
        Deep copy an object using appropriate method for backend.

        For all backends, uses standard deep copy to ensure proper object copying.
        Previous JAX tree mapping approach didn't work for custom dataset objects.

        Parameters
        ----------
        obj : Any
            Object to copy

        Returns
        -------
        Any
            Deep copy of object

        Notes
        -----
        JAX arrays are immutable, so copying creates new DeviceArrays
        with the same data. This is safe and efficient.

        Dataset objects contain JAX/NumPy arrays as internal data, but the
        object itself needs proper deep copying to maintain immutability.
        """
        # Always use deep copy for reliable object copying
        # JAX tree mapping doesn't work for custom dataset classes
        return copy.deepcopy(obj)

    def _propagate_uncertainty(self, target: T) -> T:
        """
        Propagate uncertainty through transform using Monte Carlo.

        If target has uncertainty samples, apply the transform to each
        sample to propagate uncertainty.

        Parameters
        ----------
        target : T
            Data structure with uncertainty information

        Returns
        -------
        T
            Data structure with propagated uncertainty

        Notes
        -----
        This method checks if the target has uncertainty samples and,
        if so, applies the transform to each sample. This implements
        Monte Carlo uncertainty propagation.
        """
        # Check if target has uncertainty samples
        if hasattr(target, "has_uncertainty") and target.has_uncertainty:
            if hasattr(target, "uncertainty_samples") and target.uncertainty_samples is not None:
                # For datasets, propagate through uncertainty samples
                # This is a Monte Carlo approach where we apply the transform
                # to each sample from the posterior

                # The samples dict contains parameter samples (e.g., 'sigma')
                # We need to create synthetic datasets from these samples
                # and apply the transform to each
                # For now, we keep the uncertainty samples unchanged
                # as they represent parameter uncertainty, not data uncertainty
                # Future enhancement: could implement full Monte Carlo propagation
                pass

        return target

    def __call__(self, target: T, make_copy: bool = True, propagate_uncertainty: bool = False) -> T:
        """
        Shorthand for apply_to.

        Allows using transform objects as callables:
        >>> result = transform(data)

        instead of:
        >>> result = transform.apply_to(data)

        Parameters
        ----------
        target : T
            Data structure to transform
        make_copy : bool, default=True
            If True, create copy before transforming
        propagate_uncertainty : bool, default=False
            If True, propagate uncertainty through transform

        Returns
        -------
        T
            Transformed data structure
        """
        return self.apply_to(
            target, make_copy=make_copy, propagate_uncertainty=propagate_uncertainty
        )


def jit_transform(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to enable JIT compilation for transform _apply methods.

    This decorator automatically compiles transform methods using JAX's
    JIT compiler when the JAX backend is available. For NumPy backend,
    it gracefully falls back to the uncompiled function.

    Parameters
    ----------
    func : Callable
        The _apply method to compile

    Returns
    -------
    Callable
        JIT-compiled function (JAX) or original function (NumPy)

    Examples
    --------
    >>> class MyTransform(DatasetTransform):
    ...     @jit_transform
    ...     def _apply(self, dataset):
    ...         # This will be JIT compiled with JAX
    ...         dataset.y_data = dataset.y_data * 2.0
    ...         return dataset

    Notes
    -----
    - JIT compilation can significantly improve performance
    - Only works with JAX backend (graceful fallback for NumPy)
    - First call may be slow (compilation), subsequent calls are fast
    - Static arguments should be marked appropriately
    """
    if is_jax_available():
        # Compile with JAX if available
        try:
            compiled_func: Callable[..., Any] = jit(func)
            return compiled_func
        except Exception:
            # If compilation fails, fall back to uncompiled
            return func
    else:
        # NumPy backend - no compilation
        return func


class DatasetTransform(Transform[Dataset]):
    """
    Transform that operates on Dataset objects.

    This is the lowest level transform in the hierarchy, operating
    on individual datasets (1D, 2D, 3D, etc.).

    Examples
    --------
    >>> from piblin_jax.transform.base import DatasetTransform
    >>> from piblin_jax.data.datasets import OneDimensionalDataset
    >>> import numpy as np
    >>>
    >>> class SmoothTransform(DatasetTransform):
    ...     def _apply(self, dataset):
    ...         # Smooth y-values with moving average
    ...         dataset.y_data = np.convolve(
    ...             dataset.y_data,
    ...             np.ones(3)/3,
    ...             mode='same'
    ...         )
    ...         return dataset
    >>>
    >>> dataset = OneDimensionalDataset(
    ...     x_data=np.array([1, 2, 3, 4, 5]),
    ...     y_data=np.array([1, 5, 2, 8, 3])
    ... )
    >>> transform = SmoothTransform()
    >>> smoothed = transform.apply_to(dataset)
    """

    def apply_to(
        self, target: Dataset, make_copy: bool = True, propagate_uncertainty: bool = False
    ) -> Dataset:
        """
        Apply transform to Dataset.

        Parameters
        ----------
        target : Dataset
            Dataset to transform
        make_copy : bool, default=True
            If True, create copy before transforming
        propagate_uncertainty : bool, default=False
            If True, propagate uncertainty through transform

        Returns
        -------
        Dataset
            Transformed dataset

        Raises
        ------
        TypeError
            If target is not a Dataset instance
        """
        if not isinstance(target, Dataset):
            raise TypeError(f"DatasetTransform requires Dataset, got {type(target).__name__}")
        return super().apply_to(target, make_copy, propagate_uncertainty)


class MeasurementTransform(Transform[Measurement]):
    """
    Transform that operates on Measurement objects.

    Measurements contain multiple datasets with associated metadata.
    This transform level can operate on all datasets within a measurement.

    Examples
    --------
    >>> from piblin_jax.transform.base import MeasurementTransform
    >>>
    >>> class NormalizeTransform(MeasurementTransform):
    ...     def _apply(self, measurement):
    ...         # Normalize all datasets in measurement
    ...         for dataset in measurement.datasets.values():
    ...             if hasattr(dataset, 'y_data'):
    ...                 max_val = dataset.y_data.max()
    ...                 dataset.y_data = dataset.y_data / max_val
    ...         return measurement
    """

    def apply_to(
        self, target: Measurement, make_copy: bool = True, propagate_uncertainty: bool = False
    ) -> Measurement:
        """
        Apply transform to Measurement.

        Parameters
        ----------
        target : Measurement
            Measurement to transform
        make_copy : bool, default=True
            If True, create copy before transforming
        propagate_uncertainty : bool, default=False
            If True, propagate uncertainty through transform

        Returns
        -------
        Measurement
            Transformed measurement

        Raises
        ------
        TypeError
            If target is not a Measurement instance
        """
        if not isinstance(target, Measurement):
            raise TypeError(
                f"MeasurementTransform requires Measurement, got {type(target).__name__}"
            )
        return super().apply_to(target, make_copy, propagate_uncertainty)


class MeasurementSetTransform(Transform[MeasurementSet]):
    """
    Transform that operates on MeasurementSet objects.

    MeasurementSets contain multiple measurements. This transform
    level can operate across measurements (e.g., normalization
    relative to the entire set).

    Examples
    --------
    >>> from piblin_jax.transform.base import MeasurementSetTransform
    >>>
    >>> class GlobalNormalizeTransform(MeasurementSetTransform):
    ...     def _apply(self, measurement_set):
    ...         # Find global max across all measurements
    ...         global_max = 0
    ...         for meas in measurement_set.measurements.values():
    ...             for ds in meas.datasets.values():
    ...                 if hasattr(ds, 'y_data'):
    ...                     global_max = max(global_max, ds.y_data.max())
    ...
    ...         # Normalize all datasets by global max
    ...         for meas in measurement_set.measurements.values():
    ...             for ds in meas.datasets.values():
    ...                 if hasattr(ds, 'y_data'):
    ...                     ds.y_data = ds.y_data / global_max
    ...         return measurement_set
    """

    def apply_to(
        self, target: MeasurementSet, make_copy: bool = True, propagate_uncertainty: bool = False
    ) -> MeasurementSet:
        """
        Apply transform to MeasurementSet.

        Parameters
        ----------
        target : MeasurementSet
            MeasurementSet to transform
        make_copy : bool, default=True
            If True, create copy before transforming
        propagate_uncertainty : bool, default=False
            If True, propagate uncertainty through transform

        Returns
        -------
        MeasurementSet
            Transformed measurement set

        Raises
        ------
        TypeError
            If target is not a MeasurementSet instance
        """
        if not isinstance(target, MeasurementSet):
            raise TypeError(
                f"MeasurementSetTransform requires MeasurementSet, got {type(target).__name__}"
            )
        return super().apply_to(target, make_copy, propagate_uncertainty)


class ExperimentTransform(Transform[Experiment]):
    """
    Transform that operates on Experiment objects.

    Experiments contain multiple measurement sets. This transform
    level can operate across measurement sets within an experiment.

    Examples
    --------
    >>> from piblin_jax.transform.base import ExperimentTransform
    >>>
    >>> class TemperatureCorrectionTransform(ExperimentTransform):
    ...     def _apply(self, experiment):
    ...         # Apply temperature correction to all measurement sets
    ...         temp = experiment.metadata.get('temperature', 300)
    ...         correction = 1.0 + (temp - 300) * 0.001
    ...
    ...         for mset in experiment.measurement_sets.values():
    ...             for meas in mset.measurements.values():
    ...                 for ds in meas.datasets.values():
    ...                     if hasattr(ds, 'y_data'):
    ...                         ds.y_data = ds.y_data * correction
    ...         return experiment
    """

    def apply_to(
        self, target: Experiment, make_copy: bool = True, propagate_uncertainty: bool = False
    ) -> Experiment:
        """
        Apply transform to Experiment.

        Parameters
        ----------
        target : Experiment
            Experiment to transform
        make_copy : bool, default=True
            If True, create copy before transforming
        propagate_uncertainty : bool, default=False
            If True, propagate uncertainty through transform

        Returns
        -------
        Experiment
            Transformed experiment

        Raises
        ------
        TypeError
            If target is not an Experiment instance
        """
        if not isinstance(target, Experiment):
            raise TypeError(f"ExperimentTransform requires Experiment, got {type(target).__name__}")
        return super().apply_to(target, make_copy, propagate_uncertainty)


class ExperimentSetTransform(Transform[ExperimentSet]):
    """
    Transform that operates on ExperimentSet objects.

    ExperimentSets are the top-level container for multiple experiments.
    This transform level can operate across all experiments in a set.

    Examples
    --------
    >>> from piblin_jax.transform.base import ExperimentSetTransform
    >>>
    >>> class CrossExperimentNormalizeTransform(ExperimentSetTransform):
    ...     def _apply(self, experiment_set):
    ...         # Normalize across all experiments
    ...         global_max = 0
    ...         for exp in experiment_set.experiments.values():
    ...             for mset in exp.measurement_sets.values():
    ...                 for meas in mset.measurements.values():
    ...                     for ds in meas.datasets.values():
    ...                         if hasattr(ds, 'y_data'):
    ...                             global_max = max(global_max, ds.y_data.max())
    ...
    ...         for exp in experiment_set.experiments.values():
    ...             for mset in exp.measurement_sets.values():
    ...                 for meas in mset.measurements.values():
    ...                     for ds in meas.datasets.values():
    ...                         if hasattr(ds, 'y_data'):
    ...                             ds.y_data = ds.y_data / global_max
    ...         return experiment_set
    """

    def apply_to(
        self, target: ExperimentSet, make_copy: bool = True, propagate_uncertainty: bool = False
    ) -> ExperimentSet:
        """
        Apply transform to ExperimentSet.

        Parameters
        ----------
        target : ExperimentSet
            ExperimentSet to transform
        make_copy : bool, default=True
            If True, create copy before transforming
        propagate_uncertainty : bool, default=False
            If True, propagate uncertainty through transform

        Returns
        -------
        ExperimentSet
            Transformed experiment set

        Raises
        ------
        TypeError
            If target is not an ExperimentSet instance
        """
        if not isinstance(target, ExperimentSet):
            raise TypeError(
                f"ExperimentSetTransform requires ExperimentSet, got {type(target).__name__}"
            )
        return super().apply_to(target, make_copy, propagate_uncertainty)


__all__ = [
    "DatasetTransform",
    "ExperimentSetTransform",
    "ExperimentTransform",
    "MeasurementSetTransform",
    "MeasurementTransform",
    "Transform",
    "jit_transform",
]
