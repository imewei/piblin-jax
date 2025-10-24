"""
Pipeline composition for transforms.

This module provides pipeline functionality for composing multiple transforms:
- Pipeline: Sequential composition of transforms
- LazyPipeline: Pipeline with lazy evaluation support

Pipelines support:
- MutableSequence interface (list-like operations)
- Sequential transform application
- Single copy at entry (memory efficient)
- Lazy evaluation (computation deferred)
- JIT compilation (entire pipeline)
"""

from collections.abc import Iterable, MutableSequence
from typing import Any, TypeVar, overload

from .base import Transform

T = TypeVar("T")


class Pipeline(Transform[T], MutableSequence[Transform[T]]):
    """
    Pipeline for composing multiple transforms sequentially.

    A pipeline applies a sequence of transforms to data in order.
    It implements the MutableSequence interface, so it can be used
    like a list of transforms.

    The pipeline is memory-efficient: when make_copy=True, it creates
    a single copy at entry, then applies all transforms in-place.

    Parameters
    ----------
    transforms : list[Transform], optional
        Initial list of transforms to include in pipeline

    Examples
    --------
    >>> from piblin_jax.transform import Pipeline, DatasetTransform
    >>> from piblin_jax.data.datasets import OneDimensionalDataset
    >>> import numpy as np
    >>>
    >>> # Create transforms
    >>> class MultiplyTransform(DatasetTransform):
    ...     def __init__(self, factor):
    ...         super().__init__()
    ...         self.factor = factor
    ...
    ...     def _apply(self, dataset):
    ...         dataset.y_data = dataset.y_data * self.factor
    ...         return dataset
    >>>
    >>> # Create pipeline
    >>> pipeline = Pipeline([
    ...     MultiplyTransform(2.0),
    ...     MultiplyTransform(3.0),  # Net effect: 6x
    ... ])
    >>>
    >>> # Apply to dataset
    >>> dataset = OneDimensionalDataset(
    ...     x_data=np.array([1, 2, 3]),
    ...     y_data=np.array([2, 4, 6])
    ... )
    >>> result = pipeline.apply_to(dataset, make_copy=True)
    >>> # result.y_data is now [12, 24, 36]

    Notes
    -----
    - Pipelines can be nested: a pipeline can contain other pipelines
    - Only one copy is made at entry, then all transforms apply in-place
    - This is much more memory efficient than copying at each step
    - Use lazy evaluation for even better performance with JAX
    """

    def __init__(self, transforms: list[Transform[T]] | None = None):
        """
        Initialize pipeline.

        Parameters
        ----------
        transforms : list[Transform], optional
            Initial transforms to include in pipeline
        """
        super().__init__()
        self._transforms: list[Transform[T]] = list(transforms) if transforms else []
        self._lazy = False  # Standard pipeline is eager by default

    def _apply(self, target: T, propagate_uncertainty: bool = False) -> T:
        """
        Apply all transforms in sequence.

        This is the internal implementation that applies each transform
        in the pipeline sequentially. Each transform operates in-place
        on the result of the previous transform.

        Parameters
        ----------
        target : T
            Data structure to transform
        propagate_uncertainty : bool, default=False
            If True, propagate uncertainty through all transforms

        Returns
        -------
        T
            Transformed data structure after all transforms applied

        Notes
        -----
        All transforms are applied with make_copy=False for efficiency,
        since the Pipeline.apply_to method handles copying at entry.
        """
        result = target
        for transform in self._transforms:
            # Apply each transform in-place (no copy)
            result = transform.apply_to(
                result, make_copy=False, propagate_uncertainty=propagate_uncertainty
            )
        return result

    def apply_to(self, target: T, make_copy: bool = True, propagate_uncertainty: bool = False) -> T:
        """
        Apply pipeline to target.

        Only makes copy once at entry, then applies all transforms
        in-place for memory efficiency.

        Parameters
        ----------
        target : T
            Data structure to transform
        make_copy : bool, default=True
            If True, create one copy at entry before applying transforms
        propagate_uncertainty : bool, default=False
            If True and target has uncertainty, propagate through all transforms

        Returns
        -------
        T
            Transformed data structure

        Notes
        -----
        This is much more efficient than copying at each transform step.
        The single copy at entry ensures immutability while minimizing
        memory overhead.

        When propagate_uncertainty=True, uncertainty is efficiently propagated
        through the entire pipeline in a single pass.
        """
        if make_copy:
            # Single copy at entry
            target = self._copy_tree(target)

        # Apply all transforms in-place
        return self._apply(target, propagate_uncertainty=propagate_uncertainty)

    # MutableSequence interface implementation
    # This allows Pipeline to be used like a list

    @overload
    def __getitem__(self, index: int) -> Transform[T]: ...

    @overload
    def __getitem__(self, index: slice) -> list[Transform[T]]: ...

    def __getitem__(self, index: int | slice) -> Transform[T] | list[Transform[T]]:
        """
        Get transform(s) at index.

        Parameters
        ----------
        index : int or slice
            Index or slice to retrieve

        Returns
        -------
        Transform or list[Transform]
            Transform at index, or list of transforms for slice

        Examples
        --------
        >>> pipeline = Pipeline([t1, t2, t3])
        >>> pipeline[0]  # Get first transform
        >>> pipeline[1:3]  # Get slice of transforms
        """
        if isinstance(index, slice):
            # For slices, return list of transforms
            return self._transforms[index]
        # For single index, return single transform
        return self._transforms[index]

    @overload
    def __setitem__(self, index: int, value: Transform[T]) -> None: ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[Transform[T]]) -> None: ...

    def __setitem__(self, index: int | slice, value: Transform[T] | Iterable[Transform[T]]) -> None:
        """
        Set transform(s) at index.

        Parameters
        ----------
        index : int or slice
            Index or slice to set
        value : Transform or list[Transform]
            Transform(s) to set at index

        Raises
        ------
        TypeError
            If value is not a Transform instance

        Examples
        --------
        >>> pipeline = Pipeline([t1, t2, t3])
        >>> pipeline[0] = new_transform  # Replace first transform
        """
        if isinstance(index, slice):
            # For slices, validate all values are transforms
            value_list = list(value) if not isinstance(value, Transform) else [value]
            if not all(isinstance(v, Transform) for v in value_list):
                raise TypeError("Pipeline can only contain Transform objects")
            self._transforms[index] = value_list
        else:
            # For single index, validate value is a transform
            if not isinstance(value, Transform):
                raise TypeError("Pipeline can only contain Transform objects")
            self._transforms[index] = value

    def __delitem__(self, index: int | slice) -> None:
        """
        Delete transform(s) at index.

        Parameters
        ----------
        index : int or slice
            Index or slice to delete

        Examples
        --------
        >>> pipeline = Pipeline([t1, t2, t3])
        >>> del pipeline[0]  # Remove first transform
        >>> del pipeline[1:]  # Remove all but first transform
        """
        del self._transforms[index]

    def __len__(self) -> int:
        """
        Get number of transforms in pipeline.

        Returns
        -------
        int
            Number of transforms

        Examples
        --------
        >>> pipeline = Pipeline([t1, t2, t3])
        >>> len(pipeline)
        3
        """
        return len(self._transforms)

    def insert(self, index: int, value: Transform[T]) -> None:
        """
        Insert transform at index.

        Parameters
        ----------
        index : int
            Index at which to insert transform
        value : Transform
            Transform to insert

        Raises
        ------
        TypeError
            If value is not a Transform instance

        Examples
        --------
        >>> pipeline = Pipeline([t1, t3])
        >>> pipeline.insert(1, t2)  # Insert t2 between t1 and t3
        """
        if not isinstance(value, Transform):
            raise TypeError("Pipeline can only contain Transform objects")
        self._transforms.insert(index, value)

    def append(self, transform: Transform[T]) -> None:
        """
        Add transform to end of pipeline.

        Parameters
        ----------
        transform : Transform
            Transform to append

        Raises
        ------
        TypeError
            If transform is not a Transform instance

        Examples
        --------
        >>> pipeline = Pipeline([t1, t2])
        >>> pipeline.append(t3)  # Add t3 to end
        """
        if not isinstance(transform, Transform):
            raise TypeError("Pipeline can only contain Transform objects")
        self._transforms.append(transform)

    def __repr__(self) -> str:
        """
        String representation of pipeline.

        Returns
        -------
        str
            String representation showing number of transforms

        Examples
        --------
        >>> pipeline = Pipeline([t1, t2, t3])
        >>> repr(pipeline)
        'Pipeline(3 transforms)'
        """
        return f"Pipeline({len(self._transforms)} transforms)"

    def __str__(self) -> str:
        """
        Human-readable string representation.

        Returns
        -------
        str
            String showing all transforms in pipeline
        """
        if not self._transforms:
            return "Pipeline(empty)"

        lines = ["Pipeline:"]
        for i, transform in enumerate(self._transforms):
            lines.append(f"  {i}. {transform.__class__.__name__}")
        return "\n".join(lines)


class LazyPipeline(Pipeline[T]):
    """
    Pipeline with lazy evaluation support.

    Unlike the standard Pipeline, LazyPipeline defers computation
    until the results are actually accessed. This allows JAX to
    optimize the entire computation graph as a single operation.

    Lazy evaluation is triggered on:
    - Property access (e.g., result.y_data)
    - Method calls (e.g., result.visualize())
    - Export operations (e.g., result.export())

    Parameters
    ----------
    transforms : list[Transform], optional
        Initial list of transforms to include in pipeline

    Examples
    --------
    >>> from piblin_jax.transform import LazyPipeline
    >>>
    >>> # Create lazy pipeline
    >>> pipeline = LazyPipeline([
    ...     MultiplyTransform(2.0),
    ...     MultiplyTransform(3.0),
    ... ])
    >>>
    >>> # Apply to dataset (computation deferred)
    >>> lazy_result = pipeline.apply_to(dataset, make_copy=True)
    >>>
    >>> # Access property (triggers computation)
    >>> y_values = lazy_result.y_data  # Computation happens here

    Notes
    -----
    - Lazy evaluation allows JAX to optimize the entire pipeline
    - First property access triggers computation and caches result
    - Subsequent accesses use cached result
    - More efficient than eager evaluation for complex pipelines
    """

    def __init__(self, transforms: list[Transform[T]] | None = None):
        """
        Initialize lazy pipeline.

        Parameters
        ----------
        transforms : list[Transform], optional
            Initial transforms to include in pipeline
        """
        super().__init__(transforms)
        self._lazy = True  # Lazy pipelines defer computation
        self._target: T | None = None
        self._result_cache: T | None = None
        self._dirty = True  # Flag indicating computation needed
        self._propagate_unc = False  # Store uncertainty propagation flag

    def apply_to(
        self, target: T, make_copy: bool = True, propagate_uncertainty: bool = False
    ) -> Any:
        """
        Apply lazy pipeline to target.

        Computation is deferred until results are accessed.
        Returns a LazyResult wrapper that triggers computation
        on property/method access.

        Parameters
        ----------
        target : T
            Data structure to transform
        make_copy : bool, default=True
            If True, create copy before transforming
        propagate_uncertainty : bool, default=False
            If True, propagate uncertainty through all transforms

        Returns
        -------
        LazyResult
            Wrapper that triggers computation on access

        Notes
        -----
        The actual transformation is not performed until the
        result is accessed. This allows JAX to optimize the
        entire computation graph.
        """
        if make_copy:
            target = self._copy_tree(target)

        # Store target and mark as dirty (needs computation)
        self._target = target
        self._dirty = True
        self._propagate_unc = propagate_uncertainty

        # Return lazy wrapper that triggers computation on access
        return LazyResult(self)

    def _compute(self) -> T | None:
        """
        Execute the pipeline computation.

        This is called when results are accessed. It performs the
        actual computation and caches the result.

        Returns
        -------
        T | None
            Computed result

        Notes
        -----
        Result is cached so subsequent accesses don't recompute.
        """
        if self._dirty and self._target is not None:
            # Perform computation with uncertainty propagation if requested
            self._result_cache = self._apply(
                self._target, propagate_uncertainty=self._propagate_unc
            )
            self._dirty = False

        return self._result_cache

    def invalidate_cache(self) -> None:
        """
        Invalidate cached results.

        Forces recomputation on next access. Useful if transforms
        have been modified or parameters changed.

        Examples
        --------
        >>> pipeline = LazyPipeline([transform1, transform2])
        >>> result = pipeline.apply_to(dataset)
        >>> _ = result.y_data  # Triggers computation
        >>>
        >>> # Modify pipeline
        >>> pipeline.append(transform3)
        >>> pipeline.invalidate_cache()  # Force recomputation
        """
        self._dirty = True
        self._result_cache = None


class LazyResult:
    """
    Wrapper that triggers lazy computation on property access.

    This class wraps the actual result and defers computation
    until properties or methods are accessed.

    Parameters
    ----------
    pipeline : LazyPipeline
        The lazy pipeline that will compute the result

    Examples
    --------
    >>> lazy_result = LazyResult(pipeline)
    >>> # No computation yet
    >>> y = lazy_result.y_data  # Triggers computation here
    >>> # Subsequent accesses use cached result
    >>> x = lazy_result.x_data  # No recomputation

    Notes
    -----
    This class is transparent to the user - it behaves like
    the actual result object, but triggers computation on
    first access.
    """

    def __init__(self, pipeline: LazyPipeline[Any]):
        """
        Initialize lazy result wrapper.

        Parameters
        ----------
        pipeline : LazyPipeline
            Pipeline that will compute the result
        """
        # Store in __dict__ to avoid triggering __getattr__
        object.__setattr__(self, "_pipeline", pipeline)
        object.__setattr__(self, "_computed", None)

    def __getattr__(self, name: str) -> Any:
        """
        Get attribute from computed result.

        Triggers computation on first access.

        Parameters
        ----------
        name : str
            Attribute name

        Returns
        -------
        Any
            Attribute value from computed result
        """
        # Trigger computation if not already done
        if object.__getattribute__(self, "_computed") is None:
            pipeline = object.__getattribute__(self, "_pipeline")
            computed = pipeline._compute()
            object.__setattr__(self, "_computed", computed)

        # Get attribute from computed result
        computed = object.__getattribute__(self, "_computed")
        return getattr(computed, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Set attribute on computed result.

        Triggers computation if not already done.

        Parameters
        ----------
        name : str
            Attribute name
        value : Any
            Attribute value
        """
        if name in ("_pipeline", "_computed"):
            # Internal attributes
            object.__setattr__(self, name, value)
        else:
            # Trigger computation if needed
            if object.__getattribute__(self, "_computed") is None:
                pipeline = object.__getattribute__(self, "_pipeline")
                computed = pipeline._compute()
                object.__setattr__(self, "_computed", computed)

            # Set attribute on computed result
            computed = object.__getattribute__(self, "_computed")
            setattr(computed, name, value)

    def __repr__(self) -> str:
        """String representation."""
        return f"LazyResult(computed={object.__getattribute__(self, '_computed') is not None})"


__all__ = [
    "LazyPipeline",
    "LazyResult",
    "Pipeline",
]
