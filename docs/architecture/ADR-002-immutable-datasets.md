# ADR-002: Immutable Datasets

## Status

**Accepted** - Implemented throughout `quantiq.data`

Date: 2024-10-19

## Context

quantiq processes scientific measurement data through transformation pipelines. We need to decide how datasets behave when transformed:

1. **In-place Modification** (mutable): `dataset.smooth(sigma=2.0)` modifies `dataset` directly
2. **Copy-on-Transform** (immutable): `dataset.smooth(sigma=2.0)` returns a new dataset, leaving original unchanged

This decision affects:
- JAX compatibility (JAX arrays are immutable)
- Thread safety (shared data across transforms)
- Debugging (can we inspect intermediate states?)
- Memory usage (copying large datasets)
- API ergonomics (explicit vs implicit behavior)

**Background**: piblin used mutable datasets, which caused subtle bugs when the same dataset was used in multiple pipelines or when debugging intermediate transformation steps.

## Decision

We make **all datasets immutable** and use the **copy-on-transform** pattern:

```python
# All dataset classes are frozen dataclasses
@dataclass(frozen=True)
class OneDimensionalDataset:
    independent_variable_data: np.ndarray
    dependent_variable_data: np.ndarray
    conditions: dict[str, Any]
    details: dict[str, Any]
```

**Transform API**:

```python
# Transforms ALWAYS create new datasets
def apply_to(dataset: Dataset, make_copy: bool = True) -> Dataset:
    """Apply transform, returning new dataset."""
    new_data = transform_function(dataset.data)
    return dataset.copy_with(data=new_data)  # New object
```

**User Pattern**:

```python
# Original dataset unchanged
original = OneDimensionalDataset(x=x, y=y)

# Transform creates new dataset
smoothed = smooth_transform.apply_to(original)

# Original still available for inspection
assert original.y[0] != smoothed.y[0]  # Different data
```

## Consequences

### Positive

1. **JAX Compatibility**:
   - JAX transformations (jit, vmap, grad) require immutability
   - No runtime errors from mutating arrays inside jit
   - Enables automatic differentiation through pipelines

2. **Predictable Behavior**:
   - Transformations don't have side effects
   - Can reuse datasets without fear of modification
   - Easy to reason about: `f(x)` never changes `x`

3. **Thread Safety**:
   - Multiple threads can safely share datasets
   - No locks or synchronization needed
   - Parallel processing "just works"

4. **Debugging Friendly**:
   - Inspect any intermediate transformation stage
   - Compare before/after states easily
   - Trace data provenance through pipeline

5. **Undo/Redo Support**:
   - Keep transformation history
   - Easy rollback to previous states
   - Time-travel debugging

### Negative

1. **Memory Overhead**:
   - Each transform creates a copy
   - Large datasets (>1GB) consume more memory
   - Pipelines with N steps = N dataset copies
   - (Mitigated: JAX's structural sharing reduces actual copying)

2. **API Verbosity**:
   - Must explicitly assign results: `data = transform.apply_to(data)`
   - Can't chain methods: `data.smooth().normalize()` not possible
   - More typing compared to in-place operations

3. **Learning Curve**:
   - Users from mutable backgrounds (NumPy, pandas) may be surprised
   - "Why can't I just modify the dataset?"
   - Need documentation explaining the pattern

4. **Performance Perception**:
   - Users may worry about "unnecessary copying"
   - Need to educate about JAX's structural sharing
   - Some overhead exists, but usually negligible

### Mitigation Strategies

**For Memory Concerns**:

```python
# Pipeline minimizes copies (single copy at entry point)
pipeline = Pipeline([Smooth(), Normalize(), Derivative()])
result = pipeline.apply_to(data, make_copy=True)  # Only 1 copy
# Internal transforms use make_copy=False
```

**For API Ergonomics**:

```python
# Provide Pipeline for chaining
pipe = Pipeline([
    GaussianSmooth(sigma=2.0),
    Normalize(method="min-max"),
    Derivative(order=1)
])
result = pipe.apply_to(data)  # Clean syntax
```

**For Documentation**:

- Examples emphasize the immutability pattern
- CONTRIBUTING.md explains the rationale
- Docstrings always show `result = transform.apply_to(data)`

## Alternatives Considered

### Alternative 1: Mutable Datasets (piblin Style)

```python
# In-place modification
dataset.smooth(sigma=2.0)  # Modifies dataset directly
dataset.normalize()         # Also modifies
```

**Pros**:
- Less memory usage (no copies)
- Simpler mental model for some users
- Less typing

**Cons**:
- JAX incompatibility (JAX arrays are immutable)
- Thread-unsafe without locks
- Surprising behavior when sharing datasets
- Difficult debugging (intermediate states lost)
- Historical bugs in piblin from this pattern

**Why Rejected**: Incompatible with JAX, too error-prone

### Alternative 2: Optional Mutability

```python
# User chooses behavior
transform.apply_to(data, inplace=True)   # Mutable
transform.apply_to(data, inplace=False)  # Immutable
```

**Pros**:
- Flexibility for users
- Performance tuning for large datasets
- Familiar from pandas/NumPy

**Cons**:
- Two code paths to maintain and test
- JAX still can't handle inplace=True
- Users have to think about the choice
- More complex API

**Why Rejected**: Complexity not worth the benefit

### Alternative 3: Copy-on-Write (CoW)

```python
# Lazy copying: only copy when modified
dataset.smooth(sigma=2.0)  # Doesn't copy until needed
```

**Pros**:
- Best of both worlds (seeming mutability + immutability)
- Memory efficient
- Used by polars, modern pandas

**Cons**:
- Complex implementation (reference counting)
- Hidden performance characteristics
- Not compatible with JAX's memory model
- Difficult to debug (when do copies happen?)

**Why Rejected**: Too complex, JAX incompatible

### Alternative 4: Explicit Copy Method

```python
# User must explicitly copy
new_dataset = dataset.copy()
new_dataset.smooth(sigma=2.0)  # Modifies the copy
```

**Pros**:
- User controls when copying happens
- Explicit is better than implicit

**Cons**:
- Easy to forget `.copy()` and get bugs
- JAX still can't mutate arrays
- Doesn't solve the core problem

**Why Rejected**: Doesn't enable JAX transformations

## Implementation Notes

### For Dataset Developers

```python
from dataclasses import dataclass

@dataclass(frozen=True)  # ✅ ALWAYS frozen
class MyDataset:
    data: np.ndarray
    metadata: dict[str, Any]

    def copy_with(self, **changes):
        """Create modified copy."""
        return dataclasses.replace(self, **changes)
```

### For Transform Developers

```python
class MyTransform(DatasetTransform):
    def apply_to(self, dataset: Dataset, make_copy: bool = True) -> Dataset:
        # Process data
        new_data = self._transform(dataset.data)

        # Return new dataset
        return dataset.copy_with(
            dependent_variable_data=new_data
        )
```

### For Users

```python
# ✅ CORRECT: Assign result
smoothed = smooth.apply_to(original)

# ❌ WRONG: Expecting in-place modification
smooth.apply_to(original)  # Returns new dataset, original unchanged!
```

## Related Decisions

- **ADR-001**: Backend Abstraction (JAX requires immutability)
- **ADR-003**: Transform Hierarchy (pipelines build on immutability)

## References

- JAX Philosophy: https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html
- Functional Programming Principles
- Python `dataclasses` with `frozen=True`: https://docs.python.org/3/library/dataclasses.html
- polars' Copy-on-Write: https://pola-rs.github.io/polars/user-guide/concepts/streaming/

## Discussion History

- piblin Issue #45: "Unexpected dataset modification in pipeline"
- quantiq Design Doc: "Immutability for JAX Compatibility"
- Team discussion 2024-09: Unanimous agreement on immutability

## Revision History

- 2024-10-19: Initial ADR creation
- Status: Accepted and implemented
