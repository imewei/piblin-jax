# ADR-004: Two-Dictionary Metadata System (Conditions vs Details)

## Status

**Accepted** - Implemented in all `piblin_jax.data` structures

Date: 2024-10-19

## Context

Scientific measurements require rich metadata to be useful:

- **Experimental parameters**: temperature, pressure, sample ID, replicate number
- **Contextual information**: operator name, instrument ID, timestamp, notes
- **Processing history**: applied transforms, calibration factors, quality flags

**Problem**: How should we organize and manage this metadata?

**Key Questions**:
1. Should all metadata be in one dictionary or separated?
2. Which metadata should propagate through transforms?
3. How do we identify "varying" vs "constant" conditions in datasets?
4. What happens when merging datasets with conflicting metadata?

**piblin's Approach**: Single `metadata` dict, no distinction between types.

**Pain Points from piblin**:
- Hard to identify experimental conditions vs administrative info
- No clear rules for metadata propagation
- Merging datasets required manual conflict resolution
- Grouping by conditions was error-prone

## Decision

We implement a **two-dictionary system** separating experimental conditions from administrative details:

```python
@dataclass(frozen=True)
class OneDimensionalDataset:
    independent_variable_data: np.ndarray
    dependent_variable_data: np.ndarray

    # SEPARATED METADATA
    conditions: dict[str, Any]  # Experimental conditions
    details: dict[str, Any]     # Administrative/contextual details
```

**Semantic Distinction**:

| Category | Purpose | Examples | Propagation | Grouping |
|----------|---------|----------|-------------|----------|
| `conditions` | Experimental parameters that affect the data | `temperature`, `sample_id`, `concentration`, `replicate` | ✅ Always | ✅ Used for groupby |
| `details` | Administrative/contextual information | `operator`, `timestamp`, `instrument`, `notes` | ⚠️ Maybe | ❌ Not for groupby |

**Usage Pattern**:

```python
# Creating a dataset
dataset = OneDimensionalDataset(
    independent_variable_data=x,
    dependent_variable_data=y,
    conditions={
        "temperature": 25.0,      # Experimental condition
        "sample": "polymer_A",     # Experimental condition
        "replicate": 1,            # Experimental condition
    },
    details={
        "operator": "Alice",       # Administrative
        "timestamp": "2024-10-19", # Administrative
        "instrument": "Rheometer_3", # Administrative
        "notes": "First run",      # Administrative
    }
)
```

**Grouping by Conditions**:

```python
# Group measurements by experimental conditions
from piblin_jax.dataio.hierarchy import group_by_conditions

groups = group_by_conditions(measurements, keys=["temperature", "sample"])
# Only looks in `conditions`, not `details`
```

## Consequences

### Positive

1. **Clear Semantics**:
   - Explicit: "This is an experimental condition"
   - No ambiguity about parameter meaning
   - Self-documenting code

2. **Intelligent Grouping**:
   - `group_by_conditions` knows where to look
   - Can identify varying vs constant conditions
   - Automatic hierarchy building

3. **Propagation Rules**:
   - Conditions always propagate (they define the data)
   - Details may be dropped/merged (they're contextual)
   - Clear expectations for transform behavior

4. **Type Safety Potential**:
   - Future: Could make `conditions` typed (Pydantic models)
   - `details` stays flexible (dict)
   - Best of both worlds

5. **Better Documentation**:
   - Users see which parameters are scientifically meaningful
   - Easier to write analysis code
   - Clearer experimental design

### Negative

1. **User Decision Required**:
   - "Is `batch_number` a condition or detail?"
   - Gray area parameters can cause confusion
   - Need guidelines for classification

2. **API Verbosity**:
   - Two dictionaries instead of one
   - More typing when creating datasets
   - May seem redundant for simple cases

3. **Migration Burden**:
   - piblin users need to split their metadata
   - No automatic migration (requires judgment)
   - Documentation needed for transition

4. **Cognitive Load**:
   - Yet another concept to learn
   - Must remember the distinction
   - More complex than single dict

### Mitigation Strategies

**Decision Guidelines**:

```python
# ✅ CONDITIONS: Answers "What experiment was this?"
conditions = {
    "temperature": 25.0,        # Would you plot this on an axis?
    "sample_id": "ABC",         # Would you group/color by this?
    "concentration": 0.5,       # Does this scientifically define the measurement?
}

# ✅ DETAILS: Answers "How/when/who/where was this measured?"
details = {
    "operator": "Alice",        # Administrative tracking
    "timestamp": "2024-10-19",  # When it happened
    "instrument": "Rheometer",  # What instrument
    "notes": "Repeated run",    # Comments
}
```

**Rule of Thumb**:

> **If you would plot it on a graph axis or use it to color/group data → `conditions`**
>
> **If it's for record-keeping or context → `details`**

**For Simple Cases**:

```python
# Can leave one empty if not needed
dataset = OneDimensionalDataset(
    x=x, y=y,
    conditions={"temperature": 25.0},  # Just the essential
    details={}                          # Empty OK
)
```

## Alternatives Considered

### Alternative 1: Single Metadata Dictionary (piblin Style)

```python
@dataclass(frozen=True)
class Dataset:
    data: np.ndarray
    metadata: dict[str, Any]  # Everything mixed together
```

**Pros**:
- Simpler mental model
- Less typing
- Familiar from piblin

**Cons**:
- No semantic distinction
- Grouping requires knowing which keys are conditions
- Propagation rules unclear
- Harder to validate

**Why Rejected**: Loses important semantic information

### Alternative 2: Typed Conditions, Flexible Details

```python
class Conditions(BaseModel):  # Pydantic
    temperature: float
    sample_id: str
    replicate: int

@dataclass(frozen=True)
class Dataset:
    data: np.ndarray
    conditions: Conditions  # Typed
    details: dict[str, Any]  # Flexible
```

**Pros**:
- Type safety for conditions
- Flexibility for details
- Best of both worlds?

**Cons**:
- Every user must define condition schema
- Too rigid for exploratory analysis
- Doesn't support dynamic conditions

**Why Rejected**: Too restrictive for general use (could be future option)

### Alternative 3: Three-Dictionary System

```python
@dataclass(frozen=True)
class Dataset:
    data: np.ndarray
    conditions: dict[str, Any]      # Experimental
    processing: dict[str, Any]      # Transform history
    administrative: dict[str, Any]  # Operator, timestamp, etc.
```

**Pros**:
- Even more semantic clarity
- Processing history separated

**Cons**:
- More complexity
- Processing history can be in details
- Overkill for most cases

**Why Rejected**: Two dictionaries sufficient, three is too many

### Alternative 4: Metadata Class with Properties

```python
class Metadata:
    def __init__(self, **kwargs):
        self._data = kwargs

    @property
    def conditions(self) -> dict:
        # Auto-detect conditions based on keys
        return {k: v for k, v in self._data.items()
                if k in KNOWN_CONDITIONS}

    @property
    def details(self) -> dict:
        return {k: v for k, v in self._data.items()
                if k not in KNOWN_CONDITIONS}
```

**Pros**:
- Automatic classification
- Single dict for user

**Cons**:
- Magic behavior
- Requires maintaining KNOWN_CONDITIONS list
- What about custom conditions?
- Doesn't scale to new experiments

**Why Rejected**: Too much magic, not explicit enough

## Implementation Notes

### For Dataset Creation

```python
from piblin_jax.data.datasets import OneDimensionalDataset

dataset = OneDimensionalDataset(
    independent_variable_data=shear_rate,
    dependent_variable_data=viscosity,
    conditions={
        "temperature": 25.0,
        "sample": "polymer_A",
        "concentration": 0.05,
    },
    details={
        "operator": "Researcher Name",
        "instrument": "Rheometer XYZ",
        "date": "2024-10-19",
    }
)
```

### For Grouping Operations

```python
from piblin_jax.dataio.hierarchy import group_by_conditions, identify_varying_conditions

# Group measurements by specific conditions
groups = group_by_conditions(measurements, keys=["temperature", "sample"])

# Identify which conditions vary across measurements
varying = identify_varying_conditions(measurements)
# Returns: {"temperature", "sample"} (if these vary)
```

### For Transform Developers

```python
class MyTransform(DatasetTransform):
    def apply_to(self, dataset: Dataset) -> Dataset:
        new_data = self._process(dataset.data)

        # Conditions ALWAYS propagate
        # Details USUALLY propagate
        return dataset.copy_with(
            dependent_variable_data=new_data
            # conditions and details automatically copied
        )
```

### Migration from piblin

```python
# piblin style
piblin_metadata = {
    "temperature": 25.0,
    "sample": "A",
    "operator": "Alice",
    "timestamp": "2024-10-19"
}

# piblin-jax style - split appropriately
conditions = {
    "temperature": 25.0,  # Experimental
    "sample": "A",         # Experimental
}
details = {
    "operator": "Alice",       # Administrative
    "timestamp": "2024-10-19", # Administrative
}
```

## Decision Guidelines Summary

**Put in `conditions` if**:
- ✅ You would plot it on a graph axis
- ✅ You would group/filter data by it
- ✅ It scientifically defines the experiment
- ✅ It's a controlled/measured variable
- ✅ It varies systematically across measurements

**Put in `details` if**:
- ✅ It's for record-keeping
- ✅ It's administrative metadata
- ✅ It's a timestamp or operator name
- ✅ It's a comment or note
- ✅ It's instrument/software version

**Examples by Category**:

| Conditions | Details |
|------------|---------|
| temperature | operator |
| pressure | timestamp |
| sample_id | instrument_id |
| concentration | software_version |
| replicate_number | calibration_date |
| strain_rate | notes/comments |
| pH | file_path |

## Related Decisions

- **ADR-003**: Transform Hierarchy (metadata propagation through levels)
- Future ADR: Typed Metadata Schemas (Pydantic models for conditions)

## References

- Scientific Data Management: Metadata Best Practices
- FAIR Principles: Findable, Accessible, Interoperable, Reusable
- Dublin Core Metadata Element Set
- DataCite Metadata Schema

## Revision History

- 2024-10-19: Initial ADR creation
- Status: Accepted and implemented
