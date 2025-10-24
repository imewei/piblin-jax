"""Data types and utilities for piblin-jax.

This package provides the core data structures for measurement data science:
- **Datasets**: Typed array containers (0D, 1D, 2D, 3D, composite, distributions)
- **Collections**: Hierarchical measurement organization (Measurement, MeasurementSet, Experiment, ExperimentSet)
- **Metadata**: Structured conditions and details with validation and merging
- **ROI**: Region of interest definitions for selective data analysis

## Package Structure

### Datasets Module (`piblin_jax.data.datasets`)

Core dataset classes for different dimensionalities:
- **ZeroDimensionalDataset**: Scalar values with metadata
- **OneDimensionalDataset**: Paired (x, y) data (time series, spectra, etc.)
- **TwoDimensionalDataset**: 2D grid data (heatmaps, images)
- **ThreeDimensionalDataset**: 3D volumetric data
- **OneDimensionalCompositeDataset**: Multiple dependent variables with shared x-axis
- **Histogram**: Binned frequency distributions
- **Distribution**: Probability density functions

All datasets include:
- JAX/NumPy backend abstraction for performance
- Metadata system (conditions and details)
- Uncertainty quantification support
- Immutable design for functional programming
- Type-safe API with comprehensive type hints

### Collections Module (`piblin_jax.data.collections`)

Hierarchical organization for experimental data::

    ExperimentSet (top level)
    └── Experiment (single experimental condition set)
        └── MeasurementSet (group of related measurements)
            └── Measurement (individual measurement with datasets)

**Collection Types**:
- **Measurement**: Container for related datasets from one measurement
- **MeasurementSet**: Group of measurements (e.g., replicate trials)
- **ConsistentMeasurementSet**: Enforces same conditions across measurements
- **TabularMeasurementSet**: Optimized for tabular data access
- **TidyMeasurementSet**: Tidy (long-form) data representation
- **Experiment**: Collection of measurement sets under same conditions
- **ExperimentSet**: Top-level container for multiple experiments

### Metadata Module (`piblin_jax.data.metadata`)

Metadata management utilities:
- **Merging**: Combine metadata from multiple sources with conflict resolution
- **Validation**: Type checking and schema validation
- **Extraction**: Parse metadata from filenames, paths, and file headers
- **Separation**: Distinguish experimental conditions from details

**Supported Operations**:
- `merge_metadata()` - Combine metadata with strategies (override, keep_first, raise, list)
- `validate_metadata()` - Validate against schemas with type checking
- `extract_from_filename()` - Parse metadata from file naming patterns
- `extract_from_path()` - Extract metadata from directory structure
- `parse_header_metadata()` - Parse comment headers in data files
- `separate_conditions_details()` - Split metadata into conditions and details

### ROI Module (`piblin_jax.data.roi`)

Region of interest definitions for selective analysis:
- **ROI**: Base class for defining regions in datasets
- Support for 1D, 2D, and 3D regions
- Boolean masking and index-based selection
- Integration with transform pipeline

## Usage Examples

### Basic Dataset Creation

Example::

    import numpy as np
    from piblin_jax.data.datasets import OneDimensionalDataset

    # Create 1D dataset
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    dataset = OneDimensionalDataset(
        independent_variable_data=x,
        dependent_variable_data=y,
        conditions={"temperature": 25.0, "sample": "A"},
        details={"operator": "John", "date": "2025-01-15"}
    )

### Building Hierarchical Collections

Example::

    from piblin_jax.data.collections import Measurement, MeasurementSet, Experiment

    # Create measurements
    m1 = Measurement({"dataset1": dataset1, "dataset2": dataset2})
    m2 = Measurement({"dataset1": dataset3, "dataset2": dataset4})

    # Group into measurement set
    mset = MeasurementSet([m1, m2])

    # Create experiment
    experiment = Experiment({"trial1": mset})

### Metadata Operations

Example::

    from piblin_jax.data import metadata

    # Merge metadata from multiple sources
    file_meta = metadata.extract_from_filename("sample_A1_25C.csv")
    path_meta = {"experiment": "viscosity"}
    combined = metadata.merge_metadata([file_meta, path_meta])

    # Validate against schema
    schema = {"temperature": float, "sample": str}
    metadata.validate_metadata(combined, schema=schema)

    # Separate conditions from details
    conditions, details = metadata.separate_conditions_details(
        combined,
        condition_keys=["temperature", "pressure"]
    )

## Design Principles

1. **Type Safety**: Comprehensive type hints for all public APIs
2. **Immutability**: Datasets are immutable by design (functional programming)
3. **Backend Agnostic**: JAX for performance, NumPy for compatibility
4. **Metadata-First**: Rich metadata support throughout the hierarchy
5. **Hierarchical Organization**: Natural experiment → measurement → dataset structure
6. **Extensibility**: Easy to add custom dataset types and collection classes

## See Also

- `piblin_jax.transform` - Transform pipelines for data processing
- `piblin_jax.bayesian` - Bayesian uncertainty quantification
- `piblin_jax.dataio` - File I/O for reading experimental data
- `piblin_jax.backend` - Backend abstraction layer (JAX/NumPy)
"""

from . import collections, datasets, metadata, roi

__all__ = ["collections", "datasets", "metadata", "roi"]
