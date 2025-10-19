# quantiq User Guide

## Overview

**quantiq** is a modern JAX-powered framework for measurement data science, providing a complete reimplementation of piblin with significant performance enhancements and advanced uncertainty quantification capabilities.

### Key Features

- **JAX Backend**: JIT compilation and automatic GPU acceleration
- **NumPyro Bayesian Inference**: Advanced uncertainty quantification
- **NLSQ Integration**: High-performance non-linear fitting
- **100% piblin API Compatibility**: Drop-in replacement for existing code
- **Performance**: 5-10x CPU speedup, 50-100x GPU speedup for large datasets

## Installation

```bash
pip install quantiq
```

## Quick Start

### Basic Usage

```python
import quantiq
# Or for piblin compatibility:
import quantiq as piblin

import numpy as np

# Create a dataset
x = np.linspace(0, 10, 50)
y = 2.0 * x + 1.0 + np.random.randn(50) * 0.1

dataset = quantiq.OneDimensionalDataset(
    independent_variable_data=x,
    dependent_variable_data=y
)
```

### Data Transforms

```python
from quantiq import Pipeline, LambdaTransform

# Create transforms
normalize = LambdaTransform(lambda y: (y - y.min()) / (y.max() - y.min()))
smooth = LambdaTransform(lambda y: np.convolve(y, np.ones(5)/5, mode='same'))

# Build pipeline
pipeline = Pipeline([smooth, normalize])

# Apply to data
result = pipeline.apply_to(dataset, make_copy=True)
```

### Uncertainty Quantification

```python
# Bootstrap uncertainty
dataset_with_unc = dataset.with_uncertainty(
    n_samples=1000,
    method='bootstrap',
    keep_samples=True,
    level=0.95
)

# Get credible intervals
lower, upper = dataset_with_unc.credible_intervals

# Visualize with uncertainty bands
fig, ax = dataset_with_unc.visualize(
    show_uncertainty=True,
    level=0.95,
    xlabel='Time (s)',
    ylabel='Signal (V)'
)
```

### Curve Fitting

```python
from quantiq.fitting import fit_curve

def power_law(x, A, n):
    return A * x**n

# Fit data
result = fit_curve(power_law, x, y, p0=[1.0, 0.5])

print(f"Parameters: {result['params']}")
print(f"Method used: {result['method']}")  # 'nlsq' or 'scipy'
```

### Bayesian Modeling

```python
from quantiq.bayesian.models import PowerLawModel

# Create model
model = PowerLawModel()

# Fit with MCMC
model.fit(
    shear_rate,
    viscosity,
    num_warmup=500,
    num_samples=1000,
    num_chains=2
)

# Get parameter distributions
A_samples = model.samples['A']
n_samples = model.samples['n']

# Get credible intervals
A_lower, A_upper = model.get_credible_intervals('A', level=0.95)

# Make predictions
x_pred = np.logspace(-2, 2, 100)
predictions = model.predict(x_pred)
```

## Core Concepts

### Datasets

quantiq provides several dataset types:

- `OneDimensionalDataset`: 1D paired data (x, y)
- `TwoDimensionalDataset`: 2D data with two independent variables
- `ThreeDimensionalDataset`: 3D volumetric data
- `ZeroDimensionalDataset`: Single scalar values
- `Histogram`: Binned data
- `Distribution`: Probability density functions

### Collections

Organize related measurements:

- `Measurement`: Single measurement with metadata
- `MeasurementSet`: Collection of measurements
- `Experiment`: Named collection with conditions
- `ExperimentSet`: Multiple experiments

### Transforms

Transform data through pipelines:

- `LambdaTransform`: Custom functions
- `Pipeline`: Chain multiple transforms
- Dataset-specific transforms (smoothing, normalization, etc.)

## Advanced Features

### Propagating Uncertainty Through Pipelines

```python
# Create dataset with uncertainty
dataset_unc = dataset.with_uncertainty(
    n_samples=1000,
    method='bootstrap',
    keep_samples=True
)

# Create pipeline
pipeline = Pipeline([smooth_transform, normalize_transform])

# Propagate uncertainty through pipeline
result = pipeline.apply_to(dataset_unc, propagate_uncertainty=True)

# Result now has uncertainty information
assert result.has_uncertainty
```

### Working with Collections

```python
from quantiq import Measurement, MeasurementSet

# Create measurements
m1 = Measurement(
    "Sample A",
    dataset1,
    conditions={"temperature": 298.15, "pressure": 101.3}
)

m2 = Measurement(
    "Sample B",
    dataset2,
    conditions={"temperature": 308.15, "pressure": 101.3}
)

# Create set
measurement_set = MeasurementSet([m1, m2])

# Apply transforms to all measurements
transformed_set = transform.apply_to(measurement_set)
```

### Custom Transforms

```python
from quantiq.transform.base import DatasetTransform

class MyCustomTransform(DatasetTransform):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def _apply(self, dataset):
        dataset._dependent_variable_data *= self.scale_factor
        return dataset

# Use it
transform = MyCustomTransform(scale_factor=2.0)
result = transform.apply_to(dataset)
```

## Performance Tips

1. **Use make_copy=False for in-place operations**:
   ```python
   # Faster (modifies original)
   transform.apply_to(dataset, make_copy=False)

   # Safer (preserves original)
   result = transform.apply_to(dataset, make_copy=True)
   ```

2. **Batch operations when possible**:
   ```python
   # Apply to collection instead of individual measurements
   pipeline.apply_to(measurement_set)
   ```

3. **Use GPU for large datasets**:
   JAX automatically uses GPU when available - no code changes needed!

## piblin Compatibility

quantiq is designed as a drop-in replacement for piblin:

```python
# Old piblin code
import piblin
dataset = piblin.OneDimensionalDataset(...)

# New quantiq code (identical API)
import quantiq as piblin  # Just change this line!
dataset = piblin.OneDimensionalDataset(...)
```

All core piblin functionality is supported with enhanced performance.

## API Reference

For detailed API documentation, see the [API Documentation](api/index.html).

## Examples

See the `examples/` directory for Jupyter notebooks demonstrating:
- Basic data manipulation
- Advanced uncertainty quantification
- Bayesian modeling workflows
- Performance optimization

## Getting Help

- Documentation: https://quantiq.readthedocs.io
- GitHub Issues: https://github.com/yourusername/quantiq/issues
- Discussion Forum: https://github.com/yourusername/quantiq/discussions
