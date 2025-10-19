"""
Basic Usage Example for quantiq
================================

This example demonstrates the fundamental usage patterns of quantiq:
- Creating 1D datasets from NumPy arrays
- Applying simple transformations (smoothing, normalization)
- Building transform pipelines
- Visualizing results

Key concepts:
- OneDimensionalDataset for paired (x, y) data
- Transform.apply_to() for data processing
- Pipeline for chaining transforms
- Dataset.visualize() for quick plots

Expected output: Matplotlib figures showing original vs. transformed data
"""

import numpy as np
import matplotlib.pyplot as plt

# Import quantiq dataset and transform classes
from quantiq.data.datasets import OneDimensionalDataset
from quantiq.transform.dataset import GaussianSmooth, Normalize
from quantiq.transform import Pipeline

print("=" * 70)
print("quantiq Basic Usage Example")
print("=" * 70)

# =============================================================================
# Section 1: Create a Dataset
# =============================================================================
print("\n[1] Creating a 1D dataset...")

# Generate synthetic noisy data
np.random.seed(42)  # For reproducibility
x = np.linspace(0, 10, 200)
y_clean = np.sin(x) * np.exp(-x / 10)  # Damped sine wave
noise = 0.15 * np.random.randn(len(x))
y_noisy = y_clean + noise

# Create quantiq dataset with metadata
dataset = OneDimensionalDataset(
    independent_variable_data=x,
    dependent_variable_data=y_noisy,
    conditions={
        "temperature": 25.0,  # °C
        "sample": "Example_A",
        "noise_level": 0.15
    },
    details={
        "operator": "Tutorial User",
        "instrument": "Synthetic Data Generator",
        "date": "2025-10-19"
    }
)

print(f"   ✓ Created dataset with {len(x)} data points")
print(f"   ✓ Conditions: {dataset.conditions}")
print(f"   ✓ Data range: x=[{x.min():.2f}, {x.max():.2f}], y=[{y_noisy.min():.2f}, {y_noisy.max():.2f}]")

# =============================================================================
# Section 2: Apply a Single Transform (Gaussian Smoothing)
# =============================================================================
print("\n[2] Applying Gaussian smoothing...")

# Create smoothing transform (sigma controls smoothing strength)
smooth_transform = GaussianSmooth(sigma=2.0)

# Apply transform (make_copy=True preserves original)
smoothed_dataset = smooth_transform.apply_to(dataset, make_copy=True)

print(f"   ✓ Applied Gaussian smoothing with sigma=2.0")
print(f"   ✓ Original dataset unchanged: {len(dataset.dependent_variable_data)} points")
print(f"   ✓ Smoothed dataset created: {len(smoothed_dataset.dependent_variable_data)} points")

# =============================================================================
# Section 3: Build and Apply a Transform Pipeline
# =============================================================================
print("\n[3] Building transform pipeline...")

# Create pipeline: smooth → normalize
pipeline = Pipeline([
    GaussianSmooth(sigma=2.0),      # Step 1: Remove noise
    Normalize(method="min-max")     # Step 2: Normalize to [0, 1]
])

# Apply entire pipeline
processed_dataset = pipeline.apply_to(dataset, make_copy=True)

print(f"   ✓ Pipeline created with {len(pipeline.transforms)} transforms")
print(f"   ✓ Pipeline applied successfully")
print(f"   ✓ Processed data range: [{processed_dataset.dependent_variable_data.min():.3f}, "
      f"{processed_dataset.dependent_variable_data.max():.3f}]")

# =============================================================================
# Section 4: Visualize Results
# =============================================================================
print("\n[4] Creating visualizations...")

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('quantiq Basic Usage Example', fontsize=16, fontweight='bold')

# Plot 1: Original noisy data
axes[0, 0].plot(x, y_noisy, 'o', markersize=3, alpha=0.5, label='Noisy data')
axes[0, 0].plot(x, y_clean, 'r-', linewidth=2, label='True signal')
axes[0, 0].set_xlabel('X')
axes[0, 0].set_ylabel('Y')
axes[0, 0].set_title('Step 1: Original Noisy Data')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: After smoothing
axes[0, 1].plot(x, y_noisy, 'o', markersize=2, alpha=0.3, label='Original')
axes[0, 1].plot(smoothed_dataset.independent_variable_data,
                smoothed_dataset.dependent_variable_data,
                'b-', linewidth=2, label='Smoothed (σ=2.0)')
axes[0, 1].set_xlabel('X')
axes[0, 1].set_ylabel('Y')
axes[0, 1].set_title('Step 2: After Gaussian Smoothing')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: After normalization
axes[1, 0].plot(processed_dataset.independent_variable_data,
                processed_dataset.dependent_variable_data,
                'g-', linewidth=2, label='Smoothed + Normalized')
axes[1, 0].set_xlabel('X')
axes[1, 0].set_ylabel('Y (normalized)')
axes[1, 0].set_title('Step 3: After Pipeline (Smooth + Normalize)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Comparison of all stages
axes[1, 1].plot(x, y_noisy, 'gray', alpha=0.3, linewidth=1, label='Original')
axes[1, 1].plot(smoothed_dataset.independent_variable_data,
                smoothed_dataset.dependent_variable_data,
                'b-', linewidth=2, label='Smoothed')
axes[1, 1].plot(processed_dataset.independent_variable_data,
                processed_dataset.dependent_variable_data * y_noisy.max(),  # Rescale for comparison
                'g-', linewidth=2, label='Smoothed + Norm (rescaled)')
axes[1, 1].set_xlabel('X')
axes[1, 1].set_ylabel('Y')
axes[1, 1].set_title('Comparison of Processing Stages')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
print("   ✓ Visualization created")

# =============================================================================
# Section 5: Summary and Key Takeaways
# =============================================================================
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print("""
Key Takeaways:
1. OneDimensionalDataset stores paired (x, y) data with metadata
   - Conditions: Experimental parameters (temperature, sample, etc.)
   - Details: Context information (operator, date, instrument)

2. Transforms process data immutably (make_copy=True)
   - Original dataset preserved
   - New dataset created with transformed data
   - Metadata automatically carried forward

3. Pipelines chain multiple transforms sequentially
   - Composable: Easy to build complex processing workflows
   - Reusable: Same pipeline can process multiple datasets
   - Efficient: Single copy made at entry, transforms applied in sequence

4. Visualization built-in
   - dataset.visualize() for quick plots
   - Or use matplotlib directly for custom visualizations

Next Steps:
- Explore bayesian_parameter_estimation.py for uncertainty quantification
- See transform_pipeline_example.py for advanced pipeline usage
- Check uncertainty_propagation_example.py for error analysis
""")

plt.show()
print("\n✓ Example completed successfully!")
