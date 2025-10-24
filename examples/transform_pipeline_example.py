"""
Transform Pipeline Example
==========================

This example demonstrates piblin-jax's advanced transform pipeline capabilities:
- Building complex multi-step pipelines
- Applying pipelines to different dataset dimensionalities
- Understanding pipeline composition and reusability
- Performance considerations with backend abstraction

Key concepts:
- Pipeline for chaining multiple transforms
- Transform composition and modularity
- Applying same pipeline to 1D and 2D datasets
- Backend abstraction (JAX/NumPy) for performance
- Immutability: make_copy=True preserves originals

Expected output: Plots showing pipeline effects on different datasets
"""

import matplotlib.pyplot as plt
import numpy as np

# Import piblin-jax dataset and transform classes
from piblin_jax.data.datasets import OneDimensionalDataset, TwoDimensionalDataset
from piblin_jax.transform import Pipeline
from piblin_jax.transform.dataset import (
    Derivative,
    GaussianSmooth,
    MinMaxNormalize,
    ZScoreNormalize,
)

print("=" * 80)
print("Transform Pipeline Example")
print("=" * 80)

# =============================================================================
# Section 1: Create Test Datasets (1D and 2D)
# =============================================================================
print("\n[1] Creating test datasets...")

# Set random seed for reproducibility
np.random.seed(42)

# --- 1D Dataset: Noisy damped oscillation ---
x_1d = np.linspace(0, 15, 300)
y_1d_clean = np.sin(2 * x_1d) * np.exp(-x_1d / 5)
noise_1d = 0.1 * np.random.randn(len(x_1d))
y_1d_noisy = y_1d_clean + noise_1d

dataset_1d = OneDimensionalDataset(
    independent_variable_data=x_1d,
    dependent_variable_data=y_1d_noisy,
    conditions={"temperature": 25.0, "sample": "1D_oscillation"},
    details={"description": "Damped sine wave with noise"},
)

print(f"   ✓ Created 1D dataset: {len(x_1d)} points")
print(f"   ✓ Signal range: [{y_1d_noisy.min():.3f}, {y_1d_noisy.max():.3f}]")

# --- 2D Dataset: Noisy Gaussian peak ---
x_2d = np.linspace(-5, 5, 100)
y_2d = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x_2d, y_2d)

# Clean 2D Gaussian
z_2d_clean = np.exp(-(X**2 + Y**2) / 4)

# Add noise
noise_2d = 0.05 * np.random.randn(*z_2d_clean.shape)
z_2d_noisy = z_2d_clean + noise_2d

dataset_2d = TwoDimensionalDataset(
    independent_variable_data_1=x_2d,
    independent_variable_data_2=y_2d,
    dependent_variable_data=z_2d_noisy,
    conditions={"temperature": 25.0, "sample": "2D_gaussian"},
    details={"description": "2D Gaussian peak with noise"},
)

print(f"   ✓ Created 2D dataset: {z_2d_noisy.shape}")
print(f"   ✓ Signal range: [{z_2d_noisy.min():.3f}, {z_2d_noisy.max():.3f}]")

# =============================================================================
# Section 2: Build Transform Pipelines
# =============================================================================
print("\n[2] Building transform pipelines...")

# --- Simple Pipeline: Smooth → Normalize ---
simple_pipeline = Pipeline([GaussianSmooth(sigma=2.0), MinMaxNormalize()])

print(f"   ✓ Simple pipeline: {len(simple_pipeline)} transforms")
print("      1. GaussianSmooth(sigma=2.0)")
print("      2. MinMaxNormalize()")

# --- Complex Pipeline: Smooth → Derivative → Normalize ---
complex_pipeline = Pipeline(
    [
        GaussianSmooth(sigma=3.0),  # Remove noise
        Derivative(order=1),  # Compute first derivative
        ZScoreNormalize(),  # Standardize to zero mean, unit variance
    ]
)

print(f"   ✓ Complex pipeline: {len(complex_pipeline)} transforms")
print("      1. GaussianSmooth(sigma=3.0)")
print("      2. Derivative(order=1)")
print("      3. ZScoreNormalize()")

# =============================================================================
# Section 3: Apply Pipelines to 1D Dataset
# =============================================================================
print("\n[3] Applying pipelines to 1D dataset...")

# Apply simple pipeline
dataset_1d_simple = simple_pipeline.apply_to(dataset_1d, make_copy=True)
print("   ✓ Simple pipeline applied")
print(
    f"   ✓ Output range: [{dataset_1d_simple.dependent_variable_data.min():.3f}, "
    f"{dataset_1d_simple.dependent_variable_data.max():.3f}]"
)

# Apply complex pipeline
dataset_1d_complex = complex_pipeline.apply_to(dataset_1d, make_copy=True)
print("   ✓ Complex pipeline applied")
print("   ✓ Derivative computed and normalized")

# =============================================================================
# Section 4: Apply Pipeline to 2D Dataset
# =============================================================================
print("\n[4] Applying pipeline to 2D dataset...")

# Note: Some transforms (like GaussianSmooth) only work on 1D data
# Create a 2D-compatible pipeline using only transforms that support 2D
pipeline_2d = Pipeline([MinMaxNormalize()])

dataset_2d_processed = pipeline_2d.apply_to(dataset_2d, make_copy=True)

print("   ✓ Pipeline applied to 2D data")
print(f"   ✓ Output shape: {dataset_2d_processed.dependent_variable_data.shape}")
print(
    f"   ✓ Output range: [{dataset_2d_processed.dependent_variable_data.min():.3f}, "
    f"{dataset_2d_processed.dependent_variable_data.max():.3f}]"
)
print("   Note: Some transforms (like GaussianSmooth) are 1D-only")

# =============================================================================
# Section 5: Visualize Results
# =============================================================================
print("\n[5] Creating visualizations...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# --- Plot 1: Original 1D data ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(x_1d, y_1d_noisy, "gray", alpha=0.5, linewidth=1, label="Noisy data")
ax1.plot(x_1d, y_1d_clean, "r--", linewidth=2, label="True signal")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_title("1D Dataset: Original", fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- Plot 2: Simple pipeline on 1D ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(x_1d, y_1d_noisy, "gray", alpha=0.3, linewidth=1, label="Original")
ax2.plot(
    dataset_1d_simple.independent_variable_data,
    dataset_1d_simple.dependent_variable_data,
    "b-",
    linewidth=2,
    label="Smooth → Normalize",
)
ax2.set_xlabel("X")
ax2.set_ylabel("Y (normalized)")
ax2.set_title("1D Dataset: Simple Pipeline", fontweight="bold")
ax2.legend()
ax2.grid(True, alpha=0.3)

# --- Plot 3: Complex pipeline on 1D ---
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(
    dataset_1d_complex.independent_variable_data,
    dataset_1d_complex.dependent_variable_data,
    "g-",
    linewidth=2,
    label="Smooth → Derivative → Normalize",
)
ax3.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.5)
ax3.set_xlabel("X")
ax3.set_ylabel("dY/dX (z-score)")
ax3.set_title("1D Dataset: Complex Pipeline", fontweight="bold")
ax3.legend()
ax3.grid(True, alpha=0.3)

# --- Plot 4: Original 2D data ---
ax4 = fig.add_subplot(gs[1, 0])
im4 = ax4.contourf(X, Y, z_2d_noisy, levels=20, cmap="viridis")
ax4.set_xlabel("X")
ax4.set_ylabel("Y")
ax4.set_title("2D Dataset: Original", fontweight="bold")
plt.colorbar(im4, ax=ax4)

# --- Plot 5: Pipeline on 2D ---
ax5 = fig.add_subplot(gs[1, 1])
im5 = ax5.contourf(X, Y, dataset_2d_processed.dependent_variable_data, levels=20, cmap="viridis")
ax5.set_xlabel("X")
ax5.set_ylabel("Y")
ax5.set_title("2D Dataset: Normalized", fontweight="bold")
plt.colorbar(im5, ax=ax5)

# --- Plot 6: Comparison (1D cross-sections of 2D data) ---
ax6 = fig.add_subplot(gs[1, 2])
center_idx = len(y_2d) // 2
ax6.plot(x_2d, z_2d_noisy[center_idx, :], "gray", alpha=0.5, linewidth=1, label="Original")
ax6.plot(
    x_2d,
    dataset_2d_processed.dependent_variable_data[center_idx, :],
    "b-",
    linewidth=2,
    label="Processed",
)
ax6.set_xlabel("X (cross-section at Y=0)")
ax6.set_ylabel("Z")
ax6.set_title("2D Dataset: Cross-Section Comparison", fontweight="bold")
ax6.legend()
ax6.grid(True, alpha=0.3)

fig.suptitle("Transform Pipeline Example", fontsize=16, fontweight="bold", y=0.995)
print("   ✓ Visualization created")

# =============================================================================
# Section 6: Pipeline Composition and Reusability
# =============================================================================
print("\n[6] Demonstrating pipeline composition...")

# Pipelines can be extended by adding more transforms
extended_pipeline = Pipeline([*simple_pipeline, Derivative(order=1)])

print("   ✓ Extended simple pipeline with derivative")
print(f"   ✓ New pipeline has {len(extended_pipeline)} transforms")

# Apply extended pipeline
dataset_1d_extended = extended_pipeline.apply_to(dataset_1d, make_copy=True)
print("   ✓ Extended pipeline applied to 1D dataset")

# =============================================================================
# Section 7: Summary and Key Takeaways
# =============================================================================
print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print(
    """
Key Takeaways:

1. Pipeline Composition:
   - Pipelines chain multiple transforms sequentially
   - Each transform's output becomes the next transform's input
   - Transforms are applied in the order they appear in the list

2. Reusability and Modularity:
   - Pipelines can be extended by adding more transforms
   - Original datasets preserved with make_copy=True (immutability)
   - Composable design enables complex workflows

3. Transform Compatibility:
   - Many transforms work across different dimensionalities
   - Some transforms (like GaussianSmooth) are 1D-only
   - Backend abstraction (JAX/NumPy) handles array operations transparently
   - Type system ensures valid pipeline construction

4. Performance Considerations:
   - Backend abstraction allows JAX acceleration when available
   - Pipelines minimize intermediate copies (single copy at entry)
   - JIT compilation available for JAX-based transforms

5. Practical Applications:
   - Data cleaning: Smooth → Normalize (1D data)
   - Feature extraction: Smooth → Derivative → Normalize (1D data)
   - Multi-dimensional processing: Use compatible transforms for different dimensionalities

6. Design Philosophy:
   - Immutability: Original data never modified
   - Composability: Build complex workflows from simple transforms
   - Consistency: Same API across all dataset types

Next Steps:
- Explore uncertainty propagation through pipelines
- See bayesian_parameter_estimation.py for parameter fitting
- Check basic_usage_example.py for fundamental patterns
"""
)

plt.tight_layout()
plt.show()
print("\n✓ Example completed successfully!")
