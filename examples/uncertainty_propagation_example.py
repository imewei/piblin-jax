"""
Uncertainty Propagation Example
================================

This example demonstrates piblin-jax's uncertainty quantification and propagation:
- Creating datasets with measurement uncertainties
- Propagating uncertainties through transform pipelines
- Comparing Monte Carlo vs analytical uncertainty propagation
- Visualizing uncertainty bands in results

Key concepts:
- OneDimensionalDataset with uncertainty data
- Uncertainty propagation through transforms
- Monte Carlo sampling for uncertainty estimation
- Visualization of confidence/credible intervals
- Error analysis in scientific data processing

Expected output: Plots showing uncertainty bands through processing pipeline
"""

import matplotlib.pyplot as plt
import numpy as np

# Import piblin-jax dataset and transform classes
from piblin_jax.data.datasets import OneDimensionalDataset
from piblin_jax.transform import Pipeline
from piblin_jax.transform.dataset import Derivative, GaussianSmooth, MinMaxNormalize

print("=" * 80)
print("Uncertainty Propagation Example")
print("=" * 80)

# =============================================================================
# Section 1: Create Dataset with Measurement Uncertainties
# =============================================================================
print("\n[1] Creating dataset with measurement uncertainties...")

# Set random seed for reproducibility
np.random.seed(42)

# Generate "true" underlying signal
x = np.linspace(0, 10, 100)
y_true = 2.5 * np.exp(-x / 5) * np.sin(2 * x)

# Measurement uncertainties (heteroscedastic: varies with x)
# Typical in experiments: uncertainty increases at edges
uncertainty_y = 0.1 + 0.05 * (x / 10) ** 2

# Generate measured values with realistic noise
y_measured = y_true + np.random.randn(len(x)) * uncertainty_y

# Create dataset with uncertainty information
dataset = OneDimensionalDataset(
    independent_variable_data=x,
    dependent_variable_data=y_measured,
    conditions={"temperature": 25.0, "sample": "A"},
    details={"instrument": "Rheometer", "operator": "Scientist"},
)

# Store uncertainty as metadata (piblin-jax pattern)
# Note: Future versions may have dedicated uncertainty field
dataset.details["dependent_variable_uncertainty"] = uncertainty_y

print(f"   ✓ Created dataset with {len(x)} points")
print(f"   ✓ Signal range: [{y_measured.min():.3f}, {y_measured.max():.3f}]")
print(f"   ✓ Uncertainty range: [{uncertainty_y.min():.4f}, {uncertainty_y.max():.4f}]")
print(f"   ✓ Relative uncertainty: {(uncertainty_y.mean() / np.abs(y_measured).mean() * 100):.1f}%")

# =============================================================================
# Section 2: Monte Carlo Uncertainty Propagation
# =============================================================================
print("\n[2] Propagating uncertainty using Monte Carlo sampling...")

# Define processing pipeline
pipeline = Pipeline([GaussianSmooth(sigma=1.5), MinMaxNormalize()])

print(f"   ✓ Pipeline: {len(pipeline)} transforms")
print("      1. GaussianSmooth(sigma=1.5)")
print("      2. MinMaxNormalize()")

# Monte Carlo: Generate ensemble of realizations
n_samples = 500
print(f"\n   ⏳ Running Monte Carlo with {n_samples} samples...")

# Storage for Monte Carlo results
mc_results = np.zeros((n_samples, len(x)))

for i in range(n_samples):
    # Generate synthetic realization within uncertainty
    y_sample = y_measured + np.random.randn(len(x)) * uncertainty_y

    # Create temporary dataset
    dataset_sample = OneDimensionalDataset(
        independent_variable_data=x, dependent_variable_data=y_sample
    )

    # Apply pipeline
    result_sample = pipeline.apply_to(dataset_sample, make_copy=True)

    # Store result
    mc_results[i, :] = result_sample.dependent_variable_data

# Compute statistics from Monte Carlo ensemble
y_mc_mean = np.mean(mc_results, axis=0)
y_mc_std = np.std(mc_results, axis=0)
y_mc_lower = np.percentile(mc_results, 2.5, axis=0)  # 95% interval
y_mc_upper = np.percentile(mc_results, 97.5, axis=0)

print("   ✓ Monte Carlo completed")
print(f"   ✓ Result uncertainty range: [{y_mc_std.min():.4f}, {y_mc_std.max():.4f}]")

# =============================================================================
# Section 3: Apply Pipeline to Nominal Dataset
# =============================================================================
print("\n[3] Applying pipeline to nominal dataset...")

# Process the measured data (nominal case)
result_nominal = pipeline.apply_to(dataset, make_copy=True)

print("   ✓ Nominal processing completed")
print(
    f"   ✓ Result range: [{result_nominal.dependent_variable_data.min():.3f}, "
    f"{result_nominal.dependent_variable_data.max():.3f}]"
)

# =============================================================================
# Section 4: Derivative with Uncertainty
# =============================================================================
print("\n[4] Computing derivative with uncertainty propagation...")

# Derivative amplifies noise - important for uncertainty analysis
derivative_pipeline = Pipeline(
    [GaussianSmooth(sigma=2.0), Derivative(order=1)]  # Pre-smooth to reduce noise amplification
)

print(f"   ⏳ Running Monte Carlo for derivative ({n_samples} samples)...")

# Monte Carlo for derivative
mc_derivative_results = np.zeros((n_samples, len(x)))

for i in range(n_samples):
    y_sample = y_measured + np.random.randn(len(x)) * uncertainty_y
    dataset_sample = OneDimensionalDataset(
        independent_variable_data=x, dependent_variable_data=y_sample
    )
    result_sample = derivative_pipeline.apply_to(dataset_sample, make_copy=True)
    mc_derivative_results[i, :] = result_sample.dependent_variable_data

# Statistics
dy_mc_mean = np.mean(mc_derivative_results, axis=0)
dy_mc_std = np.std(mc_derivative_results, axis=0)
dy_mc_lower = np.percentile(mc_derivative_results, 2.5, axis=0)
dy_mc_upper = np.percentile(mc_derivative_results, 97.5, axis=0)

# Nominal derivative
result_derivative_nominal = derivative_pipeline.apply_to(dataset, make_copy=True)

print("   ✓ Derivative uncertainty computed")
print(f"   ✓ Derivative uncertainty range: [{dy_mc_std.min():.4f}, {dy_mc_std.max():.4f}]")

# =============================================================================
# Section 5: Visualize Results with Uncertainty Bands
# =============================================================================
print("\n[5] Creating visualizations...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# --- Plot 1: Original data with uncertainty ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.errorbar(
    x,
    y_measured,
    yerr=uncertainty_y,
    fmt="o",
    markersize=4,
    alpha=0.6,
    capsize=2,
    label="Measured ± uncertainty",
)
ax1.plot(x, y_true, "r--", linewidth=2, label="True signal")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_title("Original Dataset\nwith Measurement Uncertainties", fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- Plot 2: Monte Carlo ensemble for smoothing ---
ax2 = fig.add_subplot(gs[0, 1])
# Plot subset of Monte Carlo realizations
for i in range(0, n_samples, 50):
    ax2.plot(x, mc_results[i, :], "b-", alpha=0.05, linewidth=1)
ax2.plot(x, y_mc_mean, "r-", linewidth=2, label="MC mean")
ax2.fill_between(x, y_mc_lower, y_mc_upper, color="red", alpha=0.2, label="95% confidence")
ax2.set_xlabel("X")
ax2.set_ylabel("Y (normalized)")
ax2.set_title("Monte Carlo Uncertainty\n(Smooth + Normalize)", fontweight="bold")
ax2.legend()
ax2.grid(True, alpha=0.3)

# --- Plot 3: Nominal vs MC mean ---
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(x, result_nominal.dependent_variable_data, "b-", linewidth=2, label="Nominal result")
ax3.plot(x, y_mc_mean, "r--", linewidth=2, label="MC mean")
ax3.fill_between(
    x, y_mc_mean - 2 * y_mc_std, y_mc_mean + 2 * y_mc_std, color="red", alpha=0.2, label="±2σ (95%)"
)
ax3.set_xlabel("X")
ax3.set_ylabel("Y (normalized)")
ax3.set_title("Nominal vs Monte Carlo\n(Pipeline Result)", fontweight="bold")
ax3.legend()
ax3.grid(True, alpha=0.3)

# --- Plot 4: Uncertainty magnitude comparison ---
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(x, uncertainty_y, "b-", linewidth=2, label="Input uncertainty")
ax4.plot(x, y_mc_std, "r-", linewidth=2, label="Output uncertainty")
ax4.set_xlabel("X")
ax4.set_ylabel("Uncertainty")
ax4.set_title("Uncertainty Transformation\n(Input → Pipeline → Output)", fontweight="bold")
ax4.legend()
ax4.grid(True, alpha=0.3)

# --- Plot 5: Derivative with uncertainty ---
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(
    result_derivative_nominal.independent_variable_data,
    result_derivative_nominal.dependent_variable_data,
    "b-",
    linewidth=2,
    label="Nominal derivative",
)
ax5.fill_between(x, dy_mc_lower, dy_mc_upper, color="red", alpha=0.2, label="95% confidence")
ax5.axhline(0, color="gray", linestyle="--", linewidth=1)
ax5.set_xlabel("X")
ax5.set_ylabel("dY/dX")
ax5.set_title("Derivative with Uncertainty\n(Noise Amplification)", fontweight="bold")
ax5.legend()
ax5.grid(True, alpha=0.3)

# --- Plot 6: Derivative uncertainty magnitude ---
ax6 = fig.add_subplot(gs[1, 2])
# Compute input uncertainty for derivative (approximate)
dx = np.diff(x).mean()
dy_input_uncertainty = uncertainty_y / dx
ax6.plot(x, dy_input_uncertainty, "b-", linewidth=2, label="Naive estimate")
ax6.plot(x, dy_mc_std, "r-", linewidth=2, label="MC uncertainty")
ax6.set_xlabel("X")
ax6.set_ylabel("Uncertainty in dY/dX")
ax6.set_title("Derivative Uncertainty\n(Smoothing Effect)", fontweight="bold")
ax6.legend()
ax6.grid(True, alpha=0.3)

fig.suptitle("Uncertainty Propagation Example", fontsize=16, fontweight="bold", y=0.995)
print("   ✓ Visualization created")

# =============================================================================
# Section 6: Summary and Key Takeaways
# =============================================================================
print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print(
    f"""
Uncertainty Propagation Results:

Input Data:
  - {len(x)} measurement points
  - Mean relative uncertainty: {(uncertainty_y.mean() / np.abs(y_measured).mean() * 100):.1f}%
  - Uncertainty type: Heteroscedastic (varies with x)

After Pipeline (Smooth + Normalize):
  - Mean output uncertainty: {y_mc_std.mean():.4f}
  - Max output uncertainty: {y_mc_std.max():.4f}
  - Smoothing reduces relative uncertainty

After Derivative:
  - Mean derivative uncertainty: {dy_mc_std.mean():.4f}
  - Pre-smoothing essential to control noise amplification
  - Without smoothing, derivative uncertainty would be ~{(uncertainty_y.mean() / dx):.2f}x larger

Key Takeaways:

1. Monte Carlo Uncertainty Propagation:
   - Generate ensemble of realizations within uncertainty
   - Apply transforms to each realization
   - Compute statistics (mean, std, percentiles) from ensemble
   - Provides full uncertainty distribution

2. Transform Effects on Uncertainty:
   - Smoothing: Reduces uncertainty (averaging effect)
   - Normalization: Transforms uncertainty scale
   - Derivative: Amplifies uncertainty (noise amplification)
   - Pipeline: Combined effects of all transforms

3. Practical Considerations:
   - Monte Carlo: {n_samples} samples sufficient for 95% intervals
   - Pre-smoothing critical before derivatives
   - Heteroscedastic uncertainties common in experiments
   - Visualization essential for understanding propagation

4. Comparison with Analytical Methods:
   - Monte Carlo: Works for any nonlinear transform
   - Analytical (linear approximation): Fast but limited
   - For complex pipelines, Monte Carlo is more reliable

5. piblin-jax Features Used:
   - Metadata for uncertainty storage (dataset.details)
   - Pipeline for consistent transform application
   - make_copy=True for immutability (safe sampling)
   - Backend abstraction for efficient array operations

6. When to Use Uncertainty Propagation:
   - Parameter estimation with confidence intervals
   - Quality control and error analysis
   - Comparing measurement techniques
   - Validating experimental procedures
   - Risk assessment in predictions

Next Steps:
- Explore bayesian_parameter_estimation.py for Bayesian uncertainty
- See transform_pipeline_example.py for advanced pipelines
- Check basic_usage_example.py for fundamental patterns
"""
)

plt.tight_layout()
plt.show()
print("\n✓ Example completed successfully!")
