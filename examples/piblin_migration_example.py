"""
piblin to piblin-jax Migration Example
====================================

This example demonstrates the migration path from piblin to piblin-jax:
- API compatibility between piblin and piblin-jax
- Side-by-side code comparison
- Performance improvements with JAX backend
- New features in piblin-jax (Bayesian fitting, uncertainty propagation)

Key concepts:
- Backward compatibility with piblin patterns
- Enhanced performance through JAX acceleration
- Extended functionality (Bayesian models, advanced transforms)
- Smooth transition path for existing piblin users

Expected output: Side-by-side comparison showing equivalent operations
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from piblin_jax import fit_curve

# Import piblin-jax (successor to piblin)
from piblin_jax.data.datasets import OneDimensionalDataset
from piblin_jax.transform import Pipeline
from piblin_jax.transform.dataset import GaussianSmooth, MinMaxNormalize

print("=" * 80)
print("piblin → piblin-jax Migration Example")
print("=" * 80)

# =============================================================================
# Section 1: Dataset Creation (API Compatibility)
# =============================================================================
print("\n[1] Dataset Creation - piblin vs piblin-jax")

# Generate test data
np.random.seed(42)
x = np.linspace(0, 10, 200)
y_clean = np.sin(x) * np.exp(-x / 10)
y_noisy = y_clean + 0.1 * np.random.randn(len(x))

print("\n--- piblin style (still works in piblin-jax) ---")
print(
    """
# Legacy piblin pattern (compatible)
from piblin_jax.data.datasets import OneDimensionalDataset

dataset = OneDimensionalDataset(
    independent_variable_data=x,
    dependent_variable_data=y_noisy
)
"""
)

# Create dataset using piblin-compatible API
dataset_piblin_style = OneDimensionalDataset(
    independent_variable_data=x, dependent_variable_data=y_noisy
)
print(f"✓ piblin-style dataset created: {len(x)} points")

print("\n--- piblin-jax enhanced style (recommended) ---")
print(
    """
# piblin-jax enhanced pattern (with metadata)
from piblin_jax.data.datasets import OneDimensionalDataset

dataset = OneDimensionalDataset(
    independent_variable_data=x,
    dependent_variable_data=y_noisy,
    conditions={"temperature": 25.0, "sample": "A"},
    details={"instrument": "Rheometer", "operator": "User"}
)
"""
)

# Create dataset with piblin-jax enhancements
dataset_piblin_jax_style = OneDimensionalDataset(
    independent_variable_data=x,
    dependent_variable_data=y_noisy,
    conditions={"temperature": 25.0, "sample": "A"},
    details={"instrument": "Rheometer", "operator": "User"},
)
print("✓ piblin-jax-style dataset created with metadata")

# =============================================================================
# Section 2: Transform Application (Performance Comparison)
# =============================================================================
print("\n[2] Transform Application - Performance Improvements")

# Create transform
smooth = GaussianSmooth(sigma=2.0)

print("\n--- Applying transforms multiple times for benchmarking ---")

# Benchmark piblin-style (NumPy backend)
start_numpy = time.time()
for _ in range(100):
    result_numpy = smooth.apply_to(dataset_piblin_style, make_copy=True)
time_numpy = time.time() - start_numpy

print(f"✓ NumPy backend: 100 iterations in {time_numpy * 1000:.2f} ms")

# Note: JAX acceleration would show greater speedup with larger datasets
# or more complex transforms (JIT compilation overhead matters less)
from piblin_jax.backend import get_backend, is_jax_available

print(f"✓ JAX backend available: {is_jax_available()}")
print(f"✓ Current backend: {get_backend()}")

# =============================================================================
# Section 3: Pipeline Building (Enhanced in piblin-jax)
# =============================================================================
print("\n[3] Pipeline Building - piblin-jax Enhancement")

print("\n--- piblin style (manual chaining) ---")
print(
    """
# piblin: Manual transform chaining
smooth = GaussianSmooth(sigma=2.0)
normalize = MinMaxNormalize()

# Apply transforms sequentially
result = smooth.apply_to(dataset, make_copy=True)
result = normalize.apply_to(result, make_copy=True)
"""
)

# piblin style: manual chaining
smooth_step1 = GaussianSmooth(sigma=2.0)
normalize_step2 = MinMaxNormalize()
result_manual = smooth_step1.apply_to(dataset_piblin_style, make_copy=True)
result_manual = normalize_step2.apply_to(result_manual, make_copy=True)
print("✓ Manual chaining completed")

print("\n--- piblin-jax style (Pipeline) ---")
print(
    """
# piblin-jax: Pipeline composition (cleaner)
from piblin_jax.transform import Pipeline

pipeline = Pipeline([
    GaussianSmooth(sigma=2.0),
    MinMaxNormalize()
])

result = pipeline.apply_to(dataset, make_copy=True)
"""
)

# piblin-jax style: pipeline
pipeline = Pipeline([GaussianSmooth(sigma=2.0), MinMaxNormalize()])
result_pipeline = pipeline.apply_to(dataset_piblin_jax_style, make_copy=True)
print("✓ Pipeline application completed")

# Verify results are identical
assert np.allclose(
    result_manual.dependent_variable_data, result_pipeline.dependent_variable_data
), "Results should be identical"
print("✓ Verified: Manual chaining ≡ Pipeline (identical results)")

# =============================================================================
# Section 4: Curve Fitting (API Consistency)
# =============================================================================
print("\n[4] Curve Fitting - piblin Compatible API")

# Generate rheological test data
shear_rate = np.logspace(-1, 2, 30)
true_K, true_n = 5.0, 0.6
viscosity = true_K * shear_rate ** (true_n - 1)
viscosity += 0.05 * viscosity * np.random.randn(len(shear_rate))


# Define power law model
def power_law(x, K, n):
    """Power law viscosity model: η = K * γ^(n-1)"""
    return K * x ** (n - 1)


print("\n--- piblin/piblin-jax unified API ---")
print(
    """
# Same API in both piblin and piblin-jax
from piblin_jax import fit_curve

# Define model function
def power_law(x, K, n):
    return K * x ** (n - 1)

result = fit_curve(
    power_law,
    shear_rate,
    viscosity,
    p0=np.array([5.0, 0.5])
)

K_fit, n_fit = result['params']
"""
)

# Fit using piblin-compatible API
fit_result = fit_curve(power_law, shear_rate, viscosity, p0=np.array([5.0, 0.5]))

K_fit, n_fit = fit_result["params"]

print(f"✓ Fitted parameters: K={K_fit:.3f}, n={n_fit:.3f}")
print(f"✓ True parameters: K={true_K:.3f}, n={true_n:.3f}")
print(
    f"✓ Relative errors: K={abs(K_fit - true_K) / true_K * 100:.1f}%, n={abs(n_fit - true_n) / true_n * 100:.1f}%"
)

# =============================================================================
# Section 5: NEW in piblin-jax - Bayesian Uncertainty Quantification
# =============================================================================
print("\n[5] NEW Features in piblin-jax - Bayesian Fitting")

print("\n--- Not available in piblin ---")
print(
    """
# NEW in piblin-jax: Bayesian parameter estimation
from piblin_jax.bayesian.models import PowerLawModel

model = PowerLawModel(n_samples=1000, n_warmup=500, n_chains=2)
model.fit(shear_rate, viscosity)

# Get posterior summary with uncertainty
summary = model.summary()
K_mean = summary['K']['mean']
K_std = summary['K']['std']
K_credible_interval = [summary['K']['q_2.5'], summary['K']['q_97.5']]

# Make predictions with uncertainty bands
predictions = model.predict(shear_rate_pred, credible_interval=0.95)
"""
)

print("✓ Bayesian models provide full uncertainty quantification")
print("✓ Not available in piblin - major enhancement in piblin-jax")

# =============================================================================
# Section 6: Visualization - Side-by-Side Comparison
# =============================================================================
print("\n[6] Creating comparison visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("piblin → piblin-jax Migration Example", fontsize=16, fontweight="bold")

# Plot 1: Dataset creation comparison
axes[0, 0].plot(x, y_noisy, "o", markersize=3, alpha=0.5, label="Data")
axes[0, 0].plot(x, y_clean, "r--", linewidth=2, label="True signal")
axes[0, 0].set_xlabel("X")
axes[0, 0].set_ylabel("Y")
axes[0, 0].set_title("Dataset Creation\n(piblin-compatible API)", fontweight="bold")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Transform pipeline comparison
axes[0, 1].plot(x, y_noisy, "gray", alpha=0.3, linewidth=1, label="Original")
axes[0, 1].plot(
    result_pipeline.independent_variable_data,
    result_pipeline.dependent_variable_data,
    "b-",
    linewidth=2,
    label="Processed (Pipeline)",
)
axes[0, 1].set_xlabel("X")
axes[0, 1].set_ylabel("Y (normalized)")
axes[0, 1].set_title("Transform Pipeline\n(Enhanced in piblin-jax)", fontweight="bold")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Curve fitting comparison
viscosity_fit = K_fit * shear_rate ** (n_fit - 1)
axes[1, 0].loglog(shear_rate, viscosity, "ko", markersize=6, alpha=0.6, label="Data")
axes[1, 0].loglog(
    shear_rate, viscosity_fit, "b-", linewidth=2, label=f"NLSQ fit (K={K_fit:.2f}, n={n_fit:.2f})"
)
axes[1, 0].set_xlabel("Shear Rate (s⁻¹)")
axes[1, 0].set_ylabel("Viscosity (Pa·s)")
axes[1, 0].set_title("Curve Fitting\n(piblin-compatible API)", fontweight="bold")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, which="both")

# Plot 4: Feature comparison table
axes[1, 1].axis("off")
comparison_text = """
Feature Comparison: piblin vs piblin-jax

✓ = Available    ✗ = Not Available

Feature                      piblin    piblin-jax
─────────────────────────────────────────────
Dataset creation             ✓         ✓
Metadata support             Limited   ✓✓
Transform application        ✓         ✓
Transform pipelines          ✗         ✓
JAX acceleration            ✗         ✓
NLSQ curve fitting          ✓         ✓
Bayesian fitting            ✗         ✓✓
Uncertainty propagation     ✗         ✓✓
Hierarchical collections    Limited   ✓✓
Backend abstraction         ✗         ✓
Type hints (strict)         Partial   ✓✓
Modern Python 3.12+         ✗         ✓

Migration Path:
1. Replace 'import piblin' → 'import piblin_jax'
2. Existing code continues to work
3. Gradually adopt new features:
   - Add metadata to datasets
   - Use Pipeline for transforms
   - Try Bayesian fitting for uncertainty
4. Enable JAX for performance boost
"""

axes[1, 1].text(
    0.1,
    0.5,
    comparison_text,
    transform=axes[1, 1].transAxes,
    fontsize=9,
    verticalalignment="center",
    fontfamily="monospace",
    bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.3},
)

plt.tight_layout()
print("   ✓ Visualization created")

# =============================================================================
# Section 7: Summary and Migration Guide
# =============================================================================
print("\n" + "=" * 80)
print("Migration Summary")
print("=" * 80)
print(
    """
Key Points:

1. Backward Compatibility:
   - piblin-jax maintains piblin API compatibility
   - Existing piblin code works without changes
   - Replace 'import piblin' with 'import piblin_jax'

2. Performance Improvements:
   - JAX backend for automatic acceleration
   - JIT compilation for complex transforms
   - Vectorized operations throughout

3. Enhanced Features in piblin-jax:
   - Transform pipelines for cleaner code
   - Bayesian uncertainty quantification (NumPyro)
   - Comprehensive metadata support
   - Hierarchical data collections
   - Backend abstraction (JAX/NumPy)

4. Migration Strategy:
   Step 1: Replace imports (piblin → piblin-jax)
   Step 2: Run existing code (should work as-is)
   Step 3: Add metadata to datasets
   Step 4: Convert manual transform chains to Pipelines
   Step 5: Try Bayesian fitting for uncertainty-critical applications
   Step 6: Enable JAX for performance boost

5. Breaking Changes:
   - None for basic functionality
   - Some advanced piblin features may have different APIs
   - Check documentation for specific cases

Next Steps:
- Review basic_usage_example.py for piblin-jax fundamentals
- Explore bayesian_parameter_estimation.py for new Bayesian features
- See transform_pipeline_example.py for advanced pipeline usage
"""
)

plt.show()
print("\n✓ Migration example completed successfully!")
