"""
Bayesian Parameter Estimation Example
======================================

This example demonstrates piblin-jax's Bayesian uncertainty quantification capabilities:
- Fitting rheological models (Power-Law) to experimental data
- Comparing Bayesian (NumPyro) with classical NLSQ estimates
- Extracting posterior distributions with credible intervals
- Visualizing parameter uncertainty
- Making predictions with uncertainty bands

Key concepts:
- PowerLawModel for Bayesian fitting: η = K * γ̇^(n-1)
- MCMC sampling with NumPyro
- Credible intervals vs. confidence intervals
- Posterior predictive distributions

Expected output: Plots showing fitted model with uncertainty quantification
"""

import matplotlib.pyplot as plt
import numpy as np

from piblin_jax import fit_curve

# Import piblin-jax Bayesian and fitting modules
from piblin_jax.bayesian.models import PowerLawModel

print("=" * 80)
print("Bayesian Parameter Estimation Example")
print("=" * 80)

# =============================================================================
# Section 1: Generate Synthetic Rheological Data
# =============================================================================
print("\n[1] Generating synthetic power-law rheological data...")

# Set random seed for reproducibility
np.random.seed(42)

# True model parameters
true_K = 5.0  # Consistency index (Pa·s^n)
true_n = 0.6  # Power-law index (< 1 for shear-thinning)

# Generate shear rate data (logarithmically spaced)
shear_rate = np.logspace(-1, 2, 30)  # 0.1 to 100 s^-1

# True viscosity: η = K * γ̇^(n-1)
viscosity_true = true_K * shear_rate ** (true_n - 1)

# Add realistic measurement noise (5% relative error)
noise_level = 0.05
noise = noise_level * viscosity_true * np.random.randn(len(shear_rate))
viscosity_measured = viscosity_true + noise

print(f"   ✓ Generated {len(shear_rate)} data points")
print(f"   ✓ True parameters: K={true_K}, n={true_n}")
print(f"   ✓ Noise level: {noise_level * 100:.1f}% relative error")
print(f"   ✓ Shear rate range: {shear_rate.min():.2f} to {shear_rate.max():.2f} s⁻¹")
print(
    f"   ✓ Viscosity range: {viscosity_measured.min():.3f} to {viscosity_measured.max():.3f} Pa·s"
)

# =============================================================================
# Section 2: Classical NLSQ Fitting (Quick Estimate)
# =============================================================================
print("\n[2] Fitting with classical NLSQ (fast, no uncertainty)...")


# Define power law model
def power_law(x, K, n):
    """Power law viscosity model: η = K * γ^(n-1)"""
    return K * x ** (n - 1)


# Use piblin-jax's NLSQ fitter for quick parameter estimates
nlsq_result = fit_curve(power_law, shear_rate, viscosity_measured, p0=np.array([5.0, 0.5]))

# Extract parameter estimates
nlsq_K, nlsq_n = nlsq_result["params"]

# Compute fitted curve
viscosity_nlsq = nlsq_K * shear_rate ** (nlsq_n - 1)

print("   ✓ NLSQ fitting completed")
print(f"   ✓ Estimated K = {nlsq_K:.3f} (true: {true_K})")
print(f"   ✓ Estimated n = {nlsq_n:.3f} (true: {true_n})")
print(f"   ✓ Residual sum of squares: {np.sum(nlsq_result['residuals'] ** 2):.2e}")

# =============================================================================
# Section 3: Bayesian Fitting with Uncertainty Quantification
# =============================================================================
print("\n[3] Fitting with Bayesian MCMC (with uncertainty)...")
print("   ⏳ Running MCMC sampler (this may take 10-30 seconds)...")

# Create Bayesian power-law model
model = PowerLawModel(
    n_samples=2000,  # Number of posterior samples
    n_warmup=1000,  # Number of warmup/burn-in samples
    n_chains=2,  # Number of independent MCMC chains
    random_seed=42,
)

# Fit the model using MCMC
model.fit(shear_rate, viscosity_measured)

print("   ✓ MCMC sampling completed")
print(f"   ✓ Collected {model.n_samples * model.n_chains} posterior samples")

# Get parameter summary statistics
summary = model.summary()

print("\n   Posterior Summary:")
print(f"   K: mean={summary['K']['mean']:.3f}, std={summary['K']['std']:.3f}")
print(f"      95% CI: [{summary['K']['q_2.5']:.3f}, {summary['K']['q_97.5']:.3f}]")
print(f"   n: mean={summary['n']['mean']:.3f}, std={summary['n']['std']:.3f}")
print(f"      95% CI: [{summary['n']['q_2.5']:.3f}, {summary['n']['q_97.5']:.3f}]")

# =============================================================================
# Section 4: Make Predictions with Uncertainty
# =============================================================================
print("\n[4] Making predictions with uncertainty...")

# Generate fine grid for smooth predictions
shear_rate_pred = np.logspace(-1, 2, 100)

# Get Bayesian predictions with 95% credible intervals
predictions = model.predict(shear_rate_pred, credible_interval=0.95)

print(f"   ✓ Generated {len(shear_rate_pred)} predictions")
print("   ✓ Each prediction includes mean, lower/upper credible bounds")

# =============================================================================
# Section 5: Visualize Results
# =============================================================================
print("\n[5] Creating visualizations...")

# Create comprehensive figure
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# --- Plot 1: Data and Fits ---
ax1 = fig.add_subplot(gs[0, :2])
ax1.loglog(shear_rate, viscosity_measured, "ko", markersize=8, label="Measured data", alpha=0.6)
ax1.loglog(shear_rate, viscosity_true, "gray", linestyle="--", linewidth=2, label="True model")
ax1.loglog(
    shear_rate,
    viscosity_nlsq,
    "b-",
    linewidth=2,
    label=f"NLSQ fit (K={nlsq_K:.2f}, n={nlsq_n:.2f})",
)
ax1.loglog(
    shear_rate_pred,
    predictions["mean"],
    "r-",
    linewidth=2,
    label=f"Bayesian mean (K={summary['K']['mean']:.2f}, n={summary['n']['mean']:.2f})",
)
ax1.fill_between(
    shear_rate_pred,
    predictions["lower"],
    predictions["upper"],
    color="red",
    alpha=0.2,
    label="95% Credible interval",
)
ax1.set_xlabel("Shear Rate (s⁻¹)", fontsize=12)
ax1.set_ylabel("Viscosity (Pa·s)", fontsize=12)
ax1.set_title("Power-Law Fitting: NLSQ vs. Bayesian", fontsize=14, fontweight="bold")
ax1.legend(fontsize=10, loc="best")
ax1.grid(True, alpha=0.3, which="both")

# --- Plot 2: Parameter K Posterior ---
ax2 = fig.add_subplot(gs[0, 2])
K_samples = model.samples["K"]
ax2.hist(K_samples, bins=50, density=True, alpha=0.7, color="blue", edgecolor="black")
ax2.axvline(true_K, color="green", linestyle="--", linewidth=2, label="True value")
ax2.axvline(summary["K"]["mean"], color="red", linestyle="-", linewidth=2, label="Posterior mean")
ax2.axvline(summary["K"]["q_2.5"], color="red", linestyle=":", linewidth=1.5, label="95% CI")
ax2.axvline(summary["K"]["q_97.5"], color="red", linestyle=":", linewidth=1.5)
ax2.set_xlabel("Consistency Index K (Pa·s^n)", fontsize=11)
ax2.set_ylabel("Probability Density", fontsize=11)
ax2.set_title("Posterior: K", fontsize=12, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# --- Plot 3: Parameter n Posterior ---
ax3 = fig.add_subplot(gs[1, 0])
n_samples = model.samples["n"]
ax3.hist(n_samples, bins=50, density=True, alpha=0.7, color="orange", edgecolor="black")
ax3.axvline(true_n, color="green", linestyle="--", linewidth=2, label="True value")
ax3.axvline(summary["n"]["mean"], color="red", linestyle="-", linewidth=2, label="Posterior mean")
ax3.axvline(summary["n"]["q_2.5"], color="red", linestyle=":", linewidth=1.5, label="95% CI")
ax3.axvline(summary["n"]["q_97.5"], color="red", linestyle=":", linewidth=1.5)
ax3.set_xlabel("Power-Law Index n", fontsize=11)
ax3.set_ylabel("Probability Density", fontsize=11)
ax3.set_title("Posterior: n", fontsize=12, fontweight="bold")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# --- Plot 4: Joint Posterior (K vs n) ---
ax4 = fig.add_subplot(gs[1, 1])
hist = ax4.hist2d(K_samples, n_samples, bins=50, cmap="Blues", density=True)
ax4.plot(
    true_K,
    true_n,
    "go",
    markersize=12,
    markeredgecolor="black",
    markeredgewidth=2,
    label="True values",
)
ax4.plot(
    summary["K"]["mean"],
    summary["n"]["mean"],
    "r*",
    markersize=15,
    markeredgecolor="black",
    markeredgewidth=1.5,
    label="Posterior mean",
)
ax4.set_xlabel("Consistency Index K", fontsize=11)
ax4.set_ylabel("Power-Law Index n", fontsize=11)
ax4.set_title("Joint Posterior: K vs n", fontsize=12, fontweight="bold")
ax4.legend(fontsize=9)
plt.colorbar(hist[3], ax=ax4, label="Probability Density")

# --- Plot 5: Residuals ---
ax5 = fig.add_subplot(gs[1, 2])
viscosity_pred_mean = predictions["mean"]
# Interpolate predictions to data points for residuals
from scipy.interpolate import interp1d

pred_interp = interp1d(
    shear_rate_pred, viscosity_pred_mean, kind="linear", fill_value="extrapolate"
)
viscosity_bayesian_at_data = pred_interp(shear_rate)
residuals = viscosity_measured - viscosity_bayesian_at_data

ax5.semilogx(shear_rate, residuals, "ko", markersize=6, alpha=0.6)
ax5.axhline(0, color="red", linestyle="--", linewidth=1.5)
ax5.axhline(
    2 * noise_level * viscosity_measured.mean(),
    color="gray",
    linestyle=":",
    linewidth=1,
    label="±2σ expected",
)
ax5.axhline(-2 * noise_level * viscosity_measured.mean(), color="gray", linestyle=":", linewidth=1)
ax5.set_xlabel("Shear Rate (s⁻¹)", fontsize=11)
ax5.set_ylabel("Residuals (Pa·s)", fontsize=11)
ax5.set_title("Fit Residuals", fontsize=12, fontweight="bold")
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

fig.suptitle("Bayesian Parameter Estimation with piblin-jax", fontsize=16, fontweight="bold", y=0.995)
print("   ✓ Visualization created")

# =============================================================================
# Section 6: Summary and Key Takeaways
# =============================================================================
print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print(
    f"""
Parameter Recovery:
  True K = {true_K:.3f}
  NLSQ K = {nlsq_K:.3f} (point estimate, no uncertainty)
  Bayesian K = {summary["K"]["mean"]:.3f} ± {summary["K"]["std"]:.3f} (with uncertainty)

  True n = {true_n:.3f}
  NLSQ n = {nlsq_n:.3f} (point estimate, no uncertainty)
  Bayesian n = {summary["n"]["mean"]:.3f} ± {summary["n"]["std"]:.3f} (with uncertainty)

Key Takeaways:
1. NLSQ Fitting (fit_curve):
   - Fast: ~1-10 ms
   - Provides point estimates only
   - Good for initial parameter guessing
   - No uncertainty quantification

2. Bayesian Fitting (PowerLawModel):
   - Slower: ~10-30 seconds (MCMC sampling)
   - Provides full posterior distributions
   - Quantifies parameter uncertainty
   - Enables uncertainty propagation
   - Credible intervals for parameters and predictions

3. Uncertainty Information:
   - Bayesian: Credible intervals (probability that parameter is in range)
   - Classical: Confidence intervals (frequency interpretation)
   - Posterior distributions reveal correlations between parameters

4. When to Use Each:
   - Quick analysis → NLSQ (fit_curve)
   - Uncertainty critical → Bayesian (PowerLawModel)
   - Workflow: NLSQ first, then Bayesian for final analysis

Next Steps:
- Try other models: ArrheniusModel, CrossModel, CarreauYasudaModel
- Explore uncertainty propagation: uncertainty_propagation_example.py
- Learn about transform pipelines: transform_pipeline_example.py
"""
)

plt.tight_layout()
plt.show()
print("\n✓ Example completed successfully!")
