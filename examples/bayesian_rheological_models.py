"""
Example: Bayesian Rheological Models

This example demonstrates the built-in rheological models for Bayesian
parameter estimation with uncertainty quantification.

Models demonstrated:
1. PowerLawModel - Simple shear-thinning behavior
2. ArrheniusModel - Temperature-dependent viscosity
3. CrossModel - Flow curves with plateaus
4. CarreauYasudaModel - Complex flow behavior

Each model uses NumPyro for MCMC sampling and provides:
- Parameter estimates with credible intervals
- Predictions with uncertainty
- Full posterior samples
"""

import matplotlib.pyplot as plt
import numpy as np

from quantiq.bayesian import (
    ArrheniusModel,
    CarreauYasudaModel,
    CrossModel,
    PowerLawModel,
)


def example_power_law():
    """Example 1: Power-Law Model for shear-thinning fluid."""
    print("=" * 60)
    print("Example 1: Power-Law Model")
    print("=" * 60)

    # Generate synthetic power-law data
    np.random.seed(42)
    shear_rate = np.logspace(-1, 2, 30)  # 0.1 to 100 s^-1
    true_K = 5.0  # Consistency
    true_n = 0.6  # Shear-thinning
    viscosity = true_K * shear_rate ** (true_n - 1)
    viscosity += 0.05 * viscosity * np.random.randn(len(shear_rate))  # 5% noise

    # Fit model
    print("\nFitting Power-Law Model...")
    model = PowerLawModel(n_samples=1000, n_warmup=500, n_chains=1)
    model.fit(shear_rate, viscosity)

    # Get parameter estimates
    summary = model.summary()
    print("\nParameter Estimates:")
    print(f"  K = {summary['K']['mean']:.2f} ± {summary['K']['std']:.2f}")
    print(f"  n = {summary['n']['mean']:.3f} ± {summary['n']['std']:.3f}")
    print(f"  (True values: K={true_K}, n={true_n})")

    # Get credible intervals
    K_lower, K_upper = model.get_credible_intervals("K", level=0.95)
    n_lower, n_upper = model.get_credible_intervals("n", level=0.95)
    print("\n95% Credible Intervals:")
    print(f"  K: [{K_lower:.2f}, {K_upper:.2f}]")
    print(f"  n: [{n_lower:.3f}, {n_upper:.3f}]")

    # Predict with uncertainty
    shear_rate_pred = np.logspace(-1, 2, 100)
    predictions = model.predict(shear_rate_pred, credible_interval=0.95)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.loglog(shear_rate, viscosity, "o", label="Data", alpha=0.6)
    plt.loglog(shear_rate_pred, predictions["mean"], "-", label="Mean prediction")
    plt.fill_between(
        shear_rate_pred,
        predictions["lower"],
        predictions["upper"],
        alpha=0.3,
        label="95% CI",
    )
    plt.xlabel("Shear rate (s⁻¹)")
    plt.ylabel("Viscosity (Pa·s)")
    plt.title("Power-Law Model: Shear-Thinning Fluid")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("power_law_example.png", dpi=150)
    print("\nPlot saved as: power_law_example.png")


def example_arrhenius():
    """Example 2: Arrhenius Model for temperature-dependent viscosity."""
    print("\n" + "=" * 60)
    print("Example 2: Arrhenius Model")
    print("=" * 60)

    # Generate synthetic Arrhenius data
    np.random.seed(42)
    temperature = np.linspace(280, 400, 25)  # K
    true_A = 1e-5
    true_Ea = 50000  # J/mol
    R = 8.314
    viscosity = true_A * np.exp(true_Ea / (R * temperature))
    viscosity += 0.05 * viscosity * np.random.randn(len(temperature))

    # Fit model
    print("\nFitting Arrhenius Model...")
    model = ArrheniusModel(n_samples=1000, n_warmup=500, n_chains=1)
    model.fit(temperature, viscosity)

    # Get parameter estimates
    summary = model.summary()
    print("\nParameter Estimates:")
    print(f"  A = {summary['A']['mean']:.2e} Pa·s")
    print(f"  Ea = {summary['Ea']['mean'] / 1000:.1f} kJ/mol")
    print(f"  (True values: A={true_A:.2e}, Ea={true_Ea / 1000:.1f} kJ/mol)")

    # Predict with uncertainty
    temp_pred = np.linspace(280, 400, 100)
    predictions = model.predict(temp_pred)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.semilogy(temperature, viscosity, "o", label="Data", alpha=0.6)
    plt.semilogy(temp_pred, predictions["mean"], "-", label="Mean prediction")
    plt.fill_between(
        temp_pred,
        predictions["lower"],
        predictions["upper"],
        alpha=0.3,
        label="95% CI",
    )
    plt.xlabel("Temperature (K)")
    plt.ylabel("Viscosity (Pa·s)")
    plt.title("Arrhenius Model: Temperature-Dependent Viscosity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("arrhenius_example.png", dpi=150)
    print("\nPlot saved as: arrhenius_example.png")


def example_cross():
    """Example 3: Cross Model for flow curves with plateaus."""
    print("\n" + "=" * 60)
    print("Example 3: Cross Model")
    print("=" * 60)

    # Generate synthetic Cross model data
    np.random.seed(42)
    shear_rate = np.logspace(-2, 3, 40)
    true_eta0 = 100.0
    true_eta_inf = 1.0
    true_lambda = 1.0
    true_m = 0.7
    viscosity = true_eta_inf + (true_eta0 - true_eta_inf) / (
        1 + (true_lambda * shear_rate) ** true_m
    )
    viscosity += 0.05 * viscosity * np.random.randn(len(shear_rate))

    # Fit model
    print("\nFitting Cross Model...")
    model = CrossModel(n_samples=1000, n_warmup=500, n_chains=1)
    model.fit(shear_rate, viscosity)

    # Get parameter estimates
    summary = model.summary()
    print("\nParameter Estimates:")
    print(f"  η₀ = {summary['eta0']['mean']:.1f} Pa·s (zero-shear)")
    print(f"  η∞ = {summary['eta_inf']['mean']:.2f} Pa·s (infinite-shear)")
    print(f"  λ = {summary['lambda_']['mean']:.2f} s")
    print(f"  m = {summary['m']['mean']:.2f}")

    # Predict with uncertainty
    shear_rate_pred = np.logspace(-2, 3, 100)
    predictions = model.predict(shear_rate_pred)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.loglog(shear_rate, viscosity, "o", label="Data", alpha=0.6)
    plt.loglog(shear_rate_pred, predictions["mean"], "-", label="Mean prediction")
    plt.fill_between(
        shear_rate_pred,
        predictions["lower"],
        predictions["upper"],
        alpha=0.3,
        label="95% CI",
    )
    plt.axhline(y=summary["eta0"]["mean"], color="r", linestyle="--", alpha=0.5, label="η₀")
    plt.axhline(y=summary["eta_inf"]["mean"], color="b", linestyle="--", alpha=0.5, label="η∞")
    plt.xlabel("Shear rate (s⁻¹)")
    plt.ylabel("Viscosity (Pa·s)")
    plt.title("Cross Model: Flow Curve with Plateaus")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("cross_example.png", dpi=150)
    print("\nPlot saved as: cross_example.png")


def example_carreau_yasuda():
    """Example 4: Carreau-Yasuda Model for complex flow behavior."""
    print("\n" + "=" * 60)
    print("Example 4: Carreau-Yasuda Model")
    print("=" * 60)

    # Generate synthetic Carreau-Yasuda data
    np.random.seed(42)
    shear_rate = np.logspace(-2, 3, 50)
    true_eta0 = 100.0
    true_eta_inf = 1.0
    true_lambda = 1.0
    true_a = 2.0
    true_n = 0.5
    viscosity = true_eta_inf + (true_eta0 - true_eta_inf) * (
        1 + (true_lambda * shear_rate) ** true_a
    ) ** ((true_n - 1) / true_a)
    viscosity += 0.05 * viscosity * np.random.randn(len(shear_rate))

    # Fit model
    print("\nFitting Carreau-Yasuda Model...")
    model = CarreauYasudaModel(n_samples=1000, n_warmup=500, n_chains=1)
    model.fit(shear_rate, viscosity)

    # Get parameter estimates
    summary = model.summary()
    print("\nParameter Estimates:")
    print(f"  η₀ = {summary['eta0']['mean']:.1f} Pa·s")
    print(f"  η∞ = {summary['eta_inf']['mean']:.2f} Pa·s")
    print(f"  λ = {summary['lambda_']['mean']:.2f} s")
    print(f"  a = {summary['a']['mean']:.2f}")
    print(f"  n = {summary['n']['mean']:.3f}")

    # Predict with uncertainty
    shear_rate_pred = np.logspace(-2, 3, 100)
    predictions = model.predict(shear_rate_pred)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.loglog(shear_rate, viscosity, "o", label="Data", alpha=0.6)
    plt.loglog(shear_rate_pred, predictions["mean"], "-", label="Mean prediction")
    plt.fill_between(
        shear_rate_pred,
        predictions["lower"],
        predictions["upper"],
        alpha=0.3,
        label="95% CI",
    )
    plt.xlabel("Shear rate (s⁻¹)")
    plt.ylabel("Viscosity (Pa·s)")
    plt.title("Carreau-Yasuda Model: Complex Flow Behavior")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("carreau_yasuda_example.png", dpi=150)
    print("\nPlot saved as: carreau_yasuda_example.png")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Bayesian Rheological Models Examples")
    print("=" * 60)
    print("\nThis script demonstrates 4 built-in rheological models:")
    print("1. PowerLawModel - Simple shear-thinning")
    print("2. ArrheniusModel - Temperature dependence")
    print("3. CrossModel - Flow curves with plateaus")
    print("4. CarreauYasudaModel - Complex behavior")
    print("\nEach model provides:")
    print("  - Bayesian parameter estimation")
    print("  - Uncertainty quantification")
    print("  - Credible intervals")
    print("  - Posterior samples")
    print("=" * 60)

    # Run all examples
    example_power_law()
    example_arrhenius()
    example_cross()
    example_carreau_yasuda()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
