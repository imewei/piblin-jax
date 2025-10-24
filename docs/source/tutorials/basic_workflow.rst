Basic Workflow Tutorial
=======================

This tutorial demonstrates a complete data analysis workflow using piblin-jax,
from data loading through transformation to visualization and analysis.

Overview
--------

We'll walk through a typical rheology experiment workflow:

1. Load experimental data
2. Inspect and visualize raw data
3. Apply smoothing to reduce noise
4. Normalize and interpolate data
5. Extract regions of interest
6. Perform statistical analysis
7. Generate publication-quality plots

This tutorial assumes you have piblin-jax installed. See :doc:`../user_guide/installation`
if you need to install it first.

Step 1: Loading Data
---------------------

Let's start by loading some experimental rheology data. We'll create synthetic
data for this tutorial, but in practice you'd load from a file.

Creating Sample Data
^^^^^^^^^^^^^^^^^^^^

::

    import numpy as np
    import matplotlib.pyplot as plt
    from piblin_jax.data import OneDimensionalDataset

    # Generate synthetic flow curve data
    # (shear rate vs viscosity for a shear-thinning fluid)
    np.random.seed(42)

    # Shear rate from 0.1 to 100 s^-1
    shear_rate = np.logspace(-1, 2, 50)

    # Power-law fluid: eta = K * gamma_dot^(n-1)
    K = 5.0  # Consistency index
    n = 0.6  # Flow behavior index (< 1 = shear-thinning)

    # True viscosity with added noise
    viscosity_true = K * shear_rate**(n - 1)
    noise = 0.05 * viscosity_true * np.random.randn(len(shear_rate))
    viscosity = viscosity_true + noise

    # Create dataset
    dataset = OneDimensionalDataset(
        x=shear_rate,
        y=viscosity,
        x_label='Shear Rate (1/s)',
        y_label='Viscosity (Pa.s)',
        name='Flow Curve'
    )

    print(f"Dataset: {dataset.name}")
    print(f"Points: {len(dataset.x)}")
    print(f"X range: [{dataset.x.min():.2f}, {dataset.x.max():.2f}]")
    print(f"Y range: [{dataset.y.min():.2f}, {dataset.y.max():.2f}]")

Loading from File
^^^^^^^^^^^^^^^^^

In real applications, you'd load data from files::

    import piblin_jax

    # Load CSV file
    dataset = piblin_jax.read_file('flow_curve.csv')

    # Or use specific reader
    from piblin_jax.dataio import CSVReader

    reader = CSVReader(x_column=0, y_column=1)
    dataset = reader.read('flow_curve.csv')

Step 2: Initial Visualization
------------------------------

Always visualize your raw data first to understand its characteristics::

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot on log-log scale (common for rheology)
    ax.loglog(dataset.x, dataset.y, 'o', alpha=0.6, label='Raw Data')
    ax.set_xlabel(dataset.x_label)
    ax.set_ylabel(dataset.y_label)
    ax.set_title(f'{dataset.name} - Raw Data')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()

Key observations from the plot:

- Data shows power-law behavior (linear on log-log plot)
- Some scatter due to measurement noise
- No obvious outliers
- Good coverage of shear rate range

Step 3: Data Smoothing
----------------------

Apply Gaussian smoothing to reduce noise while preserving trends::

    from piblin_jax.transform import GaussianSmoothing

    # Create smoothing transform
    # sigma controls smoothness (higher = more smooth)
    smoother = GaussianSmoothing(sigma=1.5)

    # Apply to dataset
    smoothed = smoother.apply_to(dataset)

    print(f"Original dataset: {len(dataset.x)} points")
    print(f"Smoothed dataset: {len(smoothed.x)} points")

Compare raw and smoothed data::

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.loglog(dataset.x, dataset.y, 'o', alpha=0.4, label='Raw Data')
    ax.loglog(smoothed.x, smoothed.y, '-', linewidth=2, label='Smoothed')
    ax.set_xlabel(dataset.x_label)
    ax.set_ylabel(dataset.y_label)
    ax.set_title('Smoothing Effect')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()

Step 4: Interpolation
---------------------

Interpolate to a regular grid for easier analysis::

    from piblin_jax.transform import Interpolate1D

    # Create regular grid on log scale
    new_shear_rate = np.logspace(-1, 2, 100)

    # Interpolate
    interpolator = Interpolate1D(
        new_x=new_shear_rate,
        kind='cubic'  # Use cubic interpolation
    )

    interpolated = interpolator.apply_to(smoothed)

    print(f"Interpolated to {len(interpolated.x)} points")

Step 5: Building a Pipeline
----------------------------

Combine multiple transforms into a reusable pipeline::

    from piblin_jax.transform import Pipeline

    # Create pipeline: smooth  ->  interpolate
    pipeline = Pipeline([
        GaussianSmoothing(sigma=1.5),
        Interpolate1D(new_x=new_shear_rate, kind='cubic')
    ])

    # Apply pipeline
    processed = pipeline.apply_to(dataset)

    # Visualize result
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.loglog(dataset.x, dataset.y, 'o', alpha=0.4, label='Raw')
    ax.loglog(processed.x, processed.y, '-', linewidth=2, label='Processed')
    ax.set_xlabel(dataset.x_label)
    ax.set_ylabel(dataset.y_label)
    ax.set_title('Pipeline Result')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()

Pipelines are reusable - apply to multiple datasets::

    dataset1 = piblin_jax.read_file('sample1.csv')
    dataset2 = piblin_jax.read_file('sample2.csv')

    result1 = pipeline.apply_to(dataset1)
    result2 = pipeline.apply_to(dataset2)

Step 6: Region of Interest
---------------------------

Extract and analyze specific regions::

    from piblin_jax.transform import SelectRegion

    # Extract low shear rate region (gamma_dot < 10 s^-1)
    low_shear_selector = SelectRegion(x_min=0.1, x_max=10.0)
    low_shear = low_shear_selector.apply_to(processed)

    # Extract high shear rate region (gamma_dot > 10 s^-1)
    high_shear_selector = SelectRegion(x_min=10.0, x_max=100.0)
    high_shear = high_shear_selector.apply_to(processed)

    # Visualize regions
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    # Full range
    ax1.loglog(processed.x, processed.y, '-', linewidth=2)
    ax1.set_xlabel(dataset.x_label)
    ax1.set_ylabel(dataset.y_label)
    ax1.set_title('Full Range')
    ax1.grid(True, alpha=0.3)

    # Low shear
    ax2.loglog(low_shear.x, low_shear.y, '-', linewidth=2, color='orange')
    ax2.set_xlabel(dataset.x_label)
    ax2.set_ylabel(dataset.y_label)
    ax2.set_title('Low Shear Rate')
    ax2.grid(True, alpha=0.3)

    # High shear
    ax3.loglog(high_shear.x, high_shear.y, '-', linewidth=2, color='green')
    ax3.set_xlabel(dataset.x_label)
    ax3.set_ylabel(dataset.y_label)
    ax3.set_title('High Shear Rate')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

Step 7: Numerical Derivatives
------------------------------

Calculate shear stress from viscosity and shear rate::

    from piblin_jax.transform import Derivative

    # Shear stress tau = eta * gamma_dot
    # In log-log space, this is addition: log(tau) = log(eta) + log(gamma_dot)

    # For direct calculation, use element-wise operations
    log_shear_rate = np.log10(processed.x)
    log_viscosity = np.log10(processed.y)
    log_shear_stress = log_viscosity + log_shear_rate

    # Create shear stress dataset
    from piblin_jax.data import OneDimensionalDataset

    stress_dataset = OneDimensionalDataset(
        x=processed.x,
        y=10**log_shear_stress,
        x_label='Shear Rate (1/s)',
        y_label='Shear Stress (Pa)',
        name='Shear Stress Curve'
    )

    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.loglog(processed.x, processed.y, '-', linewidth=2)
    ax1.set_xlabel('Shear Rate (1/s)')
    ax1.set_ylabel('Viscosity (Pa.s)')
    ax1.set_title('Flow Curve')
    ax1.grid(True, alpha=0.3)

    ax2.loglog(stress_dataset.x, stress_dataset.y, '-', linewidth=2, color='red')
    ax2.set_xlabel('Shear Rate (1/s)')
    ax2.set_ylabel('Shear Stress (Pa)')
    ax2.set_title('Stress Curve')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

Step 8: Statistical Analysis
-----------------------------

Perform statistical analysis on processed data::

    # Calculate statistics
    mean_viscosity = np.mean(processed.y)
    std_viscosity = np.std(processed.y)
    min_viscosity = np.min(processed.y)
    max_viscosity = np.max(processed.y)

    print("\\nViscosity Statistics:")
    print(f"  Mean: {mean_viscosity:.2f} Pa.s")
    print(f"  Std Dev: {std_viscosity:.2f} Pa.s")
    print(f"  Range: [{min_viscosity:.2f}, {max_viscosity:.2f}] Pa.s")

    # Power-law parameters from log-log slope
    log_x = np.log10(processed.x)
    log_y = np.log10(processed.y)

    # Linear fit in log-log space
    coeffs = np.polyfit(log_x, log_y, 1)
    slope = coeffs[0]
    intercept = coeffs[1]

    n_fitted = slope + 1  # Power-law index
    K_fitted = 10**intercept  # Consistency

    print("\\nPower-Law Fit (eta = K*gamma_dot^(n-1)):")
    print(f"  K (consistency): {K_fitted:.2f} Pa.s^n")
    print(f"  n (flow index): {n_fitted:.2f}")
    print(f"  True values: K={K:.2f}, n={n:.2f}")

Step 9: Publication-Quality Plot
---------------------------------

Create a polished figure for publication::

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Main plot: Flow curve
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.loglog(dataset.x, dataset.y, 'o', alpha=0.3,
                   markersize=6, label='Raw Data')
    ax_main.loglog(processed.x, processed.y, '-', linewidth=2.5,
                   color='darkblue', label='Smoothed & Interpolated')

    # Add power-law fit
    y_fit = K_fitted * processed.x**(n_fitted - 1)
    ax_main.loglog(processed.x, y_fit, '--', linewidth=2,
                   color='red', alpha=0.7,
                   label=f'Power-Law Fit (n={n_fitted:.2f})')

    ax_main.set_xlabel('Shear Rate, $\\dot{\\gamma}$ (s$^{-1}$)', fontsize=12)
    ax_main.set_ylabel('Viscosity, $\\eta$ (Pa.s)', fontsize=12)
    ax_main.set_title('Rheological Flow Curve', fontsize=14, fontweight='bold')
    ax_main.grid(True, alpha=0.3, which='both')
    ax_main.legend(fontsize=10, framealpha=0.9)

    # Bottom left: Residuals
    ax_resid = fig.add_subplot(gs[1, 0])
    residuals = (processed.y - y_fit) / y_fit * 100  # Percent error
    ax_resid.semilogx(processed.x, residuals, 'o-', markersize=4, alpha=0.7)
    ax_resid.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax_resid.set_xlabel('Shear Rate (s$^{-1}$)', fontsize=10)
    ax_resid.set_ylabel('Residual (%)', fontsize=10)
    ax_resid.set_title('Fit Residuals', fontsize=11)
    ax_resid.grid(True, alpha=0.3)

    # Bottom right: Statistics
    ax_stats = fig.add_subplot(gs[1, 1])
    ax_stats.axis('off')

    stats_text = f"""
    Dataset Statistics
                      
    Data Points: {len(processed.x)}

    Shear Rate Range:
      {processed.x.min():.2f} - {processed.x.max():.2f} s{^-1

    Viscosity Range:
      {processed.y.min():.2f} - {processed.y.max():.2f} Pa.s

    Power-Law Parameters:
      K = {K_fitted:.2f} Pa.s
      n = {n_fitted:.2f}

    Shear-Thinning Index:
      {((1-n_fitted)*100):.0f}% (n < 1)
    """

    ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes,
                  fontsize=9, verticalalignment='top',
                  fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('Rheological Analysis with piblin-jax', fontsize=15,
                 fontweight='bold', y=0.98)

    # Save figure
    plt.savefig('rheology_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\\nFigure saved as 'rheology_analysis.png'")

Step 10: Working with Multiple Samples
---------------------------------------

Analyze multiple samples using measurement sets::

    from piblin_jax.data.collections import MeasurementSet

    # Create multiple datasets (e.g., different temperatures)
    temperatures = [20, 40, 60]  #  degC
    datasets = {}

    for temp in temperatures:
        # Generate data with temperature-dependent viscosity
        # (Arrhenius behavior)
        viscosity_temp = viscosity * np.exp(0.02 * (temp - 20))
        noise_temp = 0.05 * viscosity_temp * np.random.randn(len(shear_rate))

        datasets[temp] = OneDimensionalDataset(
            x=shear_rate,
            y=viscosity_temp + noise_temp,
            x_label='Shear Rate (1/s)',
            y_label='Viscosity (Pa.s)',
            name=f'Flow Curve @ {temp} degC'
        )

    # Apply same pipeline to all datasets
    processed_datasets = {}
    for temp, ds in datasets.items():
        processed_datasets[temp] = pipeline.apply_to(ds)

    # Visualize all temperatures
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.coolwarm(np.linspace(0, 1, len(temperatures)))

    for i, (temp, ds) in enumerate(processed_datasets.items()):
        ax.loglog(ds.x, ds.y, '-', linewidth=2,
                 color=colors[i], label=f'{temp} degC')

    ax.set_xlabel('Shear Rate (s$^{-1}$)', fontsize=12)
    ax.set_ylabel('Viscosity (Pa.s)', fontsize=12)
    ax.set_title('Temperature-Dependent Flow Curves', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10, title='Temperature')

    plt.tight_layout()
    plt.show()

Summary
-------

In this tutorial, we've covered a complete workflow:

1.  Data loading (synthetic and from files)
2.  Initial visualization and inspection
3.  Data smoothing and noise reduction
4.  Interpolation to regular grids
5.  Building reusable transform pipelines
6.  Region selection and analysis
7.  Derivative calculations
8.  Statistical analysis and model fitting
9.  Publication-quality visualization
10.  Multi-sample analysis

Next Steps
----------

- **Bayesian Analysis**: See :doc:`uncertainty_quantification` for advanced
  parameter estimation with uncertainty
- **Custom Transforms**: Learn to create your own transforms in
  :doc:`custom_transforms`
- **Rheological Models**: Explore built-in models in :doc:`rheological_models`
- **Performance**: Optimize for large datasets in :doc:`../user_guide/performance`

Complete Code
-------------

The complete code for this tutorial is available in the
``examples/`` directory as ``basic_workflow.py``.

To run it::

    python examples/basic_workflow.py
