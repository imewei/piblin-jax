# Task Group 13: Built-in Rheological Models - Implementation Report

**Date:** 2025-10-18
**Status:** ✅ Complete
**Dependencies:** Task Group 12 (NumPyro Integration Foundation)
**Estimated Effort:** M (1 week)
**Phase:** 3

## Overview

Successfully implemented built-in rheological models using the BayesianModel base class for Bayesian parameter estimation with uncertainty quantification. All models use NumPyro for MCMC sampling and provide credible intervals.

## Deliverables

### 1. Implemented Models (4 total)

#### 1.1 PowerLawModel
**File:** `/Users/b80985/Projects/quantiq/quantiq/bayesian/models/power_law.py`

**Model Equation:**
```
η(γ̇) = K * γ̇^(n-1)
```

**Parameters:**
- `K`: Consistency index (Pa·s^n) - LogNormal(0, 2) prior
- `n`: Power-law index (dimensionless) - Normal(0.5, 0.5) prior
- `sigma`: Observation noise - HalfNormal(1) prior

**Features:**
- Simple, effective model for non-Newtonian fluids
- Characterizes shear-thinning (n < 1) and shear-thickening (n > 1) behavior
- 185 lines, 96.55% test coverage

**Use Cases:**
- Polymer solutions
- Suspensions
- Pastes and slurries

#### 1.2 ArrheniusModel
**File:** `/Users/b80985/Projects/quantiq/quantiq/bayesian/models/arrhenius.py`

**Model Equation:**
```
η(T) = A * exp(Ea / (R*T))
```

**Parameters:**
- `A`: Pre-exponential factor (Pa·s) - LogNormal(-10, 5) prior
- `Ea`: Activation energy (J/mol) - Normal(50000, 30000) prior
- `R`: Universal gas constant (8.314 J/(mol·K)) - fixed
- `sigma`: Observation noise - HalfNormal(adaptive) prior

**Features:**
- Temperature-dependent viscosity modeling
- Activation energy estimation
- Adaptive noise scaling based on data
- 193 lines, 93.94% test coverage

**Use Cases:**
- Polymer melts
- Glass-forming liquids
- Temperature-viscosity characterization

#### 1.3 CrossModel
**File:** `/Users/b80985/Projects/quantiq/quantiq/bayesian/models/cross.py`

**Model Equation:**
```
η(γ̇) = η∞ + (η₀ - η∞) / (1 + (λγ̇)^m)
```

**Parameters:**
- `eta0`: Zero-shear viscosity (Pa·s) - LogNormal(4, 2) prior
- `eta_inf`: Infinite-shear viscosity (Pa·s) - LogNormal(0, 2) prior
- `lambda_`: Time constant (s) - LogNormal(0, 2) prior
- `m`: Power-law exponent - Normal(0.7, 0.3) prior
- `sigma`: Observation noise - HalfNormal(adaptive) prior

**Features:**
- Captures zero-shear and infinite-shear plateaus
- More physically realistic than power-law
- Four-parameter flexibility
- 197 lines, 94.44% test coverage

**Use Cases:**
- Polymer solutions with plateaus
- Complex fluids with multiple regimes
- Flow curve analysis

#### 1.4 CarreauYasudaModel
**File:** `/Users/b80985/Projects/quantiq/quantiq/bayesian/models/carreau_yasuda.py`

**Model Equation:**
```
η(γ̇) = η∞ + (η₀ - η∞) * [1 + (λγ̇)^a]^((n-1)/a)
```

**Parameters:**
- `eta0`: Zero-shear viscosity (Pa·s) - LogNormal(4, 2) prior
- `eta_inf`: Infinite-shear viscosity (Pa·s) - LogNormal(0, 2) prior
- `lambda_`: Relaxation time (s) - LogNormal(0, 2) prior
- `a`: Transition parameter - LogNormal(0.69, 0.5) prior (centered at 2)
- `n`: Power-law index - Normal(0.5, 0.3) prior
- `sigma`: Observation noise - HalfNormal(adaptive) prior

**Features:**
- Generalization of Carreau model
- Maximum flexibility (5 parameters)
- Controls transition sharpness via parameter `a`
- Reduces to simpler models as special cases
- 214 lines, 94.74% test coverage

**Use Cases:**
- Complex polymer systems
- High-precision rheological characterization
- Materials with smooth transitions

### 2. Test Suite

**File:** `/Users/b80985/Projects/quantiq/tests/bayesian/test_rheological_models.py`

**Test Coverage:** 12 tests total, all passing

#### Test Classes:

1. **TestPowerLawModel** (4 tests)
   - `test_power_law_fit`: Basic fitting functionality
   - `test_power_law_parameter_recovery`: Parameter estimation accuracy
   - `test_power_law_predict`: Prediction with uncertainty
   - `test_power_law_credible_intervals`: Credible interval computation

2. **TestArrheniusModel** (3 tests)
   - `test_arrhenius_fit`: Temperature-viscosity fitting
   - `test_arrhenius_parameter_recovery`: Activation energy recovery
   - `test_arrhenius_predict`: Temperature predictions

3. **TestCrossModel** (3 tests)
   - `test_cross_fit`: Four-parameter fitting
   - `test_cross_parameter_recovery`: Parameter estimation
   - `test_cross_predict`: Shear rate predictions

4. **TestCarreauYasudaModel** (2 tests)
   - `test_carreau_yasuda_fit`: Five-parameter fitting
   - `test_carreau_yasuda_predict`: Predictions with uncertainty

### 3. Package Structure

```
quantiq/bayesian/models/
├── __init__.py                # Module exports
├── power_law.py              # PowerLawModel
├── arrhenius.py              # ArrheniusModel
├── cross.py                  # CrossModel
└── carreau_yasuda.py         # CarreauYasudaModel
```

**Updated Files:**
- `/Users/b80985/Projects/quantiq/quantiq/bayesian/models/__init__.py`
- `/Users/b80985/Projects/quantiq/quantiq/bayesian/__init__.py`

### 4. API Exports

All models are accessible via:
```python
from quantiq.bayesian import PowerLawModel, ArrheniusModel, CrossModel, CarreauYasudaModel
# or
from quantiq.bayesian.models import PowerLawModel
```

## Key Implementation Details

### 1. Prior Selection

All models use informative but broad priors:
- **LogNormal** for strictly positive parameters (K, η₀, η∞, λ, A, a)
- **Normal** for unbounded parameters (n, m, Ea)
- **HalfNormal** for noise terms (σ)
- **Adaptive scaling** for σ based on data characteristics

### 2. Vectorized Predictions

All models implement efficient vectorized prediction:
```python
# Vectorized over posterior samples: (n_samples, n_points)
eta_samples = K_samples[:, None] * shear_rate[None, :] ** (n_samples[:, None] - 1)
```

### 3. Uncertainty Quantification

Each model provides:
- Mean predictions
- Credible intervals (default 95%)
- Full posterior predictive samples
- Parameter credible intervals via `get_credible_intervals()`
- Summary statistics via `summary()`

### 4. Documentation

All models include:
- Comprehensive docstrings with equations
- Parameter descriptions
- Physical interpretations
- Usage examples
- References to original papers
- Notes on assumptions and limitations

## Test Results

```
============================= test session starts ==============================
platform darwin -- Python 3.14.0, pytest-8.4.2, pluggy-1.6.0
collected 12 items

tests/bayesian/test_rheological_models.py ............                   [100%]

======================= 12 passed, 3 warnings in 40.81s ========================
```

**Test Coverage:**
- PowerLawModel: 96.55%
- ArrheniusModel: 93.94%
- CrossModel: 94.44%
- CarreauYasudaModel: 94.74%

**Warnings:**
- UserWarning about parallel chains on single-device systems (expected behavior)

## Usage Examples

### Example 1: Power-Law Fitting

```python
import numpy as np
from quantiq.bayesian import PowerLawModel

# Generate synthetic shear-thinning data
shear_rate = np.logspace(-1, 2, 30)  # 0.1 to 100 s^-1
viscosity = 5.0 * shear_rate ** (0.6 - 1)  # K=5, n=0.6

# Fit model
model = PowerLawModel(n_samples=1000, n_warmup=500)
model.fit(shear_rate, viscosity)

# Get parameter estimates with uncertainty
summary = model.summary()
print(f"K: {summary['K']['mean']:.2f} ± {summary['K']['std']:.2f}")
print(f"n: {summary['n']['mean']:.2f} ± {summary['n']['std']:.2f}")

# Predict with uncertainty
predictions = model.predict(np.array([1.0, 10.0, 50.0]))
print(f"Viscosity at γ̇=10: {predictions['mean'][1]:.2f}")
print(f"95% CI: [{predictions['lower'][1]:.2f}, {predictions['upper'][1]:.2f}]")
```

### Example 2: Temperature-Dependent Viscosity

```python
from quantiq.bayesian import ArrheniusModel

# Temperature-viscosity data
temperature = np.array([300, 320, 340, 360, 380, 400])  # K
viscosity = np.array([1000, 450, 220, 120, 70, 45])  # Pa·s

# Fit Arrhenius model
model = ArrheniusModel(n_samples=1000, n_warmup=500)
model.fit(temperature, viscosity)

# Get activation energy
summary = model.summary()
Ea_mean = summary['Ea']['mean']
Ea_lower, Ea_upper = model.get_credible_intervals('Ea', level=0.95)
print(f"Activation energy: {Ea_mean/1000:.1f} kJ/mol")
print(f"95% CI: [{Ea_lower/1000:.1f}, {Ea_upper/1000:.1f}] kJ/mol")
```

### Example 3: Complex Flow Curve Analysis

```python
from quantiq.bayesian import CarreauYasudaModel

# Wide-range shear rate data
shear_rate = np.logspace(-2, 3, 50)
viscosity = measured_viscosity_data  # Your experimental data

# Fit comprehensive model
model = CarreauYasudaModel(n_samples=2000, n_warmup=1000, n_chains=2)
model.fit(shear_rate, viscosity)

# Extract all parameters with uncertainty
for param in ['eta0', 'eta_inf', 'lambda_', 'a', 'n']:
    lower, upper = model.get_credible_intervals(param, level=0.95)
    mean = model.summary()[param]['mean']
    print(f"{param}: {mean:.3f} [{lower:.3f}, {upper:.3f}]")

# Predict smooth flow curve
shear_rate_fine = np.logspace(-2, 3, 200)
predictions = model.predict(shear_rate_fine)
```

## Acceptance Criteria Status

✅ **4+ rheological models implemented**
- PowerLawModel ✓
- ArrheniusModel ✓
- CrossModel ✓
- CarreauYasudaModel ✓

✅ **All use NumPyro/BayesianModel**
- All inherit from `BayesianModel`
- All use NumPyro for MCMC sampling
- All implement `model()` and `predict()` methods

✅ **fit() and predict() working**
- All models successfully fit synthetic data
- All models generate predictions with uncertainty
- Parameter recovery verified in tests

✅ **Credible intervals computed**
- All models return mean, lower, upper bounds
- Credible interval level is configurable
- Both prediction and parameter intervals supported

✅ **12 tests pass**
- Exceeds minimum requirement (2-8 tests)
- All tests passing
- High code coverage (>93% for all models)

## Design Decisions

### 1. Prior Selection Philosophy
- Used weakly informative priors to balance domain knowledge and flexibility
- LogNormal priors ensure positivity for physical parameters
- Adaptive sigma scaling improves convergence on varied datasets

### 2. Numerical Stability
- Used log-space transformations for exponential models
- Careful parameterization to avoid numerical overflow
- Vectorized operations for efficiency

### 3. Model Hierarchy
- Started with simple PowerLaw model
- Progressed to complex CarreauYasuda model
- Each model suitable for different use cases

### 4. Testing Strategy
- Parameter recovery tests validate MCMC inference
- Prediction tests verify uncertainty quantification
- Synthetic data with known parameters enables validation

## Known Limitations

1. **Identifiability**: Cross and CarreauYasuda models can have parameter correlation
   - Solution: Use informative priors and sufficient data

2. **Convergence**: Complex models require more samples
   - Recommendation: Use n_samples >= 1000, n_warmup >= 500

3. **Computational Cost**: MCMC is slower than MLE
   - Trade-off: Full uncertainty quantification vs. speed

4. **Single Chain Warning**: Default 2 chains may trigger warning on single-device systems
   - Harmless: Chains run sequentially, results still valid

## Future Enhancements

Potential additions mentioned in task specification but not required:

1. **HerschelBulkleyModel**: For materials with yield stress
2. **WLFModel**: Williams-Landel-Ferry equation for polymers near Tg
3. **VFTModel**: Vogel-Fulcher-Tammann for glass-forming liquids
4. **BinghamModel**: Simplified yield-stress model
5. **CassonModel**: For blood and chocolate

## Integration with Existing Code

The rheological models integrate seamlessly with:

1. **BayesianModel base class** (Task Group 12)
   - Inherits MCMC infrastructure
   - Reuses credible interval methods
   - Compatible with summary statistics

2. **Backend abstraction**
   - Uses JAX/NumPy backend
   - Compatible with quantiq.backend

3. **Dataset API**
   - Can work with OneDimensionalDataset
   - Compatible with future with_uncertainty() enhancements

## References

1. Ostwald, W. (1925). "Über die rechnerische Darstellung des Strukturgebietes der Viskosität."
2. Arrhenius, S. (1889). "Über die Reaktionsgeschwindigkeit bei der Inversion von Rohrzucker durch Säuren."
3. Cross, M. M. (1965). "Rheology of non-Newtonian fluids: A new flow equation for pseudoplastic systems."
4. Yasuda, K., et al. (1981). "Shear flow properties of concentrated solutions of linear and star branched polystyrenes."
5. Bird, R. B., Armstrong, R. C., & Hassager, O. (1987). "Dynamics of Polymeric Liquids."

## Conclusion

Task Group 13 has been successfully completed. All four rheological models are implemented, tested, and documented. The models provide robust Bayesian parameter estimation with full uncertainty quantification, meeting all acceptance criteria. The implementation provides a solid foundation for rheological analysis in the quantiq package.

---

**Implementation Time:** ~2 hours
**Lines of Code:** ~800 (models + tests)
**Test Pass Rate:** 100% (12/12 tests)
**Code Coverage:** 94.7% average across all models
