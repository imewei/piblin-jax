"""
Baseline subtraction transforms for 1D datasets.

This module provides baseline correction transforms to remove systematic
offsets and drifts from spectroscopic and chromatographic data.
"""

import numpy as np

from piblin_jax.backend import jnp
from piblin_jax.data.datasets import OneDimensionalDataset
from piblin_jax.transform.base import DatasetTransform


class PolynomialBaseline(DatasetTransform):
    """
    Fit and subtract polynomial baseline from data.

    This transform fits a polynomial to the data and subtracts it,
    removing systematic trends and offsets. Commonly used in spectroscopy
    to remove background signals.

    Parameters
    ----------
    degree : int, default=1
        Degree of polynomial to fit.
        - 0: Constant offset
        - 1: Linear drift
        - 2: Quadratic curvature
        - Higher: More complex baselines

    Attributes
    ----------
    degree : int
        Polynomial degree for baseline fitting.

    Examples
    --------
    >>> import numpy as np
    >>> from piblin_jax.data.datasets import OneDimensionalDataset
    >>> from piblin_jax.transform.dataset import PolynomialBaseline
    >>>
    >>> # Create data with linear drift
    >>> x = np.linspace(0, 10, 100)
    >>> signal = np.sin(x)
    >>> baseline = 2.0 * x + 5.0  # Linear drift
    >>> y = signal + baseline
    >>> dataset = OneDimensionalDataset(
    ...     independent_variable_data=x,
    ...     dependent_variable_data=y
    ... )
    >>>
    >>> # Remove linear baseline
    >>> transform = PolynomialBaseline(degree=1)
    >>> result = transform.apply_to(dataset)
    >>>
    >>> # Result should be close to original signal
    >>> np.allclose(result.dependent_variable_data, signal, atol=0.1)
    True

    Notes
    -----
    - Uses least-squares polynomial fitting (np.polyfit)
    - Works for both JAX and NumPy backends (uses NumPy for fitting)
    - Higher degree polynomials may overfit to noise
    - For complex baselines, consider spline-based methods
    - Independent variable (x) is preserved
    """

    def __init__(self, degree: int = 1):
        """
        Initialize polynomial baseline transform.

        Parameters
        ----------
        degree : int, default=1
            Polynomial degree (0 = constant, 1 = linear, etc.).
        """
        super().__init__()
        self.degree = degree

    def _apply(self, dataset: OneDimensionalDataset) -> OneDimensionalDataset:  # type: ignore[override]
        """
        Fit and subtract polynomial baseline from dataset.

        Parameters
        ----------
        dataset : OneDimensionalDataset
            Input dataset with baseline.

        Returns
        -------
        OneDimensionalDataset
            Dataset with baseline subtracted.

        Notes
        -----
        The polynomial is fit to the entire data range. For region-based
        baseline fitting, use RegionTransform composition.
        """
        x = dataset.independent_variable_data
        y = dataset.dependent_variable_data

        # Convert to NumPy for polynomial fitting
        # (polyfit not available in JAX, use NumPy for both backends)
        x_np = np.asarray(x)
        y_np = np.asarray(y)

        # Fit polynomial to data
        coeffs = np.polyfit(x_np, y_np, self.degree)

        # Evaluate polynomial at x values
        baseline = np.polyval(coeffs, x_np)

        # Subtract baseline
        y_corrected = y_np - baseline

        # Convert back to backend array
        dataset._dependent_variable_data = jnp.asarray(y_corrected)

        return dataset


class AsymmetricLeastSquaresBaseline(DatasetTransform):
    """
    Fit and subtract baseline using Asymmetric Least Squares (ALS) method.

    This advanced baseline correction method is particularly effective for
    data with positive peaks on a varying baseline (e.g., spectra, chromatograms).
    The ALS method penalizes positive residuals more than negative ones,
    causing the baseline to fit below peaks.

    Parameters
    ----------
    lambda_ : float, default=1e5
        Smoothness parameter. Larger values = smoother baseline.
    p : float, default=0.01
        Asymmetry parameter (0 < p < 1).
        Smaller values = baseline hugs peaks more closely from below.
    max_iter : int, default=10
        Maximum number of iterations for convergence.

    Examples
    --------
    >>> import numpy as np
    >>> from piblin_jax.data.datasets import OneDimensionalDataset
    >>> from piblin_jax.transform.dataset import AsymmetricLeastSquaresBaseline
    >>>
    >>> # Create spectrum with peaks on curved baseline
    >>> x = np.linspace(0, 100, 1000)
    >>> baseline = 10 + 0.1 * x + 0.001 * x**2
    >>> peaks = 50 * np.exp(-((x - 30)**2) / 20)
    >>> peaks += 30 * np.exp(-((x - 70)**2) / 15)
    >>> y = baseline + peaks
    >>> dataset = OneDimensionalDataset(x, y)
    >>>
    >>> # Remove baseline using ALS
    >>> transform = AsymmetricLeastSquaresBaseline(lambda_=1e6, p=0.01)
    >>> result = transform.apply_to(dataset)

    Notes
    -----
    - ALS is iterative and may be slower than polynomial baseline
    - Very effective for spectroscopy and chromatography data
    - lambda controls smoothness (typical: 1e4 to 1e7)
    - p controls asymmetry (typical: 0.001 to 0.1)
    - Based on Eilers & Boelens (2005) paper

    References
    ----------
    P. H. C. Eilers and H. F. M. Boelens,
    "Baseline Correction with Asymmetric Least Squares Smoothing",
    Leiden University Medical Centre Report, 2005.
    """

    def __init__(self, lambda_: float = 1e5, p: float = 0.01, max_iter: int = 10):
        """
        Initialize ALS baseline transform.

        Parameters
        ----------
        lambda_ : float, default=1e5
            Smoothness parameter.
        p : float, default=0.01
            Asymmetry parameter.
        max_iter : int, default=10
            Maximum iterations.
        """
        super().__init__()
        self.lambda_ = lambda_
        self.p = p
        self.max_iter = max_iter

    def _apply(self, dataset: OneDimensionalDataset) -> OneDimensionalDataset:  # type: ignore[override]
        """
        Apply ALS baseline correction.

        Parameters
        ----------
        dataset : OneDimensionalDataset
            Input dataset.

        Returns
        -------
        OneDimensionalDataset
            Dataset with baseline subtracted.
        """
        from scipy import sparse
        from scipy.sparse.linalg import spsolve

        y = np.asarray(dataset.dependent_variable_data)
        n = len(y)

        # Difference matrix for smoothness penalty
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(n - 2, n))
        D = self.lambda_ * D.T @ D

        # Initialize weights
        w = np.ones(n)

        # Iterative ALS fitting
        for _ in range(self.max_iter):
            W = sparse.diags(w, 0, shape=(n, n))
            Z = W + D
            baseline = spsolve(Z, w * y)
            # Update weights (asymmetric)
            w = self.p * (y > baseline) + (1 - self.p) * (y <= baseline)

        # Subtract baseline
        y_corrected = y - baseline

        dataset._dependent_variable_data = jnp.asarray(y_corrected)

        return dataset


__all__ = ["AsymmetricLeastSquaresBaseline", "PolynomialBaseline"]
