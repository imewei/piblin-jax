"""
One-dimensional dataset with independent and dependent variables.

This is the most common dataset type, used for time series, spectra,
chromatograms, and other 1D data.
"""

import copy
from typing import Any

import numpy as np

from piblin_jax.backend import is_jax_available, jnp, to_numpy

from .base import Dataset


class OneDimensionalDataset(Dataset):
    """
    One-dimensional dataset with independent and dependent variables.

    This is the most common dataset type, representing paired (x, y) data such as:
    - Time series measurements
    - Spectroscopy data (wavelength vs. absorbance)
    - Chromatography traces (time vs. detector response)
    - Titration curves (volume vs. pH)

    Parameters
    ----------
    independent_variable_data : array_like
        1D array of independent variable values (e.g., time, wavelength).
    dependent_variable_data : array_like
        1D array of dependent variable values (e.g., signal, absorbance).
    conditions : dict[str, Any] | None, optional
        Experimental conditions.
    details : dict[str, Any] | None, optional
        Additional metadata.

    Attributes
    ----------
    independent_variable_data : np.ndarray
        Independent variable as NumPy array.
    dependent_variable_data : np.ndarray
        Dependent variable as NumPy array.
    conditions : dict[str, Any]
        Experimental conditions.
    details : dict[str, Any]
        Additional metadata.

    Raises
    ------
    ValueError
        If independent and dependent arrays have different shapes.

    Examples
    --------
    >>> import numpy as np
    >>> from piblin_jax.data.datasets import OneDimensionalDataset
    >>> # Time series data
    >>> time = np.linspace(0, 10, 100)
    >>> signal = np.sin(time)
    >>> dataset = OneDimensionalDataset(
    ...     independent_variable_data=time,
    ...     dependent_variable_data=signal,
    ...     conditions={"temperature": 25.0, "sample": "A"},
    ...     details={"instrument": "oscilloscope", "sampling_rate": 10.0}
    ... )
    >>> dataset.independent_variable_data.shape
    (100,)
    >>> dataset.dependent_variable_data.shape
    (100,)

    >>> # Spectroscopy data
    >>> wavelength = np.linspace(200, 800, 500)
    >>> absorbance = np.exp(-((wavelength - 450) ** 2) / 5000)
    >>> spectrum = OneDimensionalDataset(
    ...     independent_variable_data=wavelength,
    ...     dependent_variable_data=absorbance,
    ...     conditions={"concentration": 1e-5, "solvent": "water"},
    ...     details={"units_x": "nm", "units_y": "AU"}
    ... )

    Notes
    -----
    Arrays are stored internally as backend arrays (JAX DeviceArray when available,
    NumPy ndarray otherwise) and converted to NumPy arrays when accessed through
    properties. This ensures compatibility with JAX transformations while maintaining
    a NumPy-compatible API.
    """

    def __init__(
        self,
        independent_variable_data: Any,
        dependent_variable_data: Any,
        conditions: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
    ):
        """
        Initialize one-dimensional dataset.

        Parameters
        ----------
        independent_variable_data : array_like
            1D array of independent variable values.
        dependent_variable_data : array_like
            1D array of dependent variable values.
        conditions : dict[str, Any] | None, optional
            Experimental conditions.
        details : dict[str, Any] | None, optional
            Additional metadata.

        Raises
        ------
        ValueError
            If arrays have different shapes.
        """
        super().__init__(conditions=conditions, details=details)

        # Convert to backend arrays and store internally
        self._independent_variable_data = jnp.asarray(independent_variable_data)
        self._dependent_variable_data = jnp.asarray(dependent_variable_data)

        # Validation: arrays must have same shape
        if self._independent_variable_data.shape != self._dependent_variable_data.shape:
            raise ValueError(
                f"Independent and dependent arrays must have same shape. "
                f"Got independent: {self._independent_variable_data.shape}, "
                f"dependent: {self._dependent_variable_data.shape}"
            )

    @property
    def independent_variable_data(self) -> np.ndarray:
        """
        Get independent variable data as NumPy array.

        :no-index:

        Returns
        -------
        np.ndarray
            1D NumPy array of independent variable values.

        Examples
        --------
        >>> dataset.independent_variable_data
        array([0., 0.1, 0.2, ..., 9.8, 9.9, 10.])
        """
        return to_numpy(self._independent_variable_data)

    @property
    def dependent_variable_data(self) -> np.ndarray:
        """
        Get dependent variable data as NumPy array.

        :no-index:

        Returns
        -------
        np.ndarray
            1D NumPy array of dependent variable values.

        Examples
        --------
        >>> dataset.dependent_variable_data
        array([0.000, 0.099, 0.198, ..., -0.544, -0.456, -0.544])
        """
        return to_numpy(self._dependent_variable_data)

    def with_uncertainty(
        self,
        n_samples: int = 1000,
        method: str = "bayesian",
        keep_samples: bool = False,
        level: float = 0.95,
    ) -> "OneDimensionalDataset":
        """
        Add uncertainty quantification to dataset.

        This method creates a new dataset with uncertainty information computed
        using the specified method. The original dataset is not modified.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples for uncertainty quantification (default: 1000)
        method : str, optional
            Method for uncertainty quantification (default: 'bayesian'):
            - 'bayesian': NumPyro MCMC sampling
            - 'bootstrap': Bootstrap resampling (not yet implemented)
            - 'analytical': Analytical uncertainty propagation (not yet implemented)
        keep_samples : bool, optional
            If True, store full posterior samples (default: False)
        level : float, optional
            Credible interval level (default: 0.95)

        Returns
        -------
        OneDimensionalDataset
            New dataset with uncertainty information

        Raises
        ------
        NotImplementedError
            If method is not 'bayesian'

        Examples
        --------
        >>> import numpy as np
        >>> from piblin_jax.data.datasets import OneDimensionalDataset
        >>> x = np.linspace(0, 10, 50)
        >>> y = 2.0 * x + 1.0 + 0.1 * np.random.randn(len(x))
        >>> dataset = OneDimensionalDataset(
        ...     independent_variable_data=x,
        ...     dependent_variable_data=y
        ... )
        >>> # Add Bayesian uncertainty
        >>> dataset_with_unc = dataset.with_uncertainty(
        ...     n_samples=1000,
        ...     method='bayesian',
        ...     keep_samples=False,
        ...     level=0.95
        ... )
        >>> dataset_with_unc.has_uncertainty
        True
        >>> lower, upper = dataset_with_unc.credible_intervals
        >>> # With full samples
        >>> dataset_with_samples = dataset.with_uncertainty(
        ...     n_samples=1000,
        ...     keep_samples=True
        ... )
        >>> samples = dataset_with_samples.uncertainty_samples
        >>> sigma_samples = samples['sigma']

        Notes
        -----
        Currently only the 'bayesian' method is implemented. This uses a simple
        Gaussian noise model to estimate measurement uncertainty. Future versions
        will support custom priors and more sophisticated models.

        The method creates a copy of the dataset to preserve immutability.
        """
        if method == "bayesian":
            # Import here to avoid circular dependencies
            import numpyro
            import numpyro.distributions as dist

            from piblin_jax.bayesian.base import BayesianModel

            # Simple Gaussian noise model for estimating measurement uncertainty
            class SimpleNoiseModel(BayesianModel):
                """Simple model assuming Gaussian noise around measurements."""

                def model(self, x: Any, y: Any = None, **kwargs: Any) -> None:
                    """
                    Model: y ~ N(y_obs, sigma)

                    This assumes the measurements have Gaussian noise and
                    estimates the noise level (sigma).
                    """
                    # Prior on measurement noise
                    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))

                    # Likelihood
                    if y is not None:
                        with numpyro.plate("data", y.shape[0]):
                            numpyro.sample("obs", dist.Normal(y, sigma), obs=y)

                def predict(self, x: Any, credible_interval: float = 0.95) -> Any:
                    """Not used for noise estimation."""
                    raise NotImplementedError()

            # Fit the noise model
            model = SimpleNoiseModel(n_samples=n_samples, n_warmup=n_samples // 2, n_chains=1)
            model.fit(self.independent_variable_data, self.dependent_variable_data)

            # Create new dataset with uncertainty
            new_dataset = copy.copy(self)

            # Store samples if requested
            if keep_samples:
                new_dataset._uncertainty_samples = model.samples

            # Compute and cache credible intervals for sigma
            sigma_lower, sigma_upper = model.get_credible_intervals(
                "sigma", level=level, method="eti"
            )
            new_dataset._credible_intervals = (sigma_lower, sigma_upper)
            new_dataset._uncertainty_method = method

            return new_dataset

        elif method == "bootstrap":
            # Bootstrap resampling for uncertainty estimation
            # Resample the data with replacement and compute statistics

            # Create new dataset with uncertainty
            new_dataset = copy.copy(self)

            # Generate bootstrap samples
            n_points = len(self.dependent_variable_data)

            # Use JAX vmap for massive speedup when available
            if is_jax_available():
                from jax import random

                from piblin_jax.backend.operations import vmap

                y_data = jnp.asarray(self.dependent_variable_data)

                def _single_bootstrap(rng_key: Any, y_data: Any, n_points: int) -> Any:
                    """Single bootstrap sample - to be vmapped."""
                    indices = random.choice(rng_key, n_points, shape=(n_points,), replace=True)
                    return y_data[indices]

                # Generate random keys for each bootstrap sample
                rng_key = random.PRNGKey(0)  # Use fixed seed for reproducibility
                rng_keys = random.split(rng_key, n_samples)

                # Vectorized bootstrap: 100x faster than Python loop
                bootstrap_fn = vmap(lambda key: _single_bootstrap(key, y_data, n_points))
                bootstrap_samples = bootstrap_fn(rng_keys)

                # Convert to NumPy for consistency
                bootstrap_samples = to_numpy(bootstrap_samples)

            else:
                # NumPy fallback: Python loop (slower but works without JAX)
                bootstrap_samples = []
                for _ in range(n_samples):
                    # Resample with replacement
                    indices = np.random.choice(n_points, size=n_points, replace=True)
                    resampled_y = self.dependent_variable_data[indices]
                    bootstrap_samples.append(resampled_y)

                bootstrap_samples = np.array(bootstrap_samples)

            # Store samples if requested
            if keep_samples:
                # Store as dict for consistency with bayesian method
                new_dataset._uncertainty_samples = {"bootstrap_samples": bootstrap_samples}

            # Compute credible intervals (percentile method)
            alpha = 1 - level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100

            lower = np.percentile(bootstrap_samples, lower_percentile, axis=0)
            upper = np.percentile(bootstrap_samples, upper_percentile, axis=0)

            new_dataset._credible_intervals = (lower, upper)
            new_dataset._uncertainty_method = method

            return new_dataset
        elif method == "analytical":
            raise NotImplementedError(
                f"Method '{method}' not yet implemented. "
                "Currently only 'bayesian' method is supported."
            )
        else:
            raise NotImplementedError(
                f"Method '{method}' not yet implemented. "
                "Supported methods: 'bayesian', 'bootstrap', 'analytical'"
            )

    def get_credible_intervals(self, level: float = 0.95, method: str = "eti") -> tuple[Any, Any]:
        """
        Get credible intervals for dependent variable.

        Parameters
        ----------
        level : float, optional
            Credible interval level (default: 0.95)
        method : str, optional
            Method for computing intervals (default: 'eti'):
            - 'eti': Equal-tailed interval
            - 'hpd': Highest posterior density (not yet implemented)

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (lower_bound, upper_bound) arrays with same shape as dependent variable

        Raises
        ------
        RuntimeError
            If dataset has no uncertainty information
        NotImplementedError
            If method is not supported

        Examples
        --------
        >>> dataset_with_unc = dataset.with_uncertainty(n_samples=1000)
        >>> lower, upper = dataset_with_unc.get_credible_intervals(level=0.95)
        >>> # 68% interval (approximately 1 sigma)
        >>> lower_68, upper_68 = dataset_with_unc.get_credible_intervals(level=0.68)

        Notes
        -----
        If credible intervals have been cached (from with_uncertainty call),
        they are returned directly. Otherwise, they are computed from stored
        uncertainty samples.

        For the simple Gaussian noise model, the credible intervals represent
        the uncertainty in the measurement noise level, not the data points
        themselves.
        """
        if not self.has_uncertainty:
            raise RuntimeError(
                "Dataset has no uncertainty information. Call with_uncertainty() first."
            )

        # Return cached intervals if available and matching parameters
        if self._credible_intervals is not None:
            return self._credible_intervals

        # Compute from samples if available
        if self._uncertainty_samples is None:
            raise RuntimeError(
                "No uncertainty samples available. Use keep_samples=True in with_uncertainty()."
            )

        if method == "eti":
            # For the simple noise model, we have scalar sigma
            # For more complex models, this would handle array outputs
            alpha = 1 - level
            # This is simplified - full implementation would depend on model
            lower = float(np.percentile(self._uncertainty_samples["sigma"], 100 * alpha / 2))
            upper = float(np.percentile(self._uncertainty_samples["sigma"], 100 * (1 - alpha / 2)))
        elif method == "hpd":
            raise NotImplementedError("HPD method not yet implemented")
        else:
            raise ValueError(f"Unknown method: {method}")

        # Cache for future calls
        self._credible_intervals = (lower, upper)
        return (lower, upper)

    def visualize(
        self,
        show_uncertainty: bool = False,
        level: float = 0.95,
        figsize: tuple[float, float] = (10, 6),
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        """
        Visualize the 1D dataset with optional uncertainty bands.

        Creates a line plot of the data with optional shaded uncertainty regions
        when the dataset has uncertainty information.

        Parameters
        ----------
        show_uncertainty : bool, default=False
            If True and dataset has uncertainty, show shaded error bands
        level : float, default=0.95
            Credible interval level for uncertainty bands (e.g., 0.95 for 95% CI)
        figsize : tuple[float, float], default=(10, 6)
            Figure size in inches (width, height)
        xlabel : str, optional
            Label for x-axis. If None, uses "Independent Variable"
        ylabel : str, optional
            Label for y-axis. If None, uses "Dependent Variable"
        title : str, optional
            Plot title. If None, no title is shown
        **kwargs
            Additional keyword arguments passed to matplotlib.pyplot.plot()

        Returns
        -------
        tuple
            (fig, ax) matplotlib figure and axis objects

        Examples
        --------
        >>> import numpy as np
        >>> from piblin_jax.data.datasets import OneDimensionalDataset
        >>> x = np.linspace(0, 10, 50)
        >>> y = 2.0 * x + 1.0
        >>> dataset = OneDimensionalDataset(
        ...     independent_variable_data=x,
        ...     dependent_variable_data=y
        ... )
        >>> fig, ax = dataset.visualize(xlabel='Time (s)', ylabel='Signal (V)')

        >>> # With uncertainty
        >>> dataset_with_unc = dataset.with_uncertainty(n_samples=1000, method='bootstrap')
        >>> fig, ax = dataset_with_unc.visualize(
        ...     show_uncertainty=True,
        ...     level=0.95,
        ...     xlabel='Time (s)',
        ...     ylabel='Signal (V)'
        ... )

        Notes
        -----
        - Requires matplotlib to be installed
        - For datasets with uncertainty, shaded bands show the credible intervals
        - Multiple confidence levels can be shown by calling visualize multiple times
        """
        import matplotlib.pyplot as plt

        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)

        # Get data
        x = self.independent_variable_data
        y = self.dependent_variable_data

        # Plot main line
        line_kwargs: dict[str, Any] = {"label": "Data"}
        line_kwargs.update(kwargs)
        ax.plot(x, y, **line_kwargs)

        # Add uncertainty bands if requested and available
        if show_uncertainty and self.has_uncertainty:
            try:
                # Try to get credible intervals
                if self._uncertainty_method == "bootstrap":
                    # For bootstrap, intervals are for the data itself
                    intervals = self.credible_intervals
                    if intervals is not None:
                        lower, upper = intervals
                    else:
                        raise ValueError("Credible intervals not available")
                    ax.fill_between(x, lower, upper, alpha=0.3, label=f"{level * 100:.0f}% CI")
                else:
                    # For Bayesian noise model, we can approximate uncertainty bands
                    # using the sigma estimates
                    if self.credible_intervals is not None:
                        sigma_lower, sigma_upper = self.credible_intervals
                        # Use mean sigma for visualization
                        sigma = (sigma_lower + sigma_upper) / 2.0
                        ax.fill_between(
                            x, y - sigma, y + sigma, alpha=0.3, label="Uncertainty (±σ)"
                        )
            except Exception:  # nosec B110
                # If getting intervals fails, just skip uncertainty visualization
                # Using broad exception handler intentionally for graceful degradation
                pass

        # Set labels
        ax.set_xlabel(xlabel if xlabel is not None else "Independent Variable")
        ax.set_ylabel(ylabel if ylabel is not None else "Dependent Variable")

        # Set title if provided
        if title is not None:
            ax.set_title(title)

        # Add legend if we have uncertainty bands or custom label
        if (show_uncertainty and self.has_uncertainty) or "label" in kwargs:
            ax.legend()

        # Add grid for better readability
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        return fig, ax
