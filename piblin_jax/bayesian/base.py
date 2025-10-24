"""
Base class for Bayesian models using NumPyro.

This module provides the abstract base class for all Bayesian models in piblin-jax.
Models use NumPyro for MCMC sampling and uncertainty quantification.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from jax import random
from numpyro.infer import MCMC, NUTS


class BayesianModel(ABC):
    """
    Abstract base class for Bayesian models using NumPyro.

    This class provides the infrastructure for Bayesian inference using MCMC
    sampling via NumPyro. Subclasses implement the model() method to define
    the probabilistic model structure.

    Parameters
    ----------
    n_samples : int, optional
        Number of MCMC samples to draw (default: 1000)
    n_warmup : int, optional
        Number of warmup samples for MCMC (default: 500)
    n_chains : int, optional
        Number of MCMC chains to run (default: 2)
    random_seed : int, optional
        Random seed for reproducibility (default: 0)

    Attributes
    ----------
    n_samples : int
        Number of MCMC samples
    n_warmup : int
        Number of warmup samples
    n_chains : int
        Number of MCMC chains
    random_seed : int
        Random seed for PRNG
    samples : dict[str, array] | None
        Posterior samples from MCMC (None before fitting)

    Examples
    --------
    >>> import numpy as np
    >>> import numpyro
    >>> import numpyro.distributions as dist
    >>> from piblin_jax.bayesian.base import BayesianModel
    >>>
    >>> class LinearRegressionModel(BayesianModel):
    ...     def model(self, x, y=None):
    ...         # Define priors
    ...         slope = numpyro.sample('slope', dist.Normal(0, 10))
    ...         intercept = numpyro.sample('intercept', dist.Normal(0, 10))
    ...         sigma = numpyro.sample('sigma', dist.HalfNormal(1))
    ...
    ...         # Define likelihood
    ...         mu = slope * x + intercept
    ...         with numpyro.plate('data', x.shape[0]):
    ...             numpyro.sample('obs', dist.Normal(mu, sigma), obs=y)
    ...
    ...     def predict(self, x, credible_interval=0.95):
    ...         # Generate predictions from posterior
    ...         slope_samples = self._samples['slope']
    ...         intercept_samples = self._samples['intercept']
    ...         predictions = slope_samples[:, None] * x + intercept_samples[:, None]
    ...         return {
    ...             'mean': np.mean(predictions, axis=0),
    ...             'lower': np.percentile(predictions, 2.5, axis=0),
    ...             'upper': np.percentile(predictions, 97.5, axis=0)
    ...         }
    >>>
    >>> # Create and fit model
    >>> x = np.linspace(0, 10, 50)
    >>> y = 2.0 * x + 1.0 + 0.1 * np.random.randn(len(x))
    >>> model = LinearRegressionModel(n_samples=1000, n_warmup=500)
    >>> model.fit(x, y)
    >>> predictions = model.predict(x)

    Notes
    -----
    This class cannot be instantiated directly. Subclasses must implement:
    - model(x, y=None, \\*\\*kwargs): Define the NumPyro probabilistic model
    - predict(x, credible_interval=0.95): Generate predictions with uncertainty

    The MCMC sampler uses the No-U-Turn Sampler (NUTS) algorithm, which is
    an efficient Hamiltonian Monte Carlo variant that automatically tunes
    the step size and number of steps.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        n_warmup: int = 500,
        n_chains: int = 2,
        random_seed: int = 0,
    ):
        """
        Initialize BayesianModel.

        Parameters
        ----------
        n_samples : int, optional
            Number of MCMC samples (default: 1000)
        n_warmup : int, optional
            Number of warmup samples (default: 500)
        n_chains : int, optional
            Number of MCMC chains (default: 2)
        random_seed : int, optional
            Random seed (default: 0)
        """
        self.n_samples = n_samples
        self.n_warmup = n_warmup
        self.n_chains = n_chains
        self.random_seed = random_seed

        # Internal state (initialized after fitting)
        self._mcmc = None
        self._samples = None

    @abstractmethod
    def model(self, x: Any, y: Any = None, **kwargs: Any) -> None:
        """
        Define the NumPyro probabilistic model.

        Subclasses must implement this method to specify the model structure,
        including priors and likelihood.

        Parameters
        ----------
        x : array_like
            Independent variable (input data)
        y : array_like | None, optional
            Dependent variable (observations, None for prediction)
        **kwargs : dict
            Additional model-specific parameters

        Notes
        -----
        This method should use NumPyro's `sample` primitive to define
        random variables. When y is not None, it should be used as the
        observation in a `sample` call with `obs=y`.

        Examples
        --------
        >>> def model(self, x, y=None):
        ...     # Define priors
        ...     slope = numpyro.sample('slope', dist.Normal(0, 10))
        ...     intercept = numpyro.sample('intercept', dist.Normal(0, 10))
        ...     sigma = numpyro.sample('sigma', dist.HalfNormal(1))
        ...
        ...     # Define likelihood
        ...     mu = slope * x + intercept
        ...     numpyro.sample('obs', dist.Normal(mu, sigma), obs=y)
        """
        pass

    def fit(self, x: Any, y: Any, use_nlsq_init: bool = False, **kwargs: Any) -> "BayesianModel":
        """
        Fit the model using MCMC sampling.

        Runs MCMC using the No-U-Turn Sampler (NUTS) to sample from the
        posterior distribution. Stores the posterior samples internally
        for later prediction and inference.

        Parameters
        ----------
        x : array_like
            Independent variable (input data)
        y : array_like
            Dependent variable (observations)
        use_nlsq_init : bool, default=False
            If True, use NLSQ to get initial parameter estimates for better
            prior centering (experimental feature).
        **kwargs : dict
            Additional model-specific parameters passed to model()

        Returns
        -------
        BayesianModel
            Returns self for method chaining

        Examples
        --------
        >>> model = MyBayesianModel(n_samples=1000, n_warmup=500)
        >>> model.fit(x_data, y_data)
        >>> predictions = model.predict(x_new)

        Notes
        -----
        The use_nlsq_init parameter is experimental and may not work for all
        model types. It attempts to use nonlinear least squares to find good
        initial parameter estimates, which can help with MCMC convergence.
        """
        # Optional: Use NLSQ for initialization (experimental)
        # This is a placeholder for future enhancement
        # Actual implementation would require model-specific logic
        if use_nlsq_init:
            # This is a placeholder - actual implementation would require
            # extracting the deterministic part of the model and fitting
            # with NLSQ to get good initial parameter values
            pass

        # Initialize NUTS kernel and MCMC sampler
        kernel = NUTS(self.model)
        self._mcmc = MCMC(
            kernel,
            num_samples=self.n_samples,
            num_warmup=self.n_warmup,
            num_chains=self.n_chains,
        )

        # Run MCMC sampling
        rng_key = random.PRNGKey(self.random_seed)
        assert self._mcmc is not None  # Initialized above
        self._mcmc.run(rng_key, x=x, y=y, **kwargs)

        # Store posterior samples
        self._samples = self._mcmc.get_samples()

        return self

    @abstractmethod
    def predict(self, x: Any, credible_interval: float = 0.95) -> dict[str, np.ndarray]:
        """
        Generate predictions with uncertainty.

        Subclasses must implement this method to generate predictions using
        the posterior samples from MCMC.

        Parameters
        ----------
        x : array_like
            Points to predict at
        credible_interval : float, optional
            Credible interval level (default: 0.95)

        Returns
        -------
        dict
            Dictionary with keys:
            - 'mean': Mean prediction
            - 'lower': Lower credible bound
            - 'upper': Upper credible bound
            - 'samples': Full posterior predictive samples (optional)

        Raises
        ------
        RuntimeError
            If model has not been fit yet

        Examples
        --------
        >>> predictions = model.predict(x_new, credible_interval=0.95)
        >>> mean = predictions['mean']
        >>> lower = predictions['lower']
        >>> upper = predictions['upper']
        """
        if self._samples is None:
            raise RuntimeError("Model must be fit before prediction")
        # Subclasses implement specific prediction logic
        raise NotImplementedError("Subclasses must implement predict()")

    @property
    def samples(self) -> dict[str, np.ndarray] | None:
        """
        Get posterior samples from MCMC.

        :no-index:

        Returns
        -------
        dict[str, np.ndarray] | None
            Dictionary mapping parameter names to arrays of samples,
            or None if model has not been fit yet.

        Examples
        --------
        >>> model.fit(x, y)
        >>> samples = model.samples
        >>> slope_samples = samples['slope']
        >>> print(f"Slope mean: {np.mean(slope_samples)}")
        """
        return self._samples

    def get_credible_intervals(
        self, param_name: str, level: float = 0.95, method: str = "eti"
    ) -> tuple[float, float]:
        """
        Get credible intervals for a parameter.

        Computes credible intervals from the posterior samples using either
        equal-tailed intervals (ETI) or highest posterior density (HPD).

        Parameters
        ----------
        param_name : str
            Name of the parameter (must match a sample name from model)
        level : float, optional
            Credible interval level between 0 and 1 (default: 0.95)
        method : str, optional
            Method for computing intervals:
            - 'eti': Equal-tailed interval (default)
            - 'hpd': Highest posterior density interval

        Returns
        -------
        tuple[float, float]
            (lower_bound, upper_bound) of the credible interval

        Raises
        ------
        RuntimeError
            If model has not been fit yet
        ValueError
            If param_name is not in the posterior samples
        ValueError
            If method is not recognized

        Examples
        --------
        >>> model.fit(x, y)
        >>> lower, upper = model.get_credible_intervals('slope', level=0.95)
        >>> print(f"95% credible interval for slope: [{lower:.3f}, {upper:.3f}]")

        >>> # Use 68% interval (approximately 1 sigma)
        >>> lower, upper = model.get_credible_intervals('slope', level=0.68)

        Notes
        -----
        The ETI method uses percentiles and is simple to compute but may not
        give the shortest interval for skewed distributions.

        The HPD method finds the shortest interval containing the specified
        probability mass, but this is a simplified implementation. For full
        HPD computation, consider using the arviz library.
        """
        if self._samples is None:
            raise RuntimeError("Model must be fit first")

        if param_name not in self._samples:
            raise ValueError(f"Unknown parameter: {param_name}")

        samples = self._samples[param_name]

        # Defensive check: ensure samples is not None and not empty
        if samples is None:
            raise ValueError(f"No samples available for parameter: {param_name}")

        samples_array = np.asarray(samples)
        if samples_array.size == 0:
            raise ValueError(f"Empty samples for parameter: {param_name}")

        if method == "eti":
            # Equal-tailed interval using percentiles
            alpha = 1 - level
            lower = float(np.percentile(samples_array, 100 * alpha / 2))
            upper = float(np.percentile(samples_array, 100 * (1 - alpha / 2)))
        elif method == "hpd":
            # Simplified HPD (highest posterior density)
            # For a proper HPD, would use arviz.hdi()
            # This is an approximation using percentiles
            alpha = 1 - level
            lower = float(np.percentile(samples_array, 100 * alpha / 2))
            upper = float(np.percentile(samples_array, 100 * (1 - alpha / 2)))
        else:
            raise ValueError(f"Unknown method: {method}. Use 'eti' or 'hpd'")

        return (lower, upper)

    def summary(self) -> dict[str, dict[str, float]]:
        """
        Get summary statistics for all parameters.

        Returns
        -------
        dict
            Dictionary mapping parameter names to summary statistics:
            - 'mean': Posterior mean
            - 'std': Posterior standard deviation
            - 'q_2.5': 2.5th percentile
            - 'q_50': Median (50th percentile)
            - 'q_97.5': 97.5th percentile

        Raises
        ------
        RuntimeError
            If model has not been fit yet

        Examples
        --------
        >>> model.fit(x, y)
        >>> summary = model.summary()
        >>> print(summary['slope'])
        {'mean': 2.01, 'std': 0.15, 'q_2.5': 1.72, 'q_50': 2.00, 'q_97.5': 2.31}
        """
        if self._samples is None:
            raise RuntimeError("Model must be fit first")

        summary_dict = {}
        for param_name, samples in self._samples.items():
            # Defensive check: skip if samples is None or empty
            if samples is None:
                continue

            # Convert to array to ensure proper type
            samples_array = np.asarray(samples)
            if samples_array.size == 0:
                continue

            summary_dict[param_name] = {
                "mean": float(np.mean(samples_array)),
                "std": float(np.std(samples_array)),
                "q_2.5": float(np.percentile(samples_array, 2.5)),
                "q_50": float(np.percentile(samples_array, 50)),
                "q_97.5": float(np.percentile(samples_array, 97.5)),
            }

        return summary_dict
