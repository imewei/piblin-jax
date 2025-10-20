"""
Task Group 17: Performance Benchmarks

Benchmark tests comparing quantiq performance against baseline.
Uses pytest-benchmark for systematic performance testing.
"""

import numpy as np
import pytest

from quantiq.bayesian.models import PowerLawModel
from quantiq.data.datasets import OneDimensionalDataset
from quantiq.fitting import fit_curve
from quantiq.transform.dataset.normalization import MinMaxNormalize
from quantiq.transform.dataset.smoothing import GaussianSmooth, MovingAverageSmooth
from quantiq.transform.pipeline import Pipeline


class TestDatasetPerformance:
    """Benchmark dataset operations."""

    @pytest.mark.benchmark(group="dataset-creation")
    def test_benchmark_dataset_creation_small(self, benchmark):
        """Benchmark creating small dataset."""

        def create_dataset():
            x = np.linspace(0, 10, 100)
            y = np.sin(x)
            return OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        result = benchmark(create_dataset)
        assert result is not None

    @pytest.mark.benchmark(group="dataset-creation")
    def test_benchmark_dataset_creation_large(self, benchmark):
        """Benchmark creating large dataset."""

        def create_dataset():
            x = np.linspace(0, 100, 10000)
            y = np.sin(x) + np.random.randn(10000) * 0.1
            return OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        result = benchmark(create_dataset)
        assert result is not None

    @pytest.mark.benchmark(group="dataset-copy")
    def test_benchmark_dataset_copy(self, benchmark):
        """Benchmark dataset copying."""
        x = np.linspace(0, 100, 10000)
        y = np.sin(x)
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        result = benchmark(dataset.copy)
        assert result is not None


class TestTransformPerformance:
    """Benchmark transform operations."""

    @pytest.mark.benchmark(group="smoothing")
    def test_benchmark_gaussian_smoothing_small(self, benchmark):
        """Benchmark Gaussian smoothing on small dataset."""
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + np.random.randn(100) * 0.1
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)
        transform = GaussianSmooth(sigma=1.0)

        result = benchmark(transform.apply_to, dataset, make_copy=True)
        assert result is not None

    @pytest.mark.benchmark(group="smoothing")
    def test_benchmark_gaussian_smoothing_large(self, benchmark):
        """Benchmark Gaussian smoothing on large dataset."""
        x = np.linspace(0, 100, 10000)
        y = np.sin(x) + np.random.randn(10000) * 0.1
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)
        transform = GaussianSmooth(sigma=2.0)

        result = benchmark(transform.apply_to, dataset, make_copy=True)
        assert result is not None

    @pytest.mark.benchmark(group="smoothing")
    def test_benchmark_moving_average_smoothing(self, benchmark):
        """Benchmark moving average smoothing."""
        x = np.linspace(0, 100, 10000)
        y = np.sin(x) + np.random.randn(10000) * 0.1
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)
        transform = MovingAverageSmooth(window_size=11)

        result = benchmark(transform.apply_to, dataset, make_copy=True)
        assert result is not None

    @pytest.mark.benchmark(group="normalization")
    def test_benchmark_normalization(self, benchmark):
        """Benchmark MinMax normalization."""
        x = np.linspace(0, 100, 10000)
        y = np.random.randn(10000) * 100 + 500
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)
        transform = MinMaxNormalize()

        result = benchmark(transform.apply_to, dataset, make_copy=True)
        assert result is not None


class TestPipelinePerformance:
    """Benchmark pipeline operations."""

    @pytest.mark.benchmark(group="pipeline")
    def test_benchmark_simple_pipeline(self, benchmark):
        """Benchmark simple 2-stage pipeline."""
        x = np.linspace(0, 100, 5000)
        y = np.sin(x) + np.random.randn(5000) * 0.1
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        pipeline = Pipeline([GaussianSmooth(sigma=1.0), MinMaxNormalize()])

        result = benchmark(pipeline.apply_to, dataset, make_copy=True)
        assert result is not None

    @pytest.mark.benchmark(group="pipeline")
    def test_benchmark_complex_pipeline(self, benchmark):
        """Benchmark complex 4-stage pipeline."""
        x = np.linspace(0, 100, 5000)
        y = np.sin(x) + np.random.randn(5000) * 0.2
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        pipeline = Pipeline(
            [
                MovingAverageSmooth(window_size=5),
                GaussianSmooth(sigma=0.5),
                MinMaxNormalize(),
                GaussianSmooth(sigma=0.2),
            ]
        )

        result = benchmark(pipeline.apply_to, dataset, make_copy=True)
        assert result is not None


class TestFittingPerformance:
    """Benchmark curve fitting operations."""

    @pytest.mark.benchmark(group="fitting")
    def test_benchmark_power_law_fitting(self, benchmark):
        """Benchmark power law model fitting."""
        # Generate power law data: y = A * x^n
        x = np.linspace(0.1, 10, 50)
        A_true, n_true = 2.5, 0.7
        y_true = A_true * x**n_true
        y = y_true + np.random.randn(50) * 0.1

        def fit_power_law():
            def power_law(x, A, n):
                return A * x**n

            return fit_curve(power_law, x, y, p0=[1.0, 0.5])

        result = benchmark(fit_power_law)
        assert result is not None
        assert "params" in result

    @pytest.mark.benchmark(group="fitting")
    def test_benchmark_exponential_fitting(self, benchmark):
        """Benchmark exponential model fitting."""
        x = np.linspace(0, 5, 50)
        a_true, b_true = 2.0, 0.5
        y_true = a_true * np.exp(b_true * x)
        y = y_true + np.random.randn(50) * 0.5

        def fit_exponential():
            def exponential(x, a, b):
                return a * np.exp(b * x)

            return fit_curve(exponential, x, y, p0=[1.0, 0.1])

        result = benchmark(fit_exponential)
        assert result is not None


class TestBayesianPerformance:
    """Benchmark Bayesian inference operations."""

    @pytest.mark.benchmark(group="bayesian")
    def test_benchmark_power_law_bayesian_small(self, benchmark):
        """Benchmark Bayesian power law fitting with small sample count."""
        # Generate power law data
        shear_rate = np.logspace(-2, 2, 30)
        A_true, n_true = 100.0, 0.7
        viscosity_true = A_true * shear_rate**n_true
        viscosity = viscosity_true + np.random.randn(30) * 5.0

        dataset = OneDimensionalDataset(
            independent_variable_data=shear_rate, dependent_variable_data=viscosity
        )

        def fit_bayesian():
            model = PowerLawModel()
            # Use fewer samples for faster benchmark
            model.fit(
                dataset.independent_variable_data,
                dataset.dependent_variable_data,
                num_warmup=200,
                num_samples=200,
                num_chains=1,
            )
            return model

        result = benchmark(fit_bayesian)
        assert result is not None


class TestMemoryEfficiency:
    """Benchmark memory efficiency (not timing, but included for completeness)."""

    @pytest.mark.benchmark(group="memory")
    def test_benchmark_copy_vs_inplace(self, benchmark):
        """Compare copy vs in-place transform performance."""
        x = np.linspace(0, 100, 10000)
        y = np.random.randn(10000)
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)
        transform = MinMaxNormalize()

        # Benchmark in-place (should be faster)
        def inplace_transform():
            ds = dataset.copy()
            return transform.apply_to(ds, make_copy=False)

        result = benchmark(inplace_transform)
        assert result is not None
