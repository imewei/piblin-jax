"""
Tests for uncertainty visualization.

This module tests Task Group 16:
- Visualize method for 1D datasets
- Visualization with uncertainty bands
- Multiple confidence levels
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from quantiq.backend import is_jax_available
from quantiq.data.datasets import OneDimensionalDataset


class TestOneDimensionalVisualization:
    """Test visualization for OneDimensionalDataset."""

    def test_basic_visualization(self):
        """Test basic visualization without uncertainty."""
        # Create dataset
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 1.0
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        # Visualize
        fig, ax = dataset.visualize()

        # Check that figure and axis are returned
        assert fig is not None
        assert ax is not None

        # Check that data is plotted
        lines = ax.get_lines()
        assert len(lines) == 1

        # Clean up
        plt.close(fig)

    def test_visualization_with_labels(self):
        """Test visualization with custom labels."""
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 1.0
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        # Visualize with custom labels
        fig, ax = dataset.visualize(xlabel="Time (s)", ylabel="Signal (V)", title="Test Plot")

        # Check labels
        assert ax.get_xlabel() == "Time (s)"
        assert ax.get_ylabel() == "Signal (V)"
        assert ax.get_title() == "Test Plot"

        plt.close(fig)

    def test_visualization_without_uncertainty(self):
        """Test that show_uncertainty=False works (default)."""
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 1.0
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        # Visualize without uncertainty
        fig, ax = dataset.visualize(show_uncertainty=False)

        # Should have only the main line, no fill_between
        lines = ax.get_lines()
        collections = ax.collections  # fill_between creates PolyCollection
        assert len(lines) == 1
        assert len(collections) == 0

        plt.close(fig)

    def test_visualization_with_bootstrap_uncertainty(self):
        """Test visualization with bootstrap uncertainty bands."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 1.0 + 0.5 * np.random.randn(len(x))
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        # Add bootstrap uncertainty
        dataset_with_unc = dataset.with_uncertainty(
            n_samples=100, method="bootstrap", keep_samples=True
        )

        # Visualize with uncertainty
        fig, ax = dataset_with_unc.visualize(show_uncertainty=True, level=0.95)

        # Should have the line and uncertainty band
        lines = ax.get_lines()
        collections = ax.collections
        assert len(lines) == 1
        assert len(collections) > 0  # fill_between creates collections

        # Check that legend exists
        legend = ax.get_legend()
        assert legend is not None

        plt.close(fig)

    @pytest.mark.skipif(not is_jax_available(), reason="JAX required for Bayesian uncertainty")
    def test_visualization_with_bayesian_uncertainty(self):
        """Test visualization with Bayesian uncertainty."""
        np.random.seed(42)
        x = np.linspace(0, 10, 30)
        y = 2.0 * x + 1.0 + 0.2 * np.random.randn(len(x))
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        # Add Bayesian uncertainty
        dataset_with_unc = dataset.with_uncertainty(
            n_samples=100, method="bayesian", keep_samples=True
        )

        # Visualize with uncertainty
        fig, ax = dataset_with_unc.visualize(show_uncertainty=True)

        # Should have line and uncertainty visualization
        lines = ax.get_lines()
        assert len(lines) == 1

        # May or may not have collections depending on implementation
        # Just check that it doesn't error

        plt.close(fig)

    def test_visualization_custom_figsize(self):
        """Test visualization with custom figure size."""
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 1.0
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        # Visualize with custom figsize
        custom_figsize = (12, 8)
        fig, _ax = dataset.visualize(figsize=custom_figsize)

        # Check figure size
        assert fig.get_size_inches()[0] == custom_figsize[0]
        assert fig.get_size_inches()[1] == custom_figsize[1]

        plt.close(fig)

    def test_visualization_with_kwargs(self):
        """Test that additional kwargs are passed to plot."""
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 1.0
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        # Visualize with custom plot kwargs
        fig, ax = dataset.visualize(
            color="red", linestyle="--", linewidth=2, marker="o", markersize=4
        )

        # Check that line has custom properties
        line = ax.get_lines()[0]
        assert line.get_color() == "red"
        assert line.get_linestyle() == "--"
        assert line.get_linewidth() == 2

        plt.close(fig)

    def test_visualization_multiple_confidence_levels(self):
        """Test that we can create visualizations with different confidence levels."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 1.0 + 0.5 * np.random.randn(len(x))
        dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

        # Add bootstrap uncertainty
        dataset_with_unc = dataset.with_uncertainty(
            n_samples=100, method="bootstrap", keep_samples=True
        )

        # Visualize with 95% CI
        fig1, _ax1 = dataset_with_unc.visualize(show_uncertainty=True, level=0.95)
        plt.close(fig1)

        # Visualize with 68% CI (should be narrower)
        fig2, _ax2 = dataset_with_unc.visualize(show_uncertainty=True, level=0.68)
        plt.close(fig2)

        # Both should work without error
        assert fig1 is not None
        assert fig2 is not None
