"""
Task Group 18: Compatibility Layer Tests

Tests to ensure piblin-jax can be used as a drop-in replacement for piblin.
This verifies that `import piblin_jax as piblin` works correctly.
"""

import numpy as np
import pytest

from piblin_jax.backend import is_jax_available


class TestPiblinCompatibility:
    """Test piblin compatibility layer."""

    def test_import_as_piblin(self):
        """Test that piblin-jax can be imported as piblin."""
        import piblin_jax as piblin

        assert piblin is not None

    def test_core_classes_available(self):
        """Test that core classes are available."""
        import piblin_jax as piblin

        # Core dataset classes
        assert hasattr(piblin, "OneDimensionalDataset")
        assert hasattr(piblin, "TwoDimensionalDataset")
        assert hasattr(piblin, "ThreeDimensionalDataset")

    def test_measurement_classes_available(self):
        """Test that measurement/collection classes are available."""
        import piblin_jax as piblin

        # Collection classes
        assert hasattr(piblin, "Measurement")
        assert hasattr(piblin, "MeasurementSet")
        assert hasattr(piblin, "Experiment")
        assert hasattr(piblin, "ExperimentSet")

    def test_transform_classes_available(self):
        """Test that transform classes are available."""
        import piblin_jax as piblin

        # Transform classes
        assert hasattr(piblin, "Pipeline")
        assert hasattr(piblin, "LambdaTransform")

    @pytest.mark.skipif(not is_jax_available(), reason="JAX required for Bayesian classes")
    def test_bayesian_classes_available(self):
        """Test that Bayesian classes are available."""
        import piblin_jax as piblin

        # Bayesian classes
        assert hasattr(piblin, "BayesianModel")
        assert hasattr(piblin, "PowerLawModel")
        assert hasattr(piblin, "ArrheniusModel")
        assert hasattr(piblin, "CrossModel")
        assert hasattr(piblin, "CarreauYasudaModel")

    def test_dataio_available(self):
        """Test that dataio functions are available."""
        import piblin_jax as piblin

        # Data IO
        assert hasattr(piblin, "read_files")
        assert hasattr(piblin, "GenericCSVReader")

    def test_basic_workflow_as_piblin(self):
        """Test a basic workflow using piblin name."""
        import piblin_jax as piblin

        # Create dataset
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 1.0 + np.random.randn(50) * 0.1

        dataset = piblin.OneDimensionalDataset(
            independent_variable_data=x, dependent_variable_data=y
        )

        assert dataset is not None
        assert len(dataset.independent_variable_data) == 50

    def test_transform_workflow_as_piblin(self):
        """Test transform workflow using piblin name."""
        import piblin_jax as piblin

        # Create dataset
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 1.0

        dataset = piblin.OneDimensionalDataset(
            independent_variable_data=x, dependent_variable_data=y
        )

        # Create lambda transform (function works on arrays, not datasets)
        transform = piblin.LambdaTransform(lambda_func=lambda y_arr: y_arr * 2.0)

        result = transform.apply_to(dataset, make_copy=True)
        np.testing.assert_array_almost_equal(
            result.dependent_variable_data,
            y * 2.0,
            decimal=5,  # JAX may have slightly different floating point precision
        )

    def test_pipeline_workflow_as_piblin(self):
        """Test pipeline workflow using piblin name."""
        import piblin_jax as piblin

        x = np.linspace(0, 10, 50)
        y = np.sin(x)

        dataset = piblin.OneDimensionalDataset(
            independent_variable_data=x, dependent_variable_data=y
        )

        # Create pipeline (functions work on arrays)
        transform1 = piblin.LambdaTransform(lambda_func=lambda y_arr: y_arr * 2.0)
        transform2 = piblin.LambdaTransform(lambda_func=lambda y_arr: y_arr + 1.0)

        pipeline = piblin.Pipeline([transform1, transform2])
        result = pipeline.apply_to(dataset, make_copy=True)

        expected = y * 2.0 + 1.0
        np.testing.assert_array_almost_equal(
            result.dependent_variable_data,
            expected,
            decimal=5,  # JAX may have slightly different floating point precision
        )

    def test_measurement_workflow_as_piblin(self):
        """Test measurement workflow using piblin name."""
        import piblin_jax as piblin

        x = np.linspace(0, 10, 20)
        y = np.random.randn(20)

        dataset = piblin.OneDimensionalDataset(
            independent_variable_data=x, dependent_variable_data=y
        )

        # Measurement takes a list of datasets
        measurement = piblin.Measurement([dataset], conditions={"temperature": 298.15})

        assert len(measurement.datasets) == 1
        assert measurement.conditions["temperature"] == 298.15


class TestBackwardCompatibility:
    """Test backward compatibility features."""

    @pytest.mark.skipif(
        not is_jax_available(), reason="JAX required for BayesianModel export check"
    )
    def test_all_exports_present(self):
        """Test that __all__ contains key exports."""
        import piblin_jax

        # Check __all__ is defined
        assert hasattr(piblin_jax, "__all__")
        assert isinstance(piblin_jax.__all__, list)

        # Check key classes are in __all__
        key_classes = [
            "OneDimensionalDataset",
            "TwoDimensionalDataset",
            "BayesianModel",
            "Pipeline",
            "LambdaTransform",
        ]

        for cls in key_classes:
            assert cls in piblin_jax.__all__, f"{cls} not in __all__"

    def test_version_available(self):
        """Test that version information is available."""
        import piblin_jax

        # Version should be available
        assert hasattr(piblin_jax, "__version__")
        assert isinstance(piblin_jax.__version__, str)


class TestAPIStability:
    """Test API stability and core interfaces."""

    def test_dataset_interface(self):
        """Test that datasets have stable interface."""
        import piblin_jax as piblin

        x = np.linspace(0, 10, 50)
        y = np.sin(x)

        dataset = piblin.OneDimensionalDataset(
            independent_variable_data=x, dependent_variable_data=y
        )

        # Check core attributes
        assert hasattr(dataset, "independent_variable_data")
        assert hasattr(dataset, "dependent_variable_data")
        assert hasattr(dataset, "conditions")
        assert hasattr(dataset, "details")

        # Check core methods
        assert hasattr(dataset, "copy")
        assert hasattr(dataset, "with_uncertainty")

    def test_transform_interface(self):
        """Test that transforms have stable interface."""
        import piblin_jax as piblin

        transform = piblin.LambdaTransform(lambda ds: ds.dependent_variable_data)

        # Check core methods
        assert hasattr(transform, "apply_to")
        assert callable(transform.apply_to)

    @pytest.mark.skipif(not is_jax_available(), reason="JAX required for BayesianModel")
    def test_bayesian_model_interface(self):
        """Test that Bayesian models have stable interface."""
        import piblin_jax as piblin

        # Just check the base class interface
        assert hasattr(piblin.BayesianModel, "fit")
        assert hasattr(piblin.BayesianModel, "predict")
        assert hasattr(piblin.BayesianModel, "get_credible_intervals")
