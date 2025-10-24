"""
Custom Transforms Example for piblin-jax.

This example demonstrates how to extend piblin-jax by creating custom
transform classes for domain-specific data processing.

Key concepts:
- Subclassing DatasetTransform
- Implementing the apply() method
- Adding custom parameters and validation
- Integration with Pipeline system
- JIT-compiled custom transforms
- Error handling and metadata preservation

Run time: <1 second

Author: piblin-jax developers
Date: 2025-10-20
"""

import numpy as np

from piblin_jax.backend import jnp
from piblin_jax.backend.operations import jit
from piblin_jax.data.datasets import OneDimensionalDataset
from piblin_jax.transform.base import DatasetTransform
from piblin_jax.transform.pipeline import Pipeline


class LogTransform(DatasetTransform):
    """
    Custom transform that applies logarithmic scaling to data.

    This is a simple example showing the minimal requirements
    for a custom transform class.

    Parameters
    ----------
    base : float, default=10
        Logarithm base (e.g., 10 for log10, np.e for natural log)
    offset : float, default=0
        Offset added before taking logarithm (for handling zeros)
    """

    def __init__(self, base: float = 10.0, offset: float = 0.0):
        """Initialize LogTransform with parameters."""
        super().__init__()
        self.base = base
        self.offset = offset

    def _apply(self, dataset: OneDimensionalDataset) -> OneDimensionalDataset:
        """
        Apply logarithmic transform to dataset y-values.

        Parameters
        ----------
        dataset : OneDimensionalDataset
            Input dataset

        Returns
        -------
        OneDimensionalDataset
            Transformed dataset with log-scaled y values
        """
        # Add offset and take logarithm
        y_offset = dataset.dependent_variable_data + self.offset
        y_log = jnp.log(y_offset) / jnp.log(self.base)

        # Create new dataset with transformed y values
        return OneDimensionalDataset(
            independent_variable_data=dataset.independent_variable_data,
            dependent_variable_data=y_log,
        )


class MovingAverageTransform(DatasetTransform):
    """
    Custom transform for moving average smoothing.

    Demonstrates parameter validation and JIT compilation
    for improved performance.

    Parameters
    ----------
    window_size : int
        Size of the moving average window (must be odd and > 0)
    mode : str, default='same'
        How to handle edges: 'same' or 'valid'

    Raises
    ------
    ValueError
        If window_size is not positive and odd
    """

    def __init__(self, window_size: int, mode: str = "same"):
        """Initialize MovingAverageTransform with validation."""
        super().__init__()

        # Validate parameters
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if window_size % 2 == 0:
            raise ValueError(f"window_size must be odd, got {window_size}")
        if mode not in ["same", "valid"]:
            raise ValueError(f"mode must be 'same' or 'valid', got {mode}")

        self.window_size = window_size
        self.mode = mode

    @staticmethod
    @jit(static_argnums=(1,))
    def _compute_moving_average(y: jnp.ndarray, window_size: int) -> jnp.ndarray:
        """
        Compute moving average using JIT compilation.

        This method is compiled once and cached for reuse,
        providing significant performance benefits.
        """
        # Simple moving average using convolution
        kernel = jnp.ones(window_size) / window_size
        # Use valid mode to avoid edge artifacts
        result = jnp.convolve(y, kernel, mode="valid")
        return result

    def _apply(self, dataset: OneDimensionalDataset) -> OneDimensionalDataset:
        """
        Apply moving average smoothing to dataset.

        Parameters
        ----------
        dataset : OneDimensionalDataset
            Input dataset

        Returns
        -------
        OneDimensionalDataset
            Smoothed dataset
        """
        # Compute moving average
        y_smooth = self._compute_moving_average(dataset.dependent_variable_data, self.window_size)

        if self.mode == "same":
            # Pad to match original length
            pad_left = (self.window_size - 1) // 2
            pad_right = self.window_size - 1 - pad_left
            y_smooth = jnp.pad(y_smooth, (pad_left, pad_right), mode="edge")
            x_out = dataset.independent_variable_data
        else:  # valid
            # Trim x to match output length
            trim = (self.window_size - 1) // 2
            x_out = dataset.independent_variable_data[trim : -trim if trim > 0 else None]

        return OneDimensionalDataset(
            independent_variable_data=x_out,
            dependent_variable_data=y_smooth,
        )


class DerivativeTransform(DatasetTransform):
    """
    Custom transform for numerical differentiation.

    Demonstrates working with both x and y values and
    handling different numerical methods.

    Parameters
    ----------
    method : str, default='central'
        Differentiation method: 'forward', 'backward', or 'central'
    order : int, default=1
        Order of derivative (1 for first derivative, 2 for second, etc.)
    """

    def __init__(self, method: str = "central", order: int = 1):
        """Initialize DerivativeTransform."""
        super().__init__()

        if method not in ["forward", "backward", "central"]:
            raise ValueError(f"Unknown method: {method}")
        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")

        self.method = method
        self.order = order

    def _apply(self, dataset: OneDimensionalDataset) -> OneDimensionalDataset:
        """
        Compute numerical derivative of dataset.

        Parameters
        ----------
        dataset : OneDimensionalDataset
            Input dataset

        Returns
        -------
        OneDimensionalDataset
            Derivative dataset
        """
        x = dataset.independent_variable_data
        y = dataset.dependent_variable_data

        # Compute derivative based on method
        if self.method == "forward":
            dy = jnp.diff(y) / jnp.diff(x)
            x_out = x[:-1]
        elif self.method == "backward":
            dy = jnp.diff(y) / jnp.diff(x)
            x_out = x[1:]
        else:  # central
            dy = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
            x_out = x[1:-1]

        # Handle higher-order derivatives recursively
        if self.order > 1:
            derivative_ds = OneDimensionalDataset(
                independent_variable_data=x_out, dependent_variable_data=dy
            )
            return DerivativeTransform(self.method, self.order - 1).apply_to(derivative_ds)

        return OneDimensionalDataset(
            independent_variable_data=x_out,
            dependent_variable_data=dy,
        )


class ConditionalTransform(DatasetTransform):
    """
    Advanced example: Apply different transforms based on conditions.

    This demonstrates how to create complex, adaptive transforms
    that make decisions based on data properties.

    Parameters
    ----------
    threshold : float
        Threshold value for condition
    transform_below : DatasetTransform
        Transform to apply when y < threshold
    transform_above : DatasetTransform
        Transform to apply when y >= threshold
    """

    def __init__(
        self,
        threshold: float,
        transform_below: DatasetTransform,
        transform_above: DatasetTransform,
    ):
        """Initialize ConditionalTransform."""
        super().__init__()
        self.threshold = threshold
        self.transform_below = transform_below
        self.transform_above = transform_above

    def _apply(self, dataset: OneDimensionalDataset) -> OneDimensionalDataset:
        """
        Apply conditional transform to dataset.

        Parameters
        ----------
        dataset : OneDimensionalDataset
            Input dataset

        Returns
        -------
        OneDimensionalDataset
            Conditionally transformed dataset
        """
        # Compute statistics
        y_mean = float(jnp.mean(dataset.dependent_variable_data))

        # Apply appropriate transform
        if y_mean < self.threshold:
            print(f"Mean {y_mean:.2f} < {self.threshold:.2f}, applying transform_below")
            return self.transform_below.apply_to(dataset)
        else:
            print(f"Mean {y_mean:.2f} >= {self.threshold:.2f}, applying transform_above")
            return self.transform_above.apply_to(dataset)


def demonstrate_basic_custom_transform():
    """Demonstrate basic custom transform creation."""
    print("=" * 70)
    print("BASIC CUSTOM TRANSFORM: LogTransform")
    print("=" * 70)

    # Create test dataset
    x = np.linspace(1, 100, 50)
    y = 10.0 * x**2

    dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

    print(f"\nOriginal data range: y = [{y.min():.1f}, {y.max():.1f}]")

    # Apply custom transform
    log_transform = LogTransform(base=10, offset=0)
    transformed = log_transform.apply_to(dataset)

    print(
        f"Log10 transformed range: y = "
        f"[{transformed.dependent_variable_data.min():.2f}, {transformed.dependent_variable_data.max():.2f}]"
    )
    print("Transformed dataset created")
    print("\n✓ Basic custom transform works!")
    print("=" * 70)
    print()


def demonstrate_jit_compiled_transform():
    """Demonstrate JIT-compiled custom transform."""
    print("=" * 70)
    print("JIT-COMPILED TRANSFORM: MovingAverageTransform")
    print("=" * 70)

    # Create noisy dataset
    x = np.linspace(0, 10, 1000)
    y = np.sin(x) + 0.2 * np.random.normal(size=len(x))

    dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

    print(f"\nOriginal data: {len(dataset.independent_variable_data)} points")
    print(f"Noise level (std): {np.std(y - np.sin(x)):.3f}")

    # Apply moving average
    smooth_transform = MovingAverageTransform(window_size=21, mode="same")
    smoothed = smooth_transform.apply_to(dataset)

    print(f"Smoothed data: {len(smoothed.independent_variable_data)} points")
    print(
        f"Residual noise (std): {np.std(smoothed.dependent_variable_data - np.sin(smoothed.independent_variable_data)):.3f}"
    )
    print("\n✓ JIT-compiled transform provides fast smoothing!")
    print("=" * 70)
    print()


def demonstrate_pipeline_integration():
    """Demonstrate integrating custom transforms with Pipeline."""
    print("=" * 70)
    print("PIPELINE INTEGRATION: Custom + Built-in Transforms")
    print("=" * 70)

    # Create dataset
    x = np.linspace(0.1, 10, 100)
    y = x**2 + np.random.normal(0, 1, size=len(x))

    dataset = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y)

    # Build pipeline with custom and built-in transforms
    pipeline = Pipeline(
        [
            MovingAverageTransform(window_size=5),  # Custom smooth
            LogTransform(base=10, offset=0),  # Custom log scale
            DerivativeTransform(method="central", order=1),  # Custom derivative
        ]
    )

    print("\nPipeline steps:")
    for i, transform in enumerate(pipeline, 1):
        print(f"  {i}. {transform.__class__.__name__}")

    # Apply pipeline
    result = pipeline.apply_to(dataset)

    print(f"\nFinal result: {len(result.independent_variable_data)} points")
    print("Pipeline result created")
    print("\n✓ Custom transforms integrate seamlessly with Pipeline!")
    print("=" * 70)
    print()


def demonstrate_conditional_transform():
    """Demonstrate conditional/adaptive transforms."""
    print("=" * 70)
    print("ADVANCED: ConditionalTransform")
    print("=" * 70)

    # Create two datasets with different characteristics
    x = np.linspace(0, 10, 100)

    # Dataset 1: Small values
    y1 = 0.5 + 0.1 * np.random.normal(size=len(x))
    dataset1 = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y1)

    # Dataset 2: Large values
    y2 = 10.0 + 1.0 * np.random.normal(size=len(x))
    dataset2 = OneDimensionalDataset(independent_variable_data=x, dependent_variable_data=y2)

    # Create conditional transform
    conditional = ConditionalTransform(
        threshold=5.0,
        transform_below=MovingAverageTransform(window_size=5),
        transform_above=LogTransform(base=10),
    )

    print("\nApplying conditional transform to dataset1:")
    conditional.apply_to(dataset1)
    print("Result: Transformed dataset\n")

    print("Applying conditional transform to dataset2:")
    conditional.apply_to(dataset2)
    print("Result: Transformed dataset")

    print("\n✓ Conditional transforms enable adaptive processing!")
    print("=" * 70)
    print()


def print_implementation_guide():
    """Print guide for implementing custom transforms."""
    print("=" * 70)
    print("CUSTOM TRANSFORM IMPLEMENTATION GUIDE")
    print("=" * 70)
    print()
    print("To create a custom transform:")
    print()
    print("1. **Subclass DatasetTransform:**")
    print("   ```python")
    print("   from piblin_jax.transform.base import DatasetTransform")
    print()
    print("   class MyTransform(DatasetTransform):")
    print("       def __init__(self, param1, param2):")
    print("           super().__init__()")
    print("           self.param1 = param1")
    print("           self.param2 = param2")
    print("   ```")
    print()
    print("2. **Implement apply() method:**")
    print("   ```python")
    print("   def apply(self, dataset):")
    print("       # Your transformation logic")
    print("       new_y = transform(dataset.dependent_variable_data)")
    print(
        "       return OneDimensionalDataset(independent_variable_data=dataset.independent_variable_data, dependent_variable_data=new_y)"
    )
    print("   ```")
    print()
    print("3. **Add validation (optional but recommended):**")
    print("   ```python")
    print("   if param1 <= 0:")
    print("       raise ValueError('param1 must be positive')")
    print("   ```")
    print()
    print("4. **Use JIT for performance (optional):**")
    print("   ```python")
    print("   from piblin_jax.backend.operations import jit")
    print()
    print("   @staticmethod")
    print("   @jit")
    print("   def _compute(x):")
    print("       return expensive_operation(x)")
    print("   ```")
    print()
    print("5. **Preserve metadata (best practice):**")
    print("   ```python")
    print("   metadata = {**dataset.metadata, 'my_param': self.param1}")
    print("   ```")
    print()
    print("6. **Use in pipelines:**")
    print("   ```python")
    print("   pipeline = Pipeline([MyTransform(param1=10)])")
    print("   result = pipeline.apply_to(dataset)")
    print("   ```")
    print()
    print("=" * 70)
    print()


def main():
    """Run all custom transform demonstrations."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 16 + "CUSTOM TRANSFORMS EXAMPLE" + " " * 27 + "║")
    print("║" + " " * 18 + "piblin-jax Framework" + " " * 33 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    # 1. Basic custom transform
    demonstrate_basic_custom_transform()

    # 2. JIT-compiled transform
    demonstrate_jit_compiled_transform()

    # 3. Pipeline integration
    demonstrate_pipeline_integration()

    # 4. Conditional transform
    demonstrate_conditional_transform()

    # 5. Implementation guide
    print_implementation_guide()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("✓ Basic custom transform (LogTransform)")
    print("✓ JIT-compiled transform (MovingAverageTransform)")
    print("✓ Derivative transform with configurable methods")
    print("✓ Pipeline integration demonstrated")
    print("✓ Conditional/adaptive transforms")
    print("✓ Implementation guide provided")
    print()
    print("💡 Key takeaways:")
    print("   - Custom transforms extend piblin-jax for domain-specific needs")
    print("   - Use @jit for performance-critical operations")
    print("   - Validate parameters in __init__")
    print("   - Preserve metadata for traceability")
    print("   - Integrate seamlessly with Pipeline system")
    print()
    print("=" * 70)
    print("\n✨ Example completed successfully!\n")


if __name__ == "__main__":
    main()
