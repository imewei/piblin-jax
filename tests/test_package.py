"""
Basic package structure tests.
"""


def test_package_import():
    """Test that piblin_jax package can be imported."""
    import piblin_jax

    assert piblin_jax.__version__ == "0.0.3"


def test_package_has_version():
    """Test that package has a version attribute."""
    import piblin_jax

    assert hasattr(piblin_jax, "__version__")
    assert isinstance(piblin_jax.__version__, str)
