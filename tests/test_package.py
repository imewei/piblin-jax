"""
Basic package structure tests.
"""


def test_package_import():
    """Test that quantiq package can be imported."""
    import quantiq

    assert quantiq.__version__ == "0.0.2"


def test_package_has_version():
    """Test that package has a version attribute."""
    import quantiq

    assert hasattr(quantiq, "__version__")
    assert isinstance(quantiq.__version__, str)
