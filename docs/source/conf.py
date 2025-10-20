"""
Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
project = "quantiq"
copyright = "2025, quantiq developers"
author = "quantiq developers"
release = "0.1.0"
version = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
    "numpydoc",
]

# Napoleon settings for NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_imath = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

# NumPyDoc settings
numpydoc_show_class_members = True
numpydoc_show_inherited_class_members = True
numpydoc_class_members_toctree = False  # Disable to prevent autosummary stub warnings
numpydoc_use_plots = True

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = False

# Strict mode: Fail on warnings during build
# Set to True for CI/CD to catch documentation issues early
#
# NOTE: Future improvement opportunity - Consider sphinx-autodoc-typehints extension
# to reduce the nitpick_ignore list by ~50%. This would provide better type rendering
# and automatic maintenance, but requires additional testing and integration work.
# See: https://github.com/tox-dev/sphinx-autodoc-typehints
nitpicky = True  # Warn about all missing references
nitpick_ignore = [
    # External library types (NumPy, JAX)
    ('py:class', 'jax.Array'),
    ('py:class', 'numpy.ndarray'),
    ('py:class', 'np.ndarray'),
    ('py:class', 'jnp.ndarray'),
    ('py:class', 'ndarray'),
    ('py:class', 'DeviceArray'),
    ('py:class', 'ArrayLike'),
    ('py:class', 'array_like'),

    # Python typing module types
    ('py:class', 'Any'),
    ('py:class', 'Optional'),
    ('py:class', 'Callable'),
    ('py:class', 'TypeVar'),
    ('py:class', 'T'),
    ('py:obj', 'Any'),
    ('py:obj', 'Optional'),

    # Common type hint patterns from docstrings
    # Note: These are broad patterns to catch variations in type formatting
    ('py:class', 'dict[str, Any]'),
    ('py:class', 'dict[str, array]'),
    ('py:class', 'dict[str'),  # Sphinx parses dict[str, X] as separate parts
    ('py:class', 'dict | None'),
    ('py:class', 'tuple[Any, Any]'),
    ('py:class', 'tuple[float, float]'),
    ('py:class', 'tuple | None'),
    ('py:class', 'array_like | None'),
    ('py:class', 'type | None'),
    ('py:class', 'Type | Callable'),
    ('py:class', 'str | Path'),
    ('py:class', 'Sequence[str | Path]'),

    # Type hint fragments (how Sphinx parses complex union/generic types)
    # When Sphinx sees "dict[str, array] | None", it splits into parts:
    # - "dict[str" (opening bracket)
    # - "array] | None" (closing bracket with union)
    ('py:class', 'Any]'),  # From dict[str, Any]
    ('py:class', 'Any] | None'),  # From dict[str, Any] | None
    ('py:class', 'array] | None'),  # From dict[str, array] | None
    ('py:class', 'bool]'),  # From Callable[[...], bool]
    ('py:class', 'bool] | None'),  # From Callable[[...], bool] | None
    ('py:class', 'float]'),  # From tuple[float, float]
    ('py:class', 'np.ndarray]'),  # From tuple[np.ndarray, np.ndarray]
    ('py:class', 'np.ndarray] | None'),  # From tuple[...] | None
    ('py:class', 'tuple[float'),  # From tuple[float, ...]
    ('py:class', 'tuple[np.ndarray'),  # From tuple[np.ndarray, ...]
    ('py:class', 'tuple[Measurement'),  # From tuple[Measurement, ...]
    ('py:class', 'tuple[Dataset'),  # From tuple[Dataset, ...]
    ('py:class', 'Callable[[Measurement]'),  # From Callable[[Measurement], ...]
    ('py:class', 'Callable[[Dataset]'),  # From Callable[[Dataset], ...]
    ('py:class', 'dict[tuple[Any'),  # From dict[tuple[Any, ...], ...]
    ('py:class', 'list[Measurement]]'),  # From list[Measurement] (extra bracket artifact)
    ('py:class', '..]'),  # From Ellipsis in type hints like tuple[int, ...]

    # Numeric literals from default values in docstrings
    # When docstrings have "default=(0, 1)" or "figsize=(10, 6)", Sphinx parses the
    # numbers as potential type references. These suppress those false positives.
    ('py:class', '(0'),  # From default=(0, ...)
    ('py:class', '1)'),  # From default=(..., 1)
    ('py:class', '(10'),  # From default=(10, ...)
    ('py:class', '6)'),  # From default=(..., 6)

    # Generic descriptive type names from docstrings
    ('py:class', 'callable'),
    ('py:class', 'sequence'),
    ('py:class', 'array-like'),
    ('py:class', 'dtype'),
    ('py:class', 'ints'),
    ('py:class', 'int/None'),
    ('py:class', 'Reader instance'),

    # quantiq-specific types (data structures)
    ('py:class', 'Dataset'),
    ('py:class', 'OneDimensionalDataset'),
    ('py:class', 'Measurement'),
    ('py:class', 'MeasurementSet'),
    ('py:class', 'Experiment'),
    ('py:class', 'ExperimentSet'),
    ('py:class', 'Transform'),
    ('py:class', 'LinearRegion | CompoundRegion'),
    ('py:class', 'quantiq.data.datasets.base.Dataset'),
    ('py:class', 'quantiq.data.collections.measurement.Measurement'),

    # quantiq-specific types (models)
    ('py:class', 'BayesianModel'),
    ('py:class', 'PowerLawModel'),
    ('py:class', 'CrossModel'),
    ('py:class', 'CarreauYasudaModel'),
    ('py:class', 'ArrheniusModel'),
    ('py:class', 'quantiq.bayesian.base.BayesianModel'),

    # quantiq-specific collection types
    ('py:class', 'list[Transform]'),
    ('py:class', 'list[Measurement]'),
    ('py:class', 'list[Dataset]'),
    ('py:class', 'list[LinearRegion]'),
    ('py:class', 'list[str]'),
    ('py:class', 'set[str]'),

    # Common object/attribute references from autosummary
    ('py:obj', 'samples'),
    ('py:obj', 'details'),
    ('py:obj', 'conditions'),
    ('py:obj', 'uncertainty_samples'),
    ('py:obj', 'has_uncertainty'),
    ('py:obj', 'credible_intervals'),
    ('py:obj', 'value'),
    ('py:obj', 'measurements'),
    ('py:obj', 'datasets'),
    ('py:obj', 'independent_variable_data'),
    ('py:obj', 'dependent_variable_data'),
    ('py:obj', 'quantiq.transform.pipeline.T'),
    ('py:obj', 'quantiq.transform.base.T'),
]

# Suppress warnings (for development builds)
suppress_warnings = [
    "autosummary",  # Suppress autosummary stub file warnings
    "ref.citation",  # Suppress unreferenced citation warnings (low priority)
    "ref.obj",  # Suppress object reference warnings from autosummary tables
]

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The suffix(es) of source filenames.
source_suffix = {
    ".rst": "restructuredtext",
}

# The master toctree document.
master_doc = "index"

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
}

# -- Options for intersphinx extension ---------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}

# -- Options for todo extension ----------------------------------------------
todo_include_todos = True

# -- Options for MathJax -----------------------------------------------------
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
