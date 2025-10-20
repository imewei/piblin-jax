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
nitpicky = True  # Warn about all missing references
nitpick_ignore = [
    # Ignore external type references that may not resolve
    ('py:class', 'jax.Array'),
    ('py:class', 'numpy.ndarray'),
    ('py:class', 'ArrayLike'),
    # Type hint references
    ('py:class', 'Any'),
    ('py:class', 'dict[str, Any]'),
    ('py:class', 'tuple[Any, Any]'),
    ('py:class', 'tuple[float, float]'),
    ('py:class', 'array_like'),
    ('py:class', 'dict[str, array]'),
    ('py:class', 'Optional'),
    ('py:obj', 'Any'),
    ('py:obj', 'Optional'),
    # NumPy/JAX array types
    ('py:class', 'jnp.ndarray'),
    ('py:class', 'DeviceArray'),
    ('py:class', 'np.ndarray'),
    # Generic types
    ('py:class', 'T'),
    ('py:class', 'Callable'),
    ('py:class', 'TypeVar'),
    # Common type patterns from docstrings
    ('py:class', 'dict[str'),
    ('py:class', 'Any] | None'),
    ('py:class', 'Any]'),
    ('py:class', 'array] | None'),
    ('py:class', 'array_like | None'),
    ('py:class', 'callable'),
    ('py:class', 'sequence'),
    ('py:class', 'ints'),
    ('py:class', 'int/None'),
    ('py:class', 'dtype'),
    ('py:class', '..'),
    ('py:class', '..]'),
    ('py:class', 'str | Path'),
    ('py:class', 'type | None'),
    ('py:class', 'Type | Callable'),
    ('py:class', 'tuple[float'),
    ('py:class', 'tuple[Measurement'),
    ('py:class', 'tuple[Dataset'),
    ('py:class', 'Sequence[str | Path]'),
    ('py:class', 'Reader instance'),
    # quantiq-specific types
    ('py:class', 'list[Transform]'),
    ('py:class', 'list[Measurement]'),
    ('py:class', 'list[Dataset]'),
    ('py:class', 'list[LinearRegion]'),
    ('py:class', 'Measurement'),
    ('py:class', 'MeasurementSet'),
    ('py:class', 'Experiment'),
    ('py:class', 'ExperimentSet'),
    ('py:class', 'Dataset'),
    ('py:class', 'Transform'),
    ('py:class', 'LinearRegion | CompoundRegion'),
    ('py:class', 'BayesianModel'),
    ('py:class', 'PowerLawModel'),
    ('py:class', 'CrossModel'),
    ('py:class', 'CarreauYasudaModel'),
    ('py:class', 'ArrheniusModel'),
    ('py:class', 'quantiq.bayesian.base.BayesianModel'),
    ('py:class', 'quantiq.data.datasets.base.Dataset'),
    # Common object/attribute references
    ('py:obj', 'samples'),
    ('py:obj', 'details'),
    ('py:obj', 'conditions'),
    ('py:obj', 'uncertainty_samples'),
    ('py:obj', 'has_uncertainty'),
    ('py:obj', 'credible_intervals'),
    ('py:obj', 'value'),
    ('py:obj', 'quantiq.transform.pipeline.T'),
    ('py:obj', 'quantiq.transform.base.T'),
    # Additional type patterns
    ('py:class', 'Callable[[Measurement]'),
    ('py:class', 'Callable[[Dataset]'),
    ('py:class', 'bool] | None'),
    ('py:class', 'bool]'),
    ('py:class', 'array-like'),
    ('py:class', 'float]'),
    ('py:class', 'tuple[np.ndarray'),
    ('py:class', 'tuple | None'),
    ('py:class', 'set[str]'),
    ('py:class', 'quantiq.data.collections.measurement.Measurement'),
    ('py:class', 'OneDimensionalDataset'),
    ('py:class', 'np.ndarray] | None'),
    ('py:class', 'np.ndarray]'),
    ('py:class', 'ndarray'),
    ('py:class', 'list[str]'),
    ('py:class', 'list[Measurement]]'),
    ('py:class', 'dict[tuple[Any'),
    ('py:class', 'dict | None'),
    ('py:class', '1)'),
    ('py:class', '(0'),
    ('py:class', '6)'),
    ('py:class', '(10'),
    # Common autosummary obj references
    ('py:obj', 'measurements'),
    ('py:obj', 'datasets'),
    ('py:obj', 'independent_variable_data'),
    ('py:obj', 'dependent_variable_data'),
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
