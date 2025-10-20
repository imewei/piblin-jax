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
    # Generic types
    ('py:class', 'T'),
    ('py:class', 'Callable'),
    ('py:class', 'TypeVar'),
]

# Suppress warnings (for development builds)
suppress_warnings = [
    "autosummary",  # Suppress autosummary stub file warnings
    "ref.citation",  # Suppress unreferenced citation warnings (low priority)
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
