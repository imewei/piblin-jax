Contributing to piblin-jax
=======================

We welcome contributions to piblin-jax! This guide will help you get started.

Development Setup
-----------------

1. Fork and clone the repository::

    git clone https://github.com/YOUR_USERNAME/piblin_jax.git
    cd piblin-jax

2. Create a virtual environment::

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install development dependencies::

    pip install -e ".[dev]"

4. Install pre-commit hooks::

    pre-commit install

Code Style
----------

piblin-jax follows these style guidelines:

* **ruff** for code formatting and linting (line length 100, replaces black/isort/flake8)
* **mypy** for type checking (strict mode)
* **NumPy-style docstrings** for documentation

Pre-commit hooks will automatically check these before each commit.

Running Tests
-------------

Run the full test suite::

    pytest

Run tests with coverage::

    pytest --cov=piblin_jax --cov-report=html

Run only fast tests::

    pytest -m "not slow"

Run benchmarks::

    pytest -m benchmark

Type Checking
-------------

Run mypy type checking::

    mypy piblin_jax

Building Documentation
----------------------

Build the documentation locally::

    cd docs
    make html
    open _build/html/index.html

Adding New Features
-------------------

Adding a New Transform
^^^^^^^^^^^^^^^^^^^^^^

1. Create the transform class in ``piblin_jax/transform/``
2. Inherit from appropriate base class (DatasetTransform, etc.)
3. Implement ``_apply()`` method with JAX operations
4. Add NumPy fallback if needed
5. Write tests in ``tests/transform/``
6. Add docstring with NumPy style
7. Add to API documentation

Adding a New Bayesian Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Create model class in ``piblin_jax/bayesian/models/``
2. Inherit from ``BayesianModel``
3. Implement ``model()`` method with NumPyro
4. Define priors for parameters
5. Write tests in ``tests/bayesian/models/``
6. Add usage example in docstring

Pull Request Process
--------------------

1. Create a feature branch::

    git checkout -b feature/your-feature-name

2. Make your changes with clear, atomic commits
3. Write tests for your changes (>95% coverage required)
4. Update documentation as needed
5. Run the full test suite and pre-commit hooks
6. Push to your fork and create a pull request

Pull Request Checklist
^^^^^^^^^^^^^^^^^^^^^^

* [ ] Tests pass locally
* [ ] Code coverage >95%
* [ ] Pre-commit hooks pass
* [ ] Documentation updated
* [ ] Changelog entry added
* [ ] Type hints added
* [ ] NumPy-style docstrings

Reporting Issues
----------------

When reporting bugs, please include:

* Python version
* piblin-jax version
* Operating system
* JAX version and backend (CPU/GPU)
* Minimal reproducible example
* Error messages and stack traces

Questions and Support
---------------------

* GitHub Issues: Bug reports and feature requests
* Discussions: Questions and general discussion

License
-------

By contributing to piblin-jax, you agree that your contributions will be
licensed under the MIT License.
