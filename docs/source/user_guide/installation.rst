Installation
============

Requirements
------------

* Python 3.12 or later
* uv package manager (recommended for development)

Basic Installation
------------------

Install quantiq using pip::

    pip install quantiq

This will install quantiq with JAX CPU support and all core dependencies.

GPU Support
-----------

GPU acceleration is available only on Linux with CUDA 12+::

    pip install quantiq[gpu-cuda]

**Platform Constraints:**

* GPU support requires Linux with CUDA 12+
* macOS and Windows users can use CPU backend with 5-10x speedup
* For maximum performance, use Linux with NVIDIA GPU (50-100x speedup)

Development Installation
------------------------

For development with all optional dependencies::

    pip install quantiq[all]

Or clone from source::

    git clone https://github.com/quantiq/quantiq.git
    cd quantiq
    pip install -e ".[dev]"

**Note:** Development requires Python 3.13+ for pre-commit hooks, though the package runs on Python 3.12+.

Verification
------------

Verify your installation::

    python -c "import quantiq; print(quantiq.__version__)"

Check backend availability::

    python -c "from quantiq.backend import get_backend; print(f'Backend: {get_backend()}')"

This will be implemented in Phase 1, Task Group 2.
