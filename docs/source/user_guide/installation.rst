Installation
============

Requirements
------------

* Python 3.12 or later
* pip or conda package manager

Basic Installation
------------------

Install quantiq using pip::

    pip install quantiq

This will install quantiq with JAX CPU support and all core dependencies.

GPU Support
-----------

For NVIDIA CUDA GPUs::

    pip install quantiq[gpu-cuda]

For Apple Silicon (Metal)::

    pip install quantiq[gpu-metal]

For AMD ROCm GPUs::

    pip install quantiq[gpu-rocm]

Development Installation
------------------------

For development with all optional dependencies::

    pip install quantiq[all]

Or clone from source::

    git clone https://github.com/quantiq/quantiq.git
    cd quantiq
    pip install -e ".[dev]"

Verification
------------

Verify your installation::

    python -c "import quantiq; print(quantiq.__version__)"

Check backend availability::

    python -c "from quantiq.backend import get_backend; print(f'Backend: {get_backend()}')"

This will be implemented in Phase 1, Task Group 2.
