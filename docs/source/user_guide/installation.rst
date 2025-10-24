Installation
============

Requirements
------------

**Runtime Requirements:**

* Python 3.12 or later
* Operating System: Linux, macOS, or Windows

**Development Requirements:**

* Python 3.13 or later (for pre-commit hooks)
* uv package manager (recommended for development)

Basic Installation
------------------

Install piblin-jax using pip::

    pip install piblin-jax

This will install piblin-jax with JAX CPU support and all core dependencies.

**What's Included:**

* JAX CPU backend (5-10x faster than piblin)
* NumPy backend fallback (automatic if JAX unavailable)
* All core data structures and transforms
* Bayesian inference capabilities (NumPyro)
* File I/O and visualization tools

GPU Support (Linux + CUDA 12+ Only)
------------------------------------

**Platform Constraints:**

* ✅ **Linux + NVIDIA GPU + CUDA 12**: Full GPU acceleration (50-100x speedup)
* ❌ **macOS**: CPU-only (no NVIDIA GPU support, still 5-10x faster than piblin)
* ❌ **Windows**: CPU-only (CUDA support experimental/unstable in JAX)

**Requirements for GPU Acceleration:**

* Linux operating system
* NVIDIA GPU with CUDA Compute Capability 7.5 or newer
* CUDA 12.1-12.9 installed on system
* NVIDIA driver >= 525

**Performance Impact:** 50-100x speedup for large datasets (>1M points)

.. note::

   **Breaking Change (v0.1.0):** The ``piblin-jax[gpu-cuda]`` pip extra has been removed.
   GPU installation now requires explicit manual installation to avoid silent CPU/GPU conflicts.

Recommended Installation (Makefile)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**From repository (works with pip, uv, conda/mamba):**

.. code-block:: bash

    git clone https://github.com/piblin/piblin-jax.git
    cd piblin-jax
    make init
    make install-gpu-cuda  # Handles everything automatically

This command:

* ✓ Validates platform (Linux only)
* ✓ Detects package manager (uv/conda/pip)
* ✓ Uninstalls CPU-only JAX
* ✓ Installs GPU-enabled JAX with CUDA 12
* ✓ Verifies GPU detection
* ✓ Shows installation summary

Manual GPU Installation
^^^^^^^^^^^^^^^^^^^^^^^^

**Why manual installation requires uninstall:**

JAX has separate CPU and GPU builds. You MUST remove the CPU build before
installing GPU to prevent silent failures where you think you have GPU but
are actually using CPU.

**Using pip:**

.. code-block:: bash

    # Step 1: Uninstall CPU-only version (REQUIRED)
    pip uninstall -y jax jaxlib

    # Step 2: Install GPU-enabled JAX
    pip install "jax[cuda12-local]>=0.8.0,<0.9.0"

    # Step 3: Install piblin-jax (if not already installed)
    pip install piblin-jax

    # Step 4: Verify GPU detection
    python -c "import jax; print('Devices:', jax.devices())"
    # Expected: [cuda(id=0)] NOT [CpuDevice(id=0)]

**Using uv:**

.. code-block:: bash

    uv pip uninstall -y jax jaxlib
    uv pip install "jax[cuda12-local]>=0.8.0,<0.9.0"
    python -c "import jax; print(jax.devices())"

**Using conda/mamba:**

Option A: Using environment file (recommended):

.. code-block:: bash

    # Using conda
    conda env create -f environment-gpu.yml
    conda activate piblin-jax-gpu

    # Using mamba (faster)
    mamba env create -f environment-gpu.yml
    mamba activate piblin-jax-gpu

Option B: Manual within conda environment:

.. code-block:: bash

    conda activate your-env
    pip uninstall -y jax jaxlib
    pip install "jax[cuda12-local]>=0.8.0,<0.9.0"

.. warning::

   Conda's extras syntax (``conda install piblin-jax[gpu-cuda]``) is not supported.
   Always use pip within your conda environment for JAX GPU installation.

Verify GPU Installation
^^^^^^^^^^^^^^^^^^^^^^^^

After installation, verify GPU is detected:

.. code-block:: bash

    python -c "from piblin_jax.backend import get_device_info; print(get_device_info())"

**Expected output:**

.. code-block:: python

    {'backend': 'jax', 'device_type': 'gpu', 'device_count': 1, ...}

**If you see ``'device_type': 'cpu'``**, GPU installation failed. See troubleshooting below.

Troubleshooting GPU Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Issue: "GPU not detected" warning or ``device_type: 'cpu'``**

.. code-block:: bash

    # 1. Check GPU hardware
    nvidia-smi  # Should show your GPU

    # 2. Check CUDA version (need 12.1-12.9)
    nvcc --version

    # 3. Verify JAX sees GPU
    python -c "import jax; print(jax.devices())"
    # Expected: [cuda(id=0)]
    # If showing: [CpuDevice(id=0)] → JAX is using CPU

    # 4. If still CPU, reinstall with explicit uninstall:
    pip uninstall -y jax jaxlib
    pip install "jax[cuda12-local]>=0.8.0,<0.9.0"

**Issue: ImportError or "CUDA library not found"**

.. code-block:: bash

    # Set CUDA library path
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

    # Make permanent (add to ~/.bashrc)
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc

**Issue: "An NVIDIA GPU may be present... but a CUDA-enabled jaxlib is not installed"**

This means you have GPU hardware but CPU-only JAX. Solution:

.. code-block:: bash

    pip uninstall -y jax jaxlib
    pip install "jax[cuda12-local]>=0.8.0,<0.9.0"

**Issue: Works in one environment but not another**

Different package managers may install different versions. Always use the same
installation method:

* **Development**: ``make install-gpu-cuda`` (recommended)
* **Production**: Docker with explicit JAX version
* **Notebooks**: Manual pip installation with version pinning

Development Installation
------------------------

**Prerequisites:**

* **Runtime**: Python 3.12+ supported
* **Development**: Python 3.13+ required (for pre-commit hooks)
* **Package Manager**: uv recommended for development (not pip or conda)

For development with all optional dependencies:

.. code-block:: bash

    git clone https://github.com/piblin/piblin-jax.git
    cd piblin-jax

    # Using uv (recommended for development)
    uv pip install -e ".[dev]"

    # Or using pip
    pip install -e ".[dev]"

This includes:

* All runtime dependencies
* Development tools (ruff, mypy, pytest)
* Documentation dependencies (Sphinx, sphinx-rtd-theme)
* Testing dependencies (pytest-cov, pytest-benchmark)

Install Pre-commit Hooks
^^^^^^^^^^^^^^^^^^^^^^^^^

After installing development dependencies (requires Python 3.13+):

.. code-block:: bash

    pre-commit install

This will automatically run code quality checks (formatting, linting, type checking)
before each commit.

Verification
------------

Verify your installation::

    python -c "import piblin_jax; print(piblin_jax.__version__)"

Check backend availability::

    python -c "from piblin_jax.backend import get_backend; print(f'Backend: {get_backend()}')"

Expected output:

* ``Backend: jax`` - JAX is available and being used
* ``Backend: numpy`` - Fallback to NumPy (JAX not installed or unavailable)

Check device type::

    python -c "from piblin_jax.backend import get_device_info; print(get_device_info())"

This provides comprehensive information about:

* Backend type (``jax`` or ``numpy``)
* Device type (``cpu``, ``gpu``, or ``tpu``)
* Available devices
* Platform information
* GPU support status
* CUDA version (if GPU available)

Optional Dependencies
---------------------

piblin-jax supports several optional dependency groups:

**Development dependencies:**

.. code-block:: bash

    pip install piblin-jax[dev]

Includes: ruff, mypy, pre-commit hooks, pytest

**Testing dependencies:**

.. code-block:: bash

    pip install piblin-jax[test]

Includes: pytest, pytest-cov, pytest-benchmark

**Documentation dependencies:**

.. code-block:: bash

    pip install piblin-jax[docs]

Includes: Sphinx, sphinx-rtd-theme, sphinx-autodoc-typehints

**Security scanning:**

.. code-block:: bash

    pip install piblin-jax[security]

Includes: pip-audit, bandit, safety

**All optional dependencies:**

.. code-block:: bash

    pip install piblin-jax[all]

Docker Installation
-------------------

For reproducible environments with GPU support:

**Create Dockerfile:**

.. code-block:: dockerfile

    FROM nvidia/cuda:12.1-runtime-ubuntu22.04

    # Install Python
    RUN apt-get update && apt-get install -y python3.12 python3-pip

    # Install piblin-jax with GPU support
    RUN pip3 uninstall -y jax jaxlib && \
        pip3 install "jax[cuda12-local]>=0.8.0,<0.9.0" && \
        pip3 install piblin-jax

    # Verify installation
    RUN python3 -c "from piblin_jax.backend import get_device_info; print(get_device_info())"

**Build and run:**

.. code-block:: bash

    docker build -t piblin-jax-gpu .
    docker run --gpus all -it piblin-jax-gpu python3

Next Steps
----------

* :doc:`quickstart` - Getting started with piblin-jax
* :doc:`../tutorials/gpu_acceleration` - Maximizing GPU performance
* :doc:`../api/index` - API reference documentation
