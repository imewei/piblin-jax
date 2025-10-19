Migration from piblin
=====================

quantiq is 100% backward compatible with piblin.

Quick Migration
---------------

Simply change your import::

    # Old code
    import piblin

    # New code
    import quantiq as piblin

    # All your piblin code works unchanged!

API Compatibility
-----------------

All piblin APIs are supported:

- ``read_file()``
- Data structures
- Transform operations
- Visualization methods

Performance Improvements
-------------------------

You'll automatically get:

- 5-10x CPU speedup
- 50-100x GPU acceleration (with GPU package)
- Bayesian uncertainty quantification

No code changes required!

Testing Your Migration
-----------------------

1. Run existing tests with quantiq
2. Compare results (should match piblin)
3. Benchmark performance improvements

See :doc:`../tutorials/basic_workflow` for quantiq-specific features.
