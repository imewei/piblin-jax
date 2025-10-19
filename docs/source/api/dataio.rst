Data I/O
========

Overview
--------

The ``quantiq.dataio`` module provides a comprehensive file I/O system for reading and
writing measurement data. It implements an extensible architecture that supports multiple
file formats with automatic format detection and intelligent hierarchy building.

The I/O system is designed around several key principles:

- **Format Agnostic**: The module provides generic readers for CSV and TXT files that
  automatically parse column-based data. The extensible reader registry allows easy
  addition of custom format parsers without modifying core code.

- **Auto-Detection**: File formats are automatically detected based on file extension
  and content inspection. You can read files without knowing their format in advance,
  making batch processing straightforward.

- **Automatic Hierarchy Building**: When reading multiple files, the system automatically
  analyzes metadata to identify constant and varying experimental conditions, then builds
  an appropriate hierarchical structure (Experiment, MeasurementSet, etc.). This eliminates
  manual organization of large datasets.

- **Batch Operations**: Read entire directories or multiple files at once. The module
  provides convenient functions for common workflows like reading all CSV files in a
  directory or processing files from multiple experimental runs.

- **Metadata Preservation**: All file-level metadata (filenames, paths, timestamps) is
  automatically captured and attached to the resulting data structures. Inline metadata
  from file headers is also preserved.

The module currently supports CSV and TXT formats out of the box, with the infrastructure
in place for adding instrument-specific readers (e.g., TA Instruments, Anton Paar) as
needed.

Quick Examples
--------------

Reading a Single File
^^^^^^^^^^^^^^^^^^^^^

Read a data file with automatic format detection::

    from quantiq.dataio import read_file

    # Read CSV file - format auto-detected
    measurement = read_file("experiment_data.csv")

    # Access the data
    print(f"Number of datasets: {len(measurement.datasets)}")
    print(f"Metadata: {measurement.metadata}")

Reading Multiple Files
^^^^^^^^^^^^^^^^^^^^^^

Read and organize multiple files into a hierarchy::

    from quantiq.dataio import read_files

    # List of files to read
    files = [
        "sample_25C_rep1.csv",
        "sample_25C_rep2.csv",
        "sample_25C_rep3.csv",
        "sample_50C_rep1.csv",
        "sample_50C_rep2.csv",
    ]

    # Read all files and build hierarchy
    experiment_set = read_files(files)

    # Hierarchy is built automatically based on conditions
    for exp in experiment_set.experiments:
        print(f"Temperature: {exp.metadata['temperature']}")
        print(f"Replicates: {len(exp.measurement_sets[0].measurements)}")

Reading Entire Directories
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Process all files in a directory::

    from quantiq.dataio import read_directory

    # Read all CSV files in directory
    experiment_set = read_directory(
        "/path/to/data",
        pattern="*.csv"
    )

    # Read recursively with custom pattern
    experiment_set = read_directory(
        "/path/to/data",
        pattern="sample_*.txt",
        recursive=True
    )

    print(f"Total measurements: {len(experiment_set.get_all_measurements())}")

Registering Custom Readers
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add support for custom file formats::

    from quantiq.dataio import register_reader
    from quantiq.data.collections import Measurement

    def my_custom_reader(filepath):
        """Read custom instrument format."""
        # Parse file
        data = parse_my_format(filepath)

        # Create datasets
        datasets = [create_dataset(data)]

        # Return Measurement
        return Measurement(
            datasets=datasets,
            metadata={"source": filepath}
        )

    # Register reader for .dat files
    register_reader(".dat", my_custom_reader)

    # Now you can read .dat files
    measurement = read_file("data.dat")

See Also
--------

- :doc:`data` - Data structures created by I/O operations
- :doc:`transform` - Processing data after loading
- `pathlib Documentation <https://docs.python.org/3/library/pathlib.html>`_ - Path handling utilities

API Reference
-------------

Module Contents
^^^^^^^^^^^^^^^

.. automodule:: quantiq.dataio
   :members:
   :undoc-members:
   :show-inheritance:

Readers
-------

Base Reader Interface
^^^^^^^^^^^^^^^^^^^^^

.. automodule:: quantiq.dataio.readers
   :members:
   :undoc-members:
   :show-inheritance:

CSV Reader
^^^^^^^^^^

.. automodule:: quantiq.dataio.readers.csv
   :members:
   :undoc-members:
   :show-inheritance:

TXT Reader
^^^^^^^^^^

.. automodule:: quantiq.dataio.readers.txt
   :members:
   :undoc-members:
   :show-inheritance:

Hierarchy Building
------------------

.. automodule:: quantiq.dataio.hierarchy
   :members:
   :undoc-members:
   :show-inheritance:

Writers
-------

.. automodule:: quantiq.dataio.writers
   :members:
   :undoc-members:
   :show-inheritance:
