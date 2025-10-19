"""Data types and utilities for quantiq.

This package provides:
- Core dataset types (Dataset, ConditionSet, Replicate)
- Hierarchical collections (Collection, DataNode, DataTree)
- Metadata utilities (merging, validation, extraction)
- Region of Interest (ROI) definitions
"""

from . import datasets
from . import collections
from . import metadata
from . import roi

__all__ = ["datasets", "collections", "metadata", "roi"]
