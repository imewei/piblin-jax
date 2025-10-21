#!/usr/bin/env python3
"""Check for print statements in source code (not tests/examples)."""

import re
import sys


def main():
    """Check for print statements."""
    if len(sys.argv) < 2:
        return 0

    # Read source file
    with open(sys.argv[1]) as f:
        code = f.read()

    # Check for print statements
    if re.search(r"^\s*print\(", code, re.MULTILINE):
        print(f"Found print statement in {sys.argv[1]}")
        print("Use logging instead of print in source code.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
