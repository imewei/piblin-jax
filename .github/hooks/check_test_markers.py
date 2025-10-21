#!/usr/bin/env python3
"""Check that pytest markers used in tests are registered in pyproject.toml."""

import re
import sys


def main():
    """Check test markers."""
    if len(sys.argv) < 2:
        return 0

    # Read test file
    with open(sys.argv[1]) as f:
        test_code = f.read()

    # Find all markers used in test file
    markers = re.findall(r"@pytest\.mark\.(\w+)", test_code)

    # Read pyproject.toml to find registered markers
    with open("pyproject.toml") as f:
        config = f.read()

    config_markers = re.findall(r'"(\w+):', config)

    # Built-in pytest markers that don't need registration
    builtin = {"parametrize", "skip", "skipif", "xfail", "usefixtures", "filterwarnings"}

    # Find unregistered markers
    unregistered = set(markers) - set(config_markers) - builtin

    if unregistered:
        print(f"Unregistered pytest markers: {unregistered}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
