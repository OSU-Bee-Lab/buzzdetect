"""Shared setup for the unittest suite.

The engine addresses `models/` and `src/stream/drivers/` by relative path (see
src/config.py), so it only runs correctly with the engine directory as the
working directory. Importing this module puts us there and puts that directory
on sys.path, which lets the tests be run from anywhere:

    .venv/bin/python3 -m unittest discover -s tests

Everything here is standard library. The engine ships no test runner and the
suite deliberately adds no dependency it doesn't already have.
"""

import os
import sys

DIR_ENGINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_FIXTURES = os.path.join(DIR_ENGINE, 'tests', 'fixtures')

os.chdir(DIR_ENGINE)
if DIR_ENGINE not in sys.path:
    sys.path.insert(0, DIR_ENGINE)


def fixture(name):
    return os.path.join(DIR_FIXTURES, name)


def engine_sources():
    """Every .py file in the engine's own source tree (not the legacy GUI)."""
    for root, dirs, files in os.walk(os.path.join(DIR_ENGINE, 'src')):
        dirs[:] = [d for d in dirs if d not in ('__pycache__', 'gui')]
        for name in files:
            if name.endswith('.py'):
                yield os.path.join(root, name)
