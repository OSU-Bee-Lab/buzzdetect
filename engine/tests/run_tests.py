"""Run the whole engine test suite.

    .venv/bin/python3 tests/run_tests.py            everything
    .venv/bin/python3 tests/run_tests.py -k manifest   just what matches

Two things are being run: the unittest suite in this directory, and
test_mp3_driver.py, which predates it and is a plain script with its own
oracle and its own reporting (see its docstring). This is the one command that
covers both.

Nothing here is a dependency the engine doesn't already have: no pytest, no
runner beyond the standard library's.
"""

import argparse
import os
import subprocess
import sys
import unittest

import _context as ctx  # noqa: F401  (chdir + sys.path)

DIR_TESTS = os.path.dirname(os.path.abspath(__file__))
SCRIPT_TESTS = ['test_mp3_driver.py']


def run_unittests(pattern, verbosity):
    suite = unittest.defaultTestLoader.discover(DIR_TESTS, top_level_dir=ctx.DIR_ENGINE)
    if pattern:
        suite = filter_suite(suite, pattern)
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    return result.wasSuccessful()


def filter_suite(suite, pattern):
    kept = unittest.TestSuite()
    for test in iterate(suite):
        if pattern in test.id():
            kept.addTest(test)
    return kept


def iterate(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from iterate(item)
        else:
            yield item


def run_scripts():
    ok = True
    for name in SCRIPT_TESTS:
        print(f'\n=== {name} ===', flush=True)
        completed = subprocess.run([sys.executable, os.path.join('tests', name)],
                                   cwd=ctx.DIR_ENGINE)
        ok = ok and completed.returncode == 0
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-k', dest='pattern', default=None,
                        help='only run tests whose id contains this substring')
    parser.add_argument('-v', dest='verbose', action='store_true')
    parser.add_argument('--unit-only', action='store_true',
                        help='skip the standalone script tests')
    args = parser.parse_args()

    ok = run_unittests(args.pattern, 2 if args.verbose else 1)
    if not args.unit_only and not args.pattern:
        ok = run_scripts() and ok

    return 0 if ok else 1


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    sys.exit(main())
