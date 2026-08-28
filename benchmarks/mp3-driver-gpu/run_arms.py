#!/usr/bin/env python3
"""Run the mp3 read-path arms back to back and report the rate of each.

One analysis at a time, a fresh output directory each time, and the corpus read
once before any of them so no arm pays for a cold page cache. Those three are
not fussiness: a microbenchmark left running alongside an analysis inflated a
result by 60% during the original investigation, and the engine skips files it
has already analysed.

    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \\
      python3 run_arms.py --python ../../engine/.venv-bench/bin/python3 \\
      --out /tmp/mp3arms soundfile scan helper nohelper

Needs an interpreter whose onnxruntime carries a GPU execution provider. On a
machine with more than one cuDNN on the loader path, pin LD_LIBRARY_PATH or the
run dies on its first FusedConv (engine/docs/linux-gpu.md).
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.abspath(os.path.join(HERE, '..', '..', 'engine'))
CORPUS = '/media/server storage/experiments/Chia - Solar Eclipse'

sys.path.insert(0, HERE)
from compare import report      # noqa: E402


def warm(path):
    """Read the corpus once so the first arm is not the one that pays for it."""
    total = 0
    for name in sorted(glob.glob(os.path.join(path, '*', '*.mp3'))):
        with open(name, 'rb') as f:
            while True:
                block = f.read(1 << 24)
                if not block:
                    break
                total += len(block)
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('arms', nargs='+')
    parser.add_argument('--python', default=os.path.join(ENGINE, '.venv-bench', 'bin', 'python3'))
    parser.add_argument('--out', default='/tmp/mp3arms')
    parser.add_argument('--repeat', type=int, default=1)
    args = parser.parse_args()

    print(f'warming the page cache: {warm(CORPUS) / 1e9:.2f} GB', flush=True)

    results = []
    for run in range(args.repeat):
        for arm in args.arms:
            dir_out = os.path.join(args.out, f'{arm}-{run}')
            shutil.rmtree(dir_out, ignore_errors=True)
            os.makedirs(dir_out, exist_ok=True)

            print(f'\n=== {arm} (run {run + 1}/{args.repeat}) -> {dir_out}', flush=True)
            started = time.monotonic()
            with open(os.path.join(dir_out, 'arm.out'), 'wb') as sink:
                code = subprocess.call(
                    [args.python, os.path.join(HERE, 'run_arm.py'), arm, dir_out],
                    cwd=ENGINE, stdout=sink, stderr=subprocess.STDOUT)
            wall = time.monotonic() - started

            logs = glob.glob(os.path.join(dir_out, '*.log'))
            measured = report(logs[0], wall) if logs else None
            if code != 0 or measured is None:
                print(f'  FAILED (exit {code}); see {dir_out}/arm.out', flush=True)
                continue
            results.append((arm, run, measured))
            print(f'  {measured["total_rate"]:.0f}x   {wall:.1f} s wall, '
                  f'{measured["audio_h"]:.1f} audio hours, '
                  f'{measured["n"]} chunks', flush=True)

    print(f'\n{"arm":<12} {"run":>4} {"rate":>9} {"wall s":>9} {"audio h":>9} '
          f'{"steady":>9}')
    for arm, run, m in results:
        print(f'{arm:<12} {run:>4} {m["total_rate"]:8.0f}x {m["wall"]:9.1f} '
              f'{m["audio_h"]:9.1f} {m["r_rate"]:8.0f}x')


if __name__ == '__main__':
    main()
