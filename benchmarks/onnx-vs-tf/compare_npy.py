"""Full-precision parity between the two runtimes' raw predictions.

The result CSVs are rounded to 2 digits before writing (write/formatting.py:33)
so they cannot resolve the 5.6e-05 agreement checked at export. The
microbenchmark saves its unrounded prediction array; this compares those.
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--a', required=True)
ap.add_argument('--b', required=True)
args = ap.parse_args()

a, b = np.load(args.a), np.load(args.b)
print(f'shapes: {a.shape} vs {b.shape}')
if a.shape != b.shape:
    raise SystemExit('SHAPE MISMATCH -- not comparable')

diff = np.abs(a - b)
print(f'max abs difference : {diff.max():.3g}')
print(f'mean abs difference: {diff.mean():.3g}')
print(f'top class agrees   : {(a.argmax(axis=-1) == b.argmax(axis=-1)).mean()*100:.4f}% of frames')
print('PARITY OK' if diff.max() < 1e-3 else 'PARITY SUSPECT')
