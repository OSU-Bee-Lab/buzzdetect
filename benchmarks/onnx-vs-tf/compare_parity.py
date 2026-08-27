"""Max absolute difference between two output trees' result CSVs.

A parity failure invalidates the timing question and is the more important
finding; the two halves were checked to 5.6e-05 at export.
"""
import argparse, os, sys
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--a', required=True)
ap.add_argument('--b', required=True)
ap.add_argument('--limit', type=int, default=0, help='0 = every file')
args = ap.parse_args()

def results_in(root):
    out = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith('.csv'):
                full = os.path.join(dirpath, f)
                out[os.path.relpath(full, root)] = full
    return out

a, b = results_in(args.a), results_in(args.b)
shared = sorted(set(a) & set(b))
if not shared:
    sys.exit(f'no result CSVs in common ({len(a)} in a, {len(b)} in b)')
if args.limit:
    shared = shared[:args.limit]

print(f'{len(a)} CSVs in a, {len(b)} in b, comparing {len(shared)}')
only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
for label, missing in (('only in a', only_a), ('only in b', only_b)):
    if missing:
        print(f'  WARNING {label}: {len(missing)} e.g. {missing[:3]}')

worst = 0.0
worst_file = None
mismatched_shape = []
for rel in shared:
    da, db = pd.read_csv(a[rel]), pd.read_csv(b[rel])
    if da.shape != db.shape or list(da.columns) != list(db.columns):
        mismatched_shape.append((rel, da.shape, db.shape))
        continue
    num = da.select_dtypes(include=[np.number]).columns
    diff = np.abs(da[num].to_numpy(dtype=np.float64) - db[num].to_numpy(dtype=np.float64))
    m = float(np.nanmax(diff)) if diff.size else 0.0
    if m > worst:
        worst, worst_file = m, rel

for rel, sa, sb in mismatched_shape:
    print(f'  SHAPE MISMATCH {rel}: {sa} vs {sb}')

print(f'max abs difference: {worst:.3g}' + (f'  (in {worst_file})' if worst_file else ''))
# The CSVs are rounded to digits_results (2) by write/formatting.py:33, so this
# resolves nothing below 0.01 -- a true parity number needs the .npy files the
# microbenchmark saves. Differences here should be 0.00, with the occasional
# 0.01 where a value sits on a rounding boundary.
print('CSV PARITY OK (to the 2 digits the CSVs carry)'
      if worst <= 0.011 and not mismatched_shape else 'CSV PARITY SUSPECT')
