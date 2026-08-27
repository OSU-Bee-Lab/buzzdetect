#!/usr/bin/env python3
"""Render results.csv as a rate table: streamers down, chunk length across."""

import argparse
import csv
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default=os.path.join(HERE, 'results.csv'))
    args = p.parse_args()

    with open(args.csv) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return print('no results')

    # Several runs of a cell: keep the best, since a slow one usually means
    # something else was using the machine.
    cells = defaultdict(list)
    for r in rows:
        key = (int(r['streamers']), float(r['chunklength']))
        cells[key].append(r)

    streamers = sorted({k[0] for k in cells})
    chunks = sorted({k[1] for k in cells})

    def best(key):
        ok = [r for r in cells.get(key, []) if r['status'] == 'ok']
        if ok:
            return max(ok, key=lambda r: float(r['rate_x']))
        return cells.get(key, [None])[0]

    print(f"corpus: {rows[0]['corpus_files']} files, "
          f"{rows[0]['corpus_hours_est']} audio hours, "
          f"{rows[0]['processor'].upper()}, {rows[0]['analyzers']} analyzer(s)\n")

    head = 'streamers | ' + ' | '.join(f'{c:>10.0f}s' for c in chunks)
    print(head)
    print('-' * len(head))
    for s in streamers:
        cellstr = []
        for c in chunks:
            r = best((s, c))
            if r is None:
                cellstr.append(f'{"-":>11}')
            elif r['status'] == 'ok':
                cellstr.append(f"{float(r['rate_x']):>10.0f}x")
            else:
                cellstr.append(f"{r['status'].upper():>11}")
        print(f'{s:>9} | ' + ' | '.join(cellstr))

    ok = [r for r in rows if r['status'] == 'ok']
    if ok:
        b = max(ok, key=lambda r: float(r['rate_x']))
        print(f"\nfastest: {b['rate_x']}x at {b['streamers']} streamers, "
              f"{float(b['chunklength']):.0f}s chunks "
              f"({b['wall_s']}s wall, {b['peak_rss_mb']}MB peak RSS)")
    bad = [r for r in rows if r['status'] != 'ok']
    for r in bad:
        print(f"  {r['status']}: {r['streamers']} streamers, "
              f"{float(r['chunklength']):.0f}s -- {r['note'][:120]}")


if __name__ == '__main__':
    main()
