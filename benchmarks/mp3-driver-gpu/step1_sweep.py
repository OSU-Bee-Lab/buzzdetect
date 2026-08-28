"""Step 1, at scale: how often does a mid-stream decode converge *exactly*?

step1_converge.py showed convergence within ~2200 samples at three offsets and
a persistent ~1e-7 residual at a fourth. That difference is the whole gate, so
this sweeps many offsets per file and reports the distribution. One continuous
decode pass serves every offset in the file, so the cost is one full decode
rather than one per probe.

Usage:  python3 step1_sweep.py <n_offsets> <file.mp3> [more.mp3 ...]
Run from engine/.
"""

import os
import sys

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from step1_converge import (audio_start, boundary_at_or_after,   # noqa: E402
                            fragment, frames_from)
from src.stream.drivers.mp3 import LocalDriver   # noqa: E402

WINDOW = 1 << 18


def reference_windows(path, offsets):
    """Decode the file once, returning WINDOW samples at each sample offset.

    A single forward pass: the offsets are visited in order, so this costs one
    decode of the file up to the last offset rather than one per probe.
    """
    windows = {}
    track = LocalDriver(path)
    position = 0
    step = 1 << 22
    for want in sorted(offsets):
        while position < want:
            got = track.read(min(step, want - position))
            if got.shape[0] == 0:
                break
            position += got.shape[0]
        data = track.read(WINDOW)
        windows[want] = data
        position += data.shape[0]
    track.close()
    return windows


def measure(path, n_offsets):
    track = LocalDriver(path)
    sr, total = track.samplerate, track.frames
    track.close()

    start = audio_start(path)
    size = os.path.getsize(path)
    audio_bytes = size - start

    probes = []
    for i in range(1, n_offsets + 1):
        target = start + int(audio_bytes * i / (n_offsets + 1))
        header = boundary_at_or_after(path, target)
        if header is None:
            continue
        offset = header['offset']
        _, after = frames_from(path, offset)
        sample_offset = total - after
        if sample_offset < 0 or sample_offset + WINDOW > total:
            continue
        probes.append((sample_offset, offset))

    windows = reference_windows(path, [s for s, _ in probes])

    print(f'\n{os.path.basename(path)}  {size / 1e6:.0f} MB  {total / sr / 3600:.2f} h')
    print(f'  {"byte":>12} {"sample":>13} {"converged@":>11} {"residual":>10}  verdict')

    exact = 0
    worst = 0
    residuals = []
    for sample_offset, offset in probes:
        reference = windows[sample_offset]
        candidate = fragment(path, offset, WINDOW)
        n = min(reference.shape[0], candidate.shape[0])
        bad = np.nonzero(reference[:n] != candidate[:n])[0]
        if bad.size == 0:
            converged, residual = 0, 0.0
        else:
            converged = int(bad[-1]) + 1
            residual = float(np.abs(reference[:n] - candidate[:n])[4096:].max())
        ok = converged < 8192
        exact += ok
        if ok:
            worst = max(worst, converged)
        residuals.append(residual)
        print(f'  {offset:>12,} {sample_offset:>13,} {converged:>11,} '
              f'{residual:>10.2e}  {"exact" if ok else "NOT EXACT"}')

    print(f'  exact: {exact}/{len(probes)}   worst converged@ among exact: {worst:,}'
          f'   max residual: {max(residuals):.2e}')
    return exact, len(probes), worst


def main():
    n_offsets = int(sys.argv[1])
    totals = [0, 0, 0]
    for path in sys.argv[2:]:
        exact, count, worst = measure(path, n_offsets)
        totals[0] += exact
        totals[1] += count
        totals[2] = max(totals[2], worst)
    print(f'\nTOTAL exact {totals[0]}/{totals[1]}   worst overlap needed {totals[2]:,} samples')


if __name__ == '__main__':
    main()
