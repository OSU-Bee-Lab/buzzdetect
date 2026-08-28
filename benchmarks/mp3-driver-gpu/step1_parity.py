"""Is exact convergence decided by the parity of the fragment's start frame?

step1_sweep.py found a clean alternation: fragments beginning at an odd frame
index reproduced the continuous decode bit for bit, fragments at an even index
left a ~1e-7 residual forever. That is not bit-reservoir behaviour -- the
reservoir is bounded and dies out in two frames -- it is the polyphase synthesis
window being summed in a different order, which is a per-block parity in
mpg123's synth. If the rule holds, the tail-scan plan is alive: pick a boundary
of the right parity and the decode is exact.

This probes consecutive frame boundaries in one region, so parity is the only
thing that varies.

Usage:  python3 step1_parity.py <file.mp3> [more.mp3 ...]
Run from engine/.
"""

import os
import sys

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from step1_converge import (audio_start, boundary_at_or_after,   # noqa: E402
                            fragment, frames_from, parse_header)
from src.stream.drivers.mp3 import LocalDriver   # noqa: E402

WINDOW = 1 << 17
N_FRAMES = 8          # consecutive boundaries probed
SAMPLES_PER_FRAME = 1152


def consecutive_boundaries(path, first_offset, count):
    """`count` frame offsets starting at `first_offset`, walked by header."""
    offsets = []
    at = first_offset
    with open(path, 'rb') as f:
        for _ in range(count):
            f.seek(at)
            buf = f.read(64)
            header = parse_header(buf, 0)
            if header is None:
                break
            offsets.append(at)
            at += header['size']
    return offsets


def main():
    for path in sys.argv[1:]:
        track = LocalDriver(path)
        sr, total = track.samplerate, track.frames
        track.close()

        start = audio_start(path)
        size = os.path.getsize(path)
        target = start + int((size - start) * 0.40)
        header = boundary_at_or_after(path, target)
        offsets = consecutive_boundaries(path, header['offset'], N_FRAMES)

        _, after = frames_from(path, offsets[0])
        base_sample = total - after
        base_frame = base_sample // SAMPLES_PER_FRAME

        # One reference decode covering every fragment's window.
        span = WINDOW + N_FRAMES * SAMPLES_PER_FRAME
        ref_track = LocalDriver(path)
        position = 0
        while position < base_sample:
            got = ref_track.read(min(1 << 22, base_sample - position))
            if got.shape[0] == 0:
                break
            position += got.shape[0]
        reference = ref_track.read(span)
        ref_track.close()

        print(f'\n{os.path.basename(path)}   total frames {total // SAMPLES_PER_FRAME:,}'
              f'   base frame {base_frame:,}')
        print(f'  {"frame":>12} {"parity":>7} {"converged@":>11} {"residual":>10}')
        for i, offset in enumerate(offsets):
            frame_index = base_frame + i
            candidate = fragment(path, offset, WINDOW)
            aligned = reference[i * SAMPLES_PER_FRAME:i * SAMPLES_PER_FRAME + WINDOW]
            n = min(aligned.shape[0], candidate.shape[0])
            bad = np.nonzero(aligned[:n] != candidate[:n])[0]
            converged = 0 if bad.size == 0 else int(bad[-1]) + 1
            residual = (0.0 if bad.size == 0
                        else float(np.abs(aligned[:n] - candidate[:n])[4096:].max()))
            print(f'  {frame_index:>12,} {"odd" if frame_index % 2 else "even":>7} '
                  f'{converged:>11,} {residual:>10.2e}')


if __name__ == '__main__':
    main()
