"""Does seeking change what today's driver decodes?

The tail-scan plan measures a candidate against "what LocalDriver produces", so
it matters whether that is even a single well-defined answer. mpg123 resets its
decoder on a seek but not necessarily every part of its synthesis state, and
the residual seen when a decode starts mid-stream is exactly the signature of
that state being aligned differently.

So: read a window by decoding straight through to it, and read the same window
by opening the file and seeking to it. If those differ, the streamer's own
chunk assignment already decides results at the 1e-7 level and "bit-identical"
has to be defined against a fixed access pattern.

Usage:  python3 step1_seek.py <file.mp3> [more.mp3 ...]
Run from engine/.
"""

import os
import sys

import numpy as np

sys.path.insert(0, '.')

from src.stream.drivers.mp3 import LocalDriver     # noqa: E402

STEP = 1 << 22
WINDOW = 1 << 20


def straight_through(path, offset, window):
    track = LocalDriver(path)
    position = 0
    while position < offset:
        got = track.read(min(STEP, offset - position))
        if got.shape[0] == 0:
            break
        position += got.shape[0]
    data = track.read(window)
    track.close()
    return data


def by_seek(path, offset, window):
    track = LocalDriver(path)
    track.seek(offset)
    data = track.read(window)
    track.close()
    return data


def by_seek_after_reading(path, offset, window, first):
    """Read `first` samples, then seek to `offset` -- the streamer's pattern."""
    track = LocalDriver(path)
    track.read(first)
    track.seek(offset)
    data = track.read(window)
    track.close()
    return data


def report(label, reference, candidate):
    n = min(reference.shape[0], candidate.shape[0])
    bad = np.nonzero(reference[:n] != candidate[:n])[0]
    residual = 0.0 if bad.size == 0 else float(np.abs(reference[:n] - candidate[:n]).max())
    print(f'    {label:<28} mismatched {bad.size:>9,}/{n:,}  max {residual:.2e}'
          f'{"   IDENTICAL" if bad.size == 0 else ""}')


def main():
    for path in sys.argv[1:]:
        track = LocalDriver(path)
        sr, total = track.samplerate, track.frames
        track.close()
        print(f'\n{os.path.basename(path)}  {total / sr / 3600:.2f} h')

        for fraction in (0.10, 0.50, 0.90):
            offset = int(total * fraction) // 1152 * 1152
            reference = straight_through(path, offset, WINDOW)
            print(f'  offset {offset:,} (frame {offset // 1152:,}):')
            report('seek from a fresh open', reference, by_seek(path, offset, WINDOW))
            report('seek after reading 1 s', reference,
                   by_seek_after_reading(path, offset, WINDOW, sr))


if __name__ == '__main__':
    main()
