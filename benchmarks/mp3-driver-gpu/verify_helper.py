"""Helper-process reads must be bit-identical to in-process reads.

Covers the ordinary case and the short read at EOF, which is where reading into
a caller-owned buffer could go wrong: the buffer is longer than the audio left,
so anything past the true end is stale from the previous read.
"""
import os
import sys

import numpy as np

sys.path.insert(0, '.')


def main():
    path = sys.argv[1]
    chunk_s = 500.0

    from src.stream.drivers.mp3 import LocalDriver, Driver

    ref = LocalDriver(path)
    sr, frames = ref.samplerate, ref.frames
    n = int(chunk_s * sr)

    os.environ['BUZZDETECT_MP3_HELPERS'] = 'always'
    got = Driver(path)
    assert got._helper is not None, 'helper did not start; test is meaningless'
    assert (got.samplerate, got.frames) == (sr, frames), 'metadata differs'

    # Ordinary full-length reads from the start.
    checked = 0
    for i in range(3):
        a = ref.read(n)
        b = got.read(n)
        assert a.shape == b.shape, f'chunk {i}: shape {a.shape} vs {b.shape}'
        assert np.array_equal(a, b), f'chunk {i}: samples differ'
        checked += a.shape[0]
    print(f'  3 full chunks   OK  ({checked:,} samples bit-identical)')

    # The short read at EOF.
    tail_start = max(0, frames - n // 3)
    ref.seek(tail_start)
    got.seek(tail_start)
    a, b = ref.read(n), got.read(n)
    assert a.shape == b.shape, f'tail: shape {a.shape} vs {b.shape}'
    assert np.array_equal(a, b), 'tail: samples differ'
    print(f'  short read @EOF OK  (asked {n:,}, got {a.shape[0]:,} both ways)')

    # Read past the end: both must return nothing.
    a, b = ref.read(n), got.read(n)
    assert a.shape[0] == b.shape[0] == 0, f'past EOF: {a.shape} vs {b.shape}'
    print('  read past EOF   OK  (both empty)')

    ref.close()
    got.close()
    print('\nPASS: helper reads are bit-identical to in-process reads')


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()
