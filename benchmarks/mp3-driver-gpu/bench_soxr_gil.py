"""Does soxr.resample release the GIL?

Resampling is 36% of a streamer's per-chunk work. If soxr holds the GIL for it,
eight streamers serialise there and moving the resample into the mp3 helper
process would buy real parallelism. If it releases the GIL, the only thing
moving it would save is IPC bytes, and it is not worth the contract change.

Method: run the same resample on N threads at once. Perfect scaling means the
GIL is released; wall time growing linearly with N means it is held. libsndfile
decode is measured the same way as a reference, since it is known to release.
"""
import sys
import threading
import time

import numpy as np
import soxr

SR_IN, SR_OUT = 44100, 16000
CHUNK_S = 500


def scaling(fn, threads):
    """Wall time for `threads` concurrent calls to fn()."""
    barrier = threading.Barrier(threads)

    def run():
        barrier.wait()
        fn()

    ts = []
    for _ in range(3):
        workers = [threading.Thread(target=run) for _ in range(threads)]
        t0 = time.perf_counter()
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def main():
    n = SR_IN * CHUNK_S
    # Independent buffers per thread, so this measures compute, not cache
    # contention on one shared array.
    buffers = [np.random.randn(n).astype(np.float32) for _ in range(8)]
    counter = iter(range(10_000))
    lock = threading.Lock()

    def resample_one():
        with lock:
            i = next(counter) % len(buffers)
        soxr.resample(buffers[i], SR_IN, SR_OUT, quality='HQ')

    print(f'soxr.resample, {CHUNK_S}s @ {SR_IN}->{SR_OUT}Hz\n')
    print(f"{'threads':>8} {'wall':>9} {'per-call':>10} {'speedup':>9}  verdict")
    base = None
    for t in (1, 2, 4, 8):
        wall = scaling(resample_one, t)
        if base is None:
            base = wall
        speedup = (base * t) / wall
        print(f'{t:>8} {wall:8.3f}s {wall/1:9.3f}s {speedup:8.2f}x'
              f'  {"scales (GIL released)" if speedup > t * 0.6 else "serialised (GIL held)"}')


if __name__ == '__main__':
    main()
