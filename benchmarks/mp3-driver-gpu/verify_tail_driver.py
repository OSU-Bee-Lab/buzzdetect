"""The tail-scan driver must decode what the scanned driver decodes.

The oracle is ordinary libsndfile opened through the length-hiding shim over
the whole file -- the original driver, and the thing whose results are already
in people's output folders. This checks the new read path against it under the
access patterns that actually occur: whole-file sequential reads at several
chunk sizes, the streamer's seek-then-read (worker.py queue_chunk), reads
placed deliberately on and across the clamp, a seek matrix, and reads past the
end.

Usage:  python3 verify_tail_driver.py <file.mp3> [more.mp3 ...]
Run from engine/.
"""

import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, '.')

import src.stream.drivers.mp3 as mp3      # noqa: E402
from src.stream.drivers.mp3 import LocalDriver   # noqa: E402


class Oracle:
    """The original driver: whole-file shim, mpg123's own scan, plain reads."""

    def __init__(self, path):
        self._shim = mp3._LengthHidingFile(path)
        self._track = sf.SoundFile(self._shim)
        self.samplerate = self._track.samplerate
        self.channels = self._track.channels
        self.frames = self._track.frames

    def read(self, n, dtype='float32'):
        return self._track.read(n, dtype=dtype)

    def seek(self, n):
        return self._track.seek(n)

    def close(self):
        self._track.close()
        self._shim.close()


class Failures:
    def __init__(self):
        self.count = 0
        self.notes = 0

    def check(self, ok, label, detail=''):
        mark = 'ok  ' if ok else 'FAIL'
        if not ok:
            self.count += 1
        print(f'    [{mark}] {label}{("  " + detail) if detail else ""}')
        return ok

    def note(self, label, detail):
        """A difference that is known, bounded, and documented."""
        self.notes += 1
        print(f'    [note] {label}  {detail}')


def same(a, b):
    return a.shape == b.shape and np.array_equal(a, b)


def detail(a, b):
    if a.shape != b.shape:
        return f'shapes {a.shape} vs {b.shape}'
    n = min(a.shape[0], b.shape[0])
    bad = np.nonzero(a[:n] != b[:n])[0]
    return (f'{bad.size:,} samples differ, first at {bad[0]:,}, '
            f'max {np.abs(a[:n] - b[:n]).max():.2e}')


def sequential(path, oracle_frames, chunk, failures, start=0, label='', tolerate=0.0):
    """Read forward from `start` to the end, `chunk` samples at a time.

    Both readers get the same seek, so a windowed run compares exactly the
    samples a whole-file run would over that window, for a fraction of the
    decoding.
    """
    oracle = Oracle(path)
    got = LocalDriver(path)
    if start:
        oracle.seek(start)
        got.seek(start)
    ok = True
    position = start
    while True:
        a = oracle.read(chunk)
        b = got.read(chunk)
        if not same(a, b):
            worst = (np.abs(a - b).max() if a.shape == b.shape else float('inf'))
            if worst <= tolerate:
                # Documented: a read broken less than a frame past the clamp
                # leaves the fragment's own read boundaries out of step with the
                # caller's, and libsndfile's mp3 output depends on where reads
                # are broken. See mp3.py's _read_fragment.
                failures.note(f'sequential read{label}, chunk {chunk:,}',
                              f'at sample {position:,}: {detail(a, b)} '
                              f'(within the documented {tolerate:.0e})')
            else:
                failures.check(False, f'sequential read{label}, chunk {chunk:,}',
                               f'at sample {position:,}: {detail(a, b)}')
            ok = False
            break
        if a.shape[0] == 0:
            break
        position += a.shape[0]
    oracle.close()
    got.close()
    if ok:
        failures.check(position == oracle_frames,
                       f'sequential read{label}, chunk {chunk:,}',
                       f'to sample {position:,}, all identical')


def streamer_pattern(path, chunk_seconds, failures, start=0, label=''):
    """seek(chunk start) then read(chunk length), as worker.py does."""
    oracle = Oracle(path)
    got = LocalDriver(path)
    chunk = int(chunk_seconds * oracle.samplerate)
    position = start - start % chunk
    bad = 0
    while position < got.frames:
        oracle.seek(position)
        got.seek(position)
        a = oracle.read(chunk)
        b = got.read(chunk)
        if not same(a, b):
            bad += 1
            if bad == 1:
                failures.check(False, f'streamer pattern{label}, {chunk_seconds:g} s chunks',
                               f'at sample {position:,}: {detail(a, b)}')
        position += chunk
    oracle.close()
    got.close()
    if bad == 0:
        failures.check(True, f'streamer pattern{label}, {chunk_seconds:g} s chunks')


def seam(path, clamp, frames, failures):
    """Reads placed deliberately on, before and across libsndfile's clamp.

    Each case starts from a seek a little before the clamp rather than from the
    file's start: the seek goes to both readers, so the samples compared are the
    ones a whole-file read would compare, without decoding the hours in front of
    them.

    The rules here are the ones tests/test_mp3_driver.py holds the driver to.
    The first half of each split must be identical. The remainder need only be
    close, because a caller that breaks a read at the clamp gets the unbroken
    decode from this driver and the broken one from libsndfile -- which is then
    checked directly, as the last case below.
    """
    approach = max(0, clamp - (1 << 20))
    if frames <= clamp:
        # libsndfile already reaches the end of this one; there is no seam.
        return
    for first in (clamp - approach - 1, clamp - approach, clamp - approach + 1,
                  (clamp + frames) // 2 - approach):
        second = frames - approach - first
        oracle = Oracle(path)
        got = LocalDriver(path)
        oracle.seek(approach)
        got.seek(approach)
        a, b = oracle.read(first), got.read(first)
        if not failures.check(same(a, b), f'read of {first:,} up to the seam',
                              detail(a, b) if not same(a, b) else f'{a.shape[0]:,} samples'):
            oracle.close()
            got.close()
            continue

        a, b = oracle.read(second), got.read(second)
        failures.check(a.shape == b.shape, f'remainder after a break at {first:,} '
                       f'is the right length', f'{b.shape} vs {a.shape}')
        if a.shape != b.shape:
            worst = float('inf')
        else:
            worst = float(np.abs(a - b).max()) if a.shape[0] else 0.0
        if worst == 0.0:
            failures.check(True, f'remainder after a break at {first:,}')
        elif worst <= 1e-6:
            failures.note(f'remainder after a break at {first:,}',
                          f'{detail(a, b)} (within the documented 1e-06)')
        else:
            failures.check(False, f'remainder after a break at {first:,}',
                           detail(a, b) if a.shape == b.shape
                           else f'shapes {a.shape} vs {b.shape}')
        oracle.close()
        got.close()

    # The documented behaviour of a break at the clamp itself.
    unbroken = Oracle(path)
    unbroken.seek(approach)
    expected = unbroken.read(frames - approach)
    unbroken.close()
    got = LocalDriver(path)
    got.seek(approach)
    joined = np.concatenate((got.read(clamp - approach), got.read(frames - clamp)))
    got.close()
    failures.check(same(expected, joined), 'a break at the clamp reads as though unbroken',
                   detail(expected, joined) if not same(expected, joined) else
                   f'{joined.shape[0]:,} samples')


def seeks(path, clamp, frames, failures):
    sr = sf.info(path).samplerate
    # Fewer targets than tests/test_mp3_driver.py sweeps on the fixtures,
    # because each case here opens a fresh Oracle and every Oracle is a full
    # mpg123 scan of a gigabyte-sized file. The fixtures cover the matrix; this
    # confirms it on the real thing.
    targets = [0, clamp - 1, clamp, clamp + 1, (clamp + frames) // 2, frames - 1]
    targets = [t for t in targets if 0 <= t <= frames]
    window = 1 << 16
    bad = 0
    for i, target in enumerate(targets):
        # A fresh pair per case. Sharing one reader across the whole matrix
        # would be measuring mpg123 rather than this driver: what a seek returns
        # depends on what was decoded before it (step1_seek.py), so an oracle
        # that has wandered the file is not a fixed answer to compare against.
        oracle = Oracle(path)
        got = LocalDriver(path)
        # consecutive pairs, so backwards seeks out of the tail are covered
        been_past_clamp = False
        history = False
        for step in (target, targets[(i + 2) % len(targets)]):
            pa, pb = oracle.seek(step), got.seek(step)
            a, b = oracle.read(window), got.read(window)
            if pa != pb or a.shape != b.shape:
                bad += 1
                if bad == 1:
                    failures.check(False, 'seek matrix',
                                   f'seek({step:,}) -> {pa:,} vs {pb:,}, '
                                   f'shapes {a.shape} vs {b.shape}')
                continue
            worst = float(np.abs(a - b).max()) if a.shape[0] else 0.0
            # The rule tests/test_mp3_driver.py holds the driver to: a seek from
            # a reader that has decoded nothing must be exact; once it has, a
            # seek near the seam or after a visit to the tail need only be
            # close, because the two readers are then no longer carrying the
            # same decode history, and what a seek returns depends on it.
            near_seam = step >= clamp - 2 * 1152
            bound = 1e-6 if (history and (near_seam or been_past_clamp)) else 0.0
            if worst > bound:
                bad += 1
                if bad == 1:
                    failures.check(False, 'seek matrix',
                                   f'seek({step:,}): {detail(a, b)}')
            elif worst:
                failures.note('seek matrix',
                              f'seek({step:,}) with history: {detail(a, b)} '
                              f'(within the documented {bound:.0e})')
            been_past_clamp = been_past_clamp or step >= clamp
            history = True
        oracle.close()
        got.close()
    if bad == 0:
        failures.check(True, 'seek matrix', f'{len(targets)} targets, both directions')


def past_eof(path, frames, failures):
    oracle = Oracle(path)
    got = LocalDriver(path)
    oracle.seek(frames - 100)
    got.seek(frames - 100)
    a, b = oracle.read(1000), got.read(1000)
    failures.check(same(a, b) and a.shape[0] == 100, 'short read at EOF',
                   f'{a.shape[0]} vs {b.shape[0]} samples')
    a, b = oracle.read(1000), got.read(1000)
    failures.check(a.shape[0] == 0 and b.shape[0] == 0, 'read past EOF returns nothing')
    oracle.close()
    got.close()


def main():
    failures = Failures()
    for path in sys.argv[1:]:
        got = LocalDriver(path)
        frames, clamp, fast = got.frames, got._clamp, got._layout is not None
        got.close()
        oracle = Oracle(path)
        oracle_frames = oracle.frames
        oracle.close()

        print(f'\n{os.path.basename(path)}  '
              f'{"tail-scan path" if fast else "fell back to the whole-file scan"}')
        failures.check(frames == oracle_frames, 'frame count matches the scan',
                       f'{frames:,} vs {oracle_frames:,}')
        print(f'    clamp {clamp:,}, tail {max(0, oracle_frames - clamp):,} samples')

        seam(path, clamp, oracle_frames, failures)
        past_eof(path, oracle_frames, failures)
        seeks(path, clamp, oracle_frames, failures)
        # The whole file at the production chunk length, then the seam and the
        # tail again at read sizes chosen to land awkwardly on it.
        streamer_pattern(path, 500, failures)
        near = max(0, clamp - (1 << 21))
        for chunk in (1 << 20, 99991):
            sequential(path, oracle_frames, chunk, failures, near, ' over the seam')
        # Small enough that a crossing read's tail is under one MPEG frame.
        sequential(path, oracle_frames, 1153, failures, near, ' over the seam',
                   tolerate=1e-6)
        streamer_pattern(path, 20, failures, near, ' over the seam')

    print(f'\n{"PASS" if failures.count == 0 else f"{failures.count} FAILURES"}'
          f'{f" ({failures.notes} documented difference(s))" if failures.notes else ""}')
    sys.exit(1 if failures.count else 0)


if __name__ == '__main__':
    main()
