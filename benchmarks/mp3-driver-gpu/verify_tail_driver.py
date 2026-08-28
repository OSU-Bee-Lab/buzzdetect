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
    """
    approach = max(0, clamp - (1 << 20))
    cases = [
        ('read exactly to the clamp, then the rest', clamp - approach, frames - clamp),
        ('read one short of the clamp, then across', clamp - approach - 1, frames - clamp + 1),
        ('read one past the clamp, then the rest', clamp - approach + 1, frames - clamp - 1),
        ('one read over the whole seam', frames - approach, 0),
    ]
    for label, first, second in cases:
        oracle = Oracle(path)
        got = LocalDriver(path)
        oracle.seek(approach)
        got.seek(approach)
        a1, b1 = oracle.read(first), got.read(first)
        ok = failures.check(same(a1, b1), label + ' (first)', detail(a1, b1) if not same(a1, b1) else f'{a1.shape[0]:,} samples')
        if second and ok:
            # Compared against an oracle that read the whole range in one call,
            # not one that broke it where this case does. libsndfile's mp3
            # output depends on where reads are broken (step1_segmentation.py),
            # and the driver deliberately answers a read broken at the clamp as
            # though it had not been broken -- see mp3.py's _read_fragment.
            whole = Oracle(path)
            whole.seek(approach)
            a2 = whole.read(first + second)[first:]
            whole.close()
            b2 = got.read(second)
            failures.check(same(a2, b2), label + ' (second)',
                           detail(a2, b2) if not same(a2, b2) else f'{a2.shape[0]:,} samples')
        oracle.close()
        got.close()


def seeks(path, clamp, frames, failures):
    sr = sf.info(path).samplerate
    targets = [0, sr, clamp // 2, clamp - sr, clamp - 1, clamp, clamp + 1,
               (clamp + frames) // 2, frames - sr, frames - 1, frames]
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
        been_in_tail = False
        for step in (target, targets[(i + 3) % len(targets)]):
            pa, pb = oracle.seek(step), got.seek(step)
            a, b = oracle.read(window), got.read(window)
            if pa == pb and same(a, b):
                been_in_tail = been_in_tail or step >= clamp
                continue
            worst = (np.abs(a - b).max()
                     if a.shape == b.shape and a.shape[0] else float('inf'))
            if pa == pb and been_in_tail and step < clamp and worst <= 1e-6:
                # Documented: reading the tail advances the old driver's one
                # decoder and not this one's body track, so a seek back into the
                # body afterwards starts from a different decode history -- and
                # what a seek returns depends on that history (step1_seek.py).
                # The streamer reads a file's chunks in order and never goes
                # back over the seam.
                failures.note('seek matrix',
                              f'seek({step:,}) after reading the tail: '
                              f'{detail(a, b)} (within the documented 1e-06)')
                been_in_tail = been_in_tail or step >= clamp
                continue
            bad += 1
            if bad == 1:
                failures.check(False, 'seek matrix',
                               f'seek({step:,}) -> {pa:,} vs {pb:,}; '
                               f'{detail(a, b) if a.shape == b.shape else f"shapes {a.shape} {b.shape}"}')
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
