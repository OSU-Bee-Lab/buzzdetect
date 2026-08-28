"""Tests for the mp3 driver's tail-scan read path.

Run from the engine directory:

    .venv/bin/python3 tests/test_mp3_driver.py

No test runner and no new dependency: this is a script that returns non-zero if
anything fails. The fixtures beside it are small excerpts of a real recording
(tests/make_fixtures.py cuts them); the corpus is not needed.

The oracle throughout is the driver's own fallback path -- ordinary libsndfile
opened through the shim that withholds the file's length, which is what the
driver did before it learned to read the body plainly, and what produced the
results already sitting in people's output folders.

Two differences from that oracle are deliberate and are asserted as bounds
rather than equalities; both are explained in mp3.py and measured in
benchmarks/mp3-driver-gpu/STEP1_RESULT.md. In short, libsndfile's mp3 output
depends on where the caller seeks and on where it breaks its reads, so the
driver can only be identical to it for a given access pattern -- and it is, for
the one the streamer uses.
"""

import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, '.')

import src.stream.drivers.mp3 as mp3            # noqa: E402
from src.stream.drivers.mp3 import LocalDriver  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures')
TOLERATED = 1e-6            # the documented bound on the two known differences

failures = []
checks = 0


def check(ok, label, detail=''):
    global checks
    checks += 1
    if not ok:
        failures.append(f'{label}  {detail}')
        print(f'  [FAIL] {label}  {detail}')
    return ok


def fixture(name):
    return os.path.join(FIXTURES, name)


class Oracle:
    """The driver before this change: whole-file shim, mpg123's own scan."""

    def __init__(self, path):
        self._shim = None
        try:
            self._shim = mp3._LengthHidingFile(path)
            track = sf.SoundFile(self._shim)
            if not 0 < track.frames < mp3._SF_COUNT_MAX:
                track.close()
                raise ValueError('implausible scan')
        except Exception:
            if self._shim is not None:
                self._shim.close()
                self._shim = None
            track = sf.SoundFile(path)
        self._track = track
        self.frames = track.frames
        self.samplerate = track.samplerate
        self.channels = track.channels

    def read(self, n):
        return self._track.read(n, dtype='float32')

    def seek(self, n):
        return self._track.seek(n)

    def close(self):
        self._track.close()
        if self._shim is not None:
            self._shim.close()


def difference(a, b):
    if a.shape != b.shape:
        return float('inf'), f'shapes {a.shape} vs {b.shape}'
    if a.shape[0] == 0:
        return 0.0, ''
    bad = np.nonzero(a != b)[0]
    if bad.size == 0:
        return 0.0, ''
    worst = float(np.abs(a - b).max())
    return worst, f'{bad.size:,} of {a.shape[0]:,} differ, first at {bad[0]:,}, max {worst:.2e}'


# The frame arithmetic
#
def test_layout():
    print('layout and frame boundaries')
    for name in ('truncating.mp3', 'tagged.mp3', 'generous.mp3', 'stereo.mp3'):
        path = fixture(name)
        layout = mp3.read_layout(path)
        if not check(layout is not None, f'{name}: layout found'):
            continue

        oracle = Oracle(path)
        check(layout.frames == oracle.frames, f'{name}: frame count matches the scan',
              f'{layout.frames:,} vs {oracle.frames:,}')
        check(layout.estimate == sf.info(path).frames,
              f'{name}: libsndfile\'s estimate is predicted',
              f'{layout.estimate:,} vs {sf.info(path).frames:,}')
        oracle.close()

        # Every offset the layout names must be one libsndfile will open, and
        # must be the frame that was asked for.
        for fraction in (0.0, 0.1, 0.5, 0.9):
            index = int((layout.n_frames - mp3._CHAIN - 1) * fraction)
            offset = layout.frame_offset(path, index)
            if not check(offset is not None, f'{name}: frame {index} located'):
                continue
            shim = mp3._LengthHidingFile(path, skip=offset)
            try:
                track = sf.SoundFile(shim)
                opened = True
                track.close()
            except Exception as e:
                opened = False
                check(False, f'{name}: libsndfile opens frame {index} as a fragment',
                      f'{type(e).__name__}: {e}')
            shim.close()
            if opened:
                check(True, f'{name}: frame {index} opens as a fragment')

        # Searching from anywhere in the frame before it must find the same one.
        index = layout.n_frames // 2
        offset = layout.frame_offset(path, index)
        with open(path, 'rb') as f:
            f.seek(offset - int(layout.bytes_per_frame))
            buf = f.read(int(layout.bytes_per_frame) * (mp3._CHAIN + 2))
        found = {mp3._find_frame(buf, start)['offset'] + offset - int(layout.bytes_per_frame)
                 for start in range(1, int(layout.bytes_per_frame))}
        check(found == {offset}, f'{name}: every search in the frame before finds it',
              f'found {sorted(found)[:4]} not just {offset}')


def test_no_layout():
    print('files the fast path must decline')
    path = fixture('tiny.mp3')
    driver = LocalDriver(path)
    check(driver._layout is None, 'tiny.mp3: falls back to the scan')
    oracle = Oracle(path)
    check(driver.frames == oracle.frames, 'tiny.mp3: same length as the scan',
          f'{driver.frames:,} vs {oracle.frames:,}')
    worst, detail = difference(oracle.read(1 << 20), driver.read(1 << 20))
    check(worst == 0.0, 'tiny.mp3: same samples as the scan', detail)
    oracle.close()
    driver.close()


def test_no_tail():
    print('files with nothing past the clamp')
    for name in ('generous.mp3', 'tagged.mp3'):
        path = fixture(name)
        driver = LocalDriver(path)
        check(driver.frames <= driver._clamp,
              f'{name}: libsndfile already reaches the end',
              f'frames {driver.frames:,}, clamp {driver._clamp:,}')
        got = driver.read(driver.frames + 1000)
        check(got.shape[0] == driver.frames, f'{name}: reads exactly its length',
              f'{got.shape[0]:,} of {driver.frames:,}')
        check(driver._tail is None, f'{name}: never opens a fragment')
        driver.close()


# The read path
#
def test_whole_file(name):
    path = fixture(name)
    driver = LocalDriver(path)
    clamp, frames = driver._clamp, driver.frames
    driver.close()

    per_frame = 1152
    sizes = [1 << 16, 1000, clamp, clamp + 1, clamp - 1, frames]
    for chunk in sizes:
        if chunk <= 0:
            continue
        # A caller that breaks a read within a frame of the clamp gets the
        # unbroken decode from this driver and the broken one from libsndfile,
        # which differ in the last digit. Every other break must be exact.
        landing = clamp % chunk
        near_seam = min(landing, chunk - landing) <= per_frame
        bound = TOLERATED if near_seam else 0.0
        oracle = Oracle(path)
        got = LocalDriver(path)
        position = 0
        worst = 0.0
        detail = ''
        while True:
            a, b = oracle.read(chunk), got.read(chunk)
            step, why = difference(a, b)
            if step > worst:
                worst, detail = step, f'at sample {position:,}: {why}'
            if a.shape[0] == 0:
                break
            position += a.shape[0]
        oracle.close()
        got.close()
        check(position == frames, f'{name}: read the whole file in {chunk:,} chunks',
              f'{position:,} of {frames:,}')
        check(worst <= bound, f'{name}: '
              f'{"identical" if bound == 0 else "within the documented bound"} '
              f'in {chunk:,} chunks', detail)


def test_streamer_pattern(name):
    """seek(chunk start) then one read per chunk, as worker.py does."""
    path = fixture(name)
    for chunk in (11025, 44100, 32768):
        oracle = Oracle(path)
        got = LocalDriver(path)
        position = 0
        worst = 0.0
        detail = ''
        while position < got.frames:
            oracle.seek(position)
            got.seek(position)
            step, why = difference(oracle.read(chunk), got.read(chunk))
            if step > worst:
                worst, detail = step, f'at sample {position:,}: {why}'
            position += chunk
        oracle.close()
        got.close()
        check(worst == 0.0, f'{name}: streamer pattern, {chunk:,} sample chunks', detail)


def test_seams(name):
    """Reads placed deliberately on and across libsndfile's clamp."""
    path = fixture(name)
    driver = LocalDriver(path)
    clamp, frames = driver._clamp, driver.frames
    driver.close()
    if frames <= clamp:
        return

    for first in (clamp - 1, clamp, clamp + 1, (clamp + frames) // 2):
        second = frames - first
        oracle = Oracle(path)
        got = LocalDriver(path)
        worst, detail = difference(oracle.read(first), got.read(first))
        check(worst == 0.0, f'{name}: read of {first:,} up to the seam', detail)

        rest = got.read(second)
        check(rest.shape[0] == second, f'{name}: remainder after a break at {first:,} '
              f'is the right length', f'{rest.shape[0]:,} of {second:,}')
        worst, detail = difference(oracle.read(second), rest)
        check(worst <= TOLERATED, f'{name}: remainder after a break at {first:,}', detail)
        oracle.close()
        got.close()

    # The documented behaviour of a break at the clamp itself: the driver
    # answers as though the read had not been broken there, because
    # libsndfile's own answer depends on where the break falls.
    unbroken = Oracle(path)
    expected = unbroken.read(frames)
    unbroken.close()
    got = LocalDriver(path)
    joined = np.concatenate((got.read(clamp), got.read(frames - clamp)))
    got.close()
    worst, detail = difference(expected, joined)
    check(worst == 0.0, f'{name}: a break at the clamp reads as though unbroken', detail)


def test_seeks(name):
    path = fixture(name)
    driver = LocalDriver(path)
    clamp, frames, sr = driver._clamp, driver.frames, driver.samplerate
    driver.close()

    targets = [t for t in (0, sr // 10, clamp // 2, clamp - 1, clamp, clamp + 1,
                           (clamp + frames) // 2, frames - 1, frames)
               if 0 <= t <= frames]
    window = 1 << 15
    per_frame = 1152
    for i, target in enumerate(targets):
        oracle = Oracle(path)
        got = LocalDriver(path)
        been_past_clamp = False
        history = False
        for step in (target, targets[(i + 3) % len(targets)]):
            pa, pb = oracle.seek(step), got.seek(step)
            check(pa == pb, f'{name}: seek({step:,}) reports the same position',
                  f'{pa:,} vs {pb:,}')
            a, b = oracle.read(window), got.read(window)
            check(a.shape == b.shape, f'{name}: seek({step:,}) then read returns as much',
                  f'{a.shape} vs {b.shape}')
            worst, detail = difference(a, b)
            # A seek from a reader that has decoded nothing yet must be exact.
            # Once it has, it need only be close: what libsndfile returns after
            # a seek depends on what was decoded before it, and near the seam
            # the two readers are no longer carrying the same history -- the
            # old driver has one decoder for the whole file, this one hands the
            # tail to a second. The streamer reads a file's chunks in order and
            # so never asks for this.
            near_seam = step >= clamp - 2 * per_frame
            bound = TOLERATED if (history and (near_seam or been_past_clamp)) else 0.0
            check(worst <= bound, f'{name}: seek({step:,}) then read', detail)
            been_past_clamp = been_past_clamp or step >= clamp
            history = True
        oracle.close()
        got.close()


def test_eof(name):
    path = fixture(name)
    oracle = Oracle(path)
    got = LocalDriver(path)
    frames = got.frames
    oracle.seek(frames - 100)
    got.seek(frames - 100)
    a, b = oracle.read(1000), got.read(1000)
    check(a.shape[0] == 100 and b.shape[0] == 100, f'{name}: a read straddling EOF stops there',
          f'{a.shape[0]} vs {b.shape[0]}')
    worst, detail = difference(a, b)
    check(worst == 0.0, f'{name}: the last samples are identical', detail)
    a, b = oracle.read(1000), got.read(1000)
    check(a.shape[0] == 0 and b.shape[0] == 0, f'{name}: reading past EOF returns nothing',
          f'{a.shape[0]} vs {b.shape[0]}')
    oracle.close()
    got.close()


def test_out_buffer(name):
    """Reads into a caller's buffer must match reads that allocate their own."""
    path = fixture(name)
    driver = LocalDriver(path)
    frames, channels = driver.frames, driver.channels
    driver.close()

    for start in (0, max(0, frames - 40000)):
        plain = LocalDriver(path)
        buffered = LocalDriver(path)
        plain.seek(start)
        buffered.seek(start)
        want = 30000
        shape = (want,) if channels == 1 else (want, channels)
        out = np.zeros(shape, dtype='float32')
        a = plain.read(want)
        b = buffered.read(want, out=out)
        worst, detail = difference(a, b)
        filled, _ = difference(a, out[:a.shape[0]])
        check(worst == 0.0 and filled == 0.0,
              f'{name}: read into a caller buffer at {start:,}', detail)
        plain.close()
        buffered.close()


def test_helper_equivalence(name):
    """The helper process must return exactly what in-process returns."""
    path = fixture(name)
    os.environ[mp3.ENV_HELPERS] = 'always'
    try:
        local = LocalDriver(path)
        remote = mp3.Driver(path)
        if not check(remote._helper is not None,
                     f'{name}: a helper process started (test is meaningless otherwise)'):
            local.close()
            remote.close()
            return
        check((remote.samplerate, remote.channels, remote.frames)
              == (local.samplerate, local.channels, local.frames),
              f'{name}: helper reports the same metadata')
        worst = 0.0
        detail = ''
        position = 0
        while True:
            a, b = local.read(1 << 16), remote.read(1 << 16)
            step, why = difference(a, b)
            if step > worst:
                worst, detail = step, f'at sample {position:,}: {why}'
            if a.shape[0] == 0:
                break
            position += a.shape[0]
        check(worst == 0.0, f'{name}: helper reads are identical to in-process', detail)
        local.close()
        remote.close()
    finally:
        os.environ.pop(mp3.ENV_HELPERS, None)


def main():
    test_layout()
    test_no_layout()
    test_no_tail()
    for name in ('truncating.mp3', 'stereo.mp3'):
        print(name)
        test_whole_file(name)
        test_streamer_pattern(name)
        test_seams(name)
        test_seeks(name)
        test_eof(name)
        test_out_buffer(name)
        test_helper_equivalence(name)

    print(f'\n{checks} checks, '
          f'{"all passed" if not failures else f"{len(failures)} FAILED"}')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()
