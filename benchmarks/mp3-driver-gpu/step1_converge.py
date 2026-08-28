"""Step 1 gate: does a decode started mid-stream converge on the real samples?

MP3's bit reservoir lets a frame borrow space from its predecessors, so a
decode that starts partway through a file produces wrong samples for the first
frame or two. The tail-scan plan rests on it then converging *exactly*. This
measures that: how many samples out of a fragment decode differ from a
continuous decode of the same file, and whether everything after them is
bit-identical.

Usage:  python3 step1_converge.py <file.mp3> [more.mp3 ...]
Run from engine/ (it imports src.stream.drivers.mp3).
"""

import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, '.')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mp3_frames import find_frame, parse_header      # noqa: E402
from src.stream.drivers.mp3 import LocalDriver, _id3v2_size   # noqa: E402

WINDOW = 1 << 19          # samples compared either side of the seam
CHAIN = 4                 # consecutive valid headers required of a boundary


class FragmentFile:
    """File-like over path[start:], withholding its length like the shim does.

    Same trick as _LengthHidingFile -- report 0 at the end so libsndfile has to
    scan -- but the scan now covers only the bytes from `start` on.
    """

    def __init__(self, path, start, stop=None):
        self._file = open(path, 'rb', buffering=1 << 20)
        size = os.path.getsize(path)
        self._start = start
        self._stop = size if stop is None else min(stop, size)
        self._length = max(0, self._stop - self._start)
        self._at_end = False
        self._file.seek(self._start)

    def read(self, count=-1):
        left = self._stop - self._file.tell()
        if count is None or count < 0:
            count = left
        return self._file.read(max(0, min(count, left)))

    def readinto(self, buffer):
        left = self._stop - self._file.tell()
        if left <= 0:
            return 0
        if len(buffer) <= left:
            return self._file.readinto(buffer)
        return self._file.readinto(memoryview(buffer)[:left])

    def tell(self):
        position = self._file.tell() - self._start
        if self._at_end and position == self._length:
            return 0
        return position

    def seek(self, offset, whence=os.SEEK_SET):
        self._at_end = (whence == os.SEEK_END and offset == 0)
        if whence == os.SEEK_END:
            position = self._file.seek(self._start + self._length + offset)
            return 0 if self._at_end else position - self._start
        if whence == os.SEEK_SET:
            return self._file.seek(offset + self._start) - self._start
        return self._file.seek(offset, whence) - self._start

    def close(self):
        self._file.close()


def audio_start(path):
    """Byte offset of the first MPEG frame, past any ID3v2 tag."""
    with open(path, 'rb') as f:
        skip = _id3v2_size(f)
        f.seek(skip)
        buf = f.read(1 << 16)
    header = find_frame(buf, 0, CHAIN)
    if header is None:
        raise RuntimeError(f'no MPEG frame found near byte {skip} of {path}')
    return skip + header['offset']


def boundary_at_or_after(path, target):
    """Validated frame boundary at or after `target`, with its parsed header."""
    with open(path, 'rb') as f:
        f.seek(target)
        buf = f.read(1 << 16)
    header = find_frame(buf, 0, CHAIN)
    if header is None:
        return None
    header = dict(header)
    header['offset'] += target
    return header


def frames_from(path, start):
    """Count MPEG frames, and their samples, from byte `start` to EOF.

    Pure header walking -- no decode -- so this is an exact byte->sample map
    for the region it covers. Returns (n_frames, n_samples).
    """
    n_frames = 0
    n_samples = 0
    at = start
    size = os.path.getsize(path)
    with open(path, 'rb') as f:
        f.seek(start)
        buf = f.read(1 << 22)
        base = start
        while True:
            index = at - base
            if index + 4 > len(buf):
                base = at
                f.seek(base)
                buf = f.read(1 << 22)
                index = 0
                if len(buf) < 4:
                    break
            header = parse_header(buf, index)
            if header is None:
                # Trailing tag (ID3v1/APE) or garbage: the audio has ended.
                break
            n_frames += 1
            n_samples += header['samples']
            at += header['size']
            if at >= size:
                break
    return n_frames, n_samples


def continuous(path, skip, count):
    """Decode from the very start, discard `skip` samples, return the next `count`.

    Deliberately not a seek: mpg123's seek does not warm the bit reservoir
    either, so seeking would beg the question this test is asking.
    """
    track = LocalDriver(path)
    step = 1 << 22
    left = skip
    while left > 0:
        got = track.read(min(step, left))
        if got.shape[0] == 0:
            break
        left -= got.shape[0]
    data = track.read(count)
    track.close()
    return data


def fragment(path, start, count):
    """Decode `count` samples from a fragment beginning at byte `start`."""
    shim = FragmentFile(path, start)
    track = sf.SoundFile(shim)
    data = track.read(count, dtype='float32')
    track.close()
    shim.close()
    return data


def convergence(reference, candidate):
    """(n_leading_bad, converged) for candidate against reference.

    Compares over the shorter of the two and reports the index just past the
    last mismatch, so `converged` says everything from there on is identical.
    """
    n = min(reference.shape[0], candidate.shape[0])
    if n == 0:
        return None, False
    bad = np.nonzero(reference[:n] != candidate[:n])[0]
    if bad.size == 0:
        return 0, True
    return int(bad[-1]) + 1, True


def probe(path, label, target_byte, start, sr, total_samples):
    header = boundary_at_or_after(path, target_byte)
    if header is None:
        print(f'  {label:<14} no frame boundary found near {target_byte:,}')
        return None
    offset = header['offset']

    _, samples_after = frames_from(path, offset)
    sample_offset = total_samples - samples_after
    if sample_offset < 0 or sample_offset + WINDOW > total_samples:
        print(f'  {label:<14} byte {offset:,} maps outside the file; skipped')
        return None

    reference = continuous(path, sample_offset, WINDOW)
    candidate = fragment(path, offset, WINDOW)

    if candidate.shape[0] < reference.shape[0]:
        print(f'  {label:<14} fragment short: {candidate.shape[0]:,} '
              f'vs {reference.shape[0]:,}')

    bad, converged = convergence(reference, candidate)
    n = min(reference.shape[0], candidate.shape[0])
    tail_equal = np.array_equal(reference[bad:n], candidate[bad:n]) if converged else False
    peak = float(np.max(np.abs(reference[:bad] - candidate[:bad]))) if bad else 0.0

    print(f'  {label:<14} byte {offset:>12,}  sample {sample_offset:>13,}  '
          f'differs in first {bad:>6,} samples ({bad / sr * 1000:7.2f} ms), '
          f'peak {peak:.4f}, then identical: {tail_equal}')
    return bad if tail_equal else None


def main():
    worst = 0
    failures = []
    for path in sys.argv[1:]:
        track = LocalDriver(path)
        sr, total = track.samplerate, track.frames
        track.close()

        start = audio_start(path)
        size = os.path.getsize(path)
        n_frames, n_samples = frames_from(path, start)

        estimate = sf.info(path).frames
        print(f'\n{os.path.basename(path)}  {size / 1e6:.0f} MB  @{sr} Hz')
        print(f'  scanned frames {total:,}   header-walk samples {n_samples:,}   '
              f'{"MATCH" if n_samples == total else "MISMATCH"}')
        print(f'  libsndfile estimate {estimate:,} '
              f'({100 * (total - estimate) / total:.3f}% short)')

        if n_samples != total:
            print('  header walk disagrees with the decode; byte->sample map is '
                  'not exact here, skipping probes')
            failures.append((path, 'map'))
            continue

        audio_bytes = size - start
        targets = [
            ('near start', start + audio_bytes // 100),
            ('mid file', start + audio_bytes // 2),
            ('before clamp', start + int(audio_bytes * 0.995)),
            ('tail 2MB', max(start, size - (2 << 20))),
        ]
        for label, target in targets:
            bad = probe(path, label, target, start, sr, total)
            if bad is None:
                failures.append((path, label))
            else:
                worst = max(worst, bad)

    print(f'\nworst-case leading garbage: {worst:,} samples')
    if failures:
        print('FAILURES:', failures)
        sys.exit(1)
    print('GATE PASSED: every fragment decode converged exactly.')


if __name__ == '__main__':
    main()
