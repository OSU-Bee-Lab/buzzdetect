"""Step 1, the case that actually matters: decoding the missing tail.

The plan only ever wants the fragment to serve the audio past libsndfile's
clamp -- 34 s on a 5.6 h file. So the question is not "does an arbitrary
mid-stream decode converge" (measured: it converges to ~1e-7 and is exactly
identical only about half the time, with no global parity rule) but "is the
*tail* bit-identical, and does more overlap help".

For each file this decodes the tail two ways -- continuously from the start,
which is today's driver and therefore the oracle, and from a fragment beginning
K frames before the clamp -- and reports whether they agree exactly.

Usage:  python3 step1_tail.py <file.mp3> [more.mp3 ...]
Run from engine/.
"""

import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, '.')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from step1_converge import FragmentFile, frames_from   # noqa: E402
from step1_converge import audio_start                 # noqa: E402
from mp3_frames import find_frame                      # noqa: E402
from src.stream.drivers.mp3 import LocalDriver         # noqa: E402

STEP = 1 << 22
SAMPLES_PER_FRAME = 1152
# Both parities at each depth. The residual alternates with the parity of the
# start frame -- of two adjacent boundaries one reproduces the continuous
# decode exactly and the other does not -- so a sweep of same-parity overlaps
# says nothing about whether the depth mattered.
OVERLAPS = (6, 7, 8, 9, 32, 33, 128, 129, 512, 513)


def boundary_with_frames_after(path, target_frames, low):
    """Byte offset of the frame boundary that has `target_frames` frames after it.

    Walks in from an estimate rather than counting from the file's start: the
    tail is the only region this is ever asked about, so frames_from() only
    ever reads a few MB.
    """
    size = os.path.getsize(path)
    guess = max(low, size - target_frames * 157 - (1 << 16))
    for _ in range(12):
        with open(path, 'rb') as f:
            f.seek(guess)
            buf = f.read(1 << 16)
        header = find_frame(buf, 0, 4)
        if header is None:
            return None
        offset = guess + header['offset']
        frames_after, _ = frames_from(path, offset)
        if frames_after == target_frames:
            return offset
        # Frames are ~157 bytes; step by the shortfall and re-snap.
        guess = max(low, min(size - 4, offset + (frames_after - target_frames) * 157))
    return None


def oracle_tail(path, clamp):
    """The samples today's driver produces from `clamp` to the end of the file."""
    track = LocalDriver(path)
    position = 0
    while position < clamp:
        got = track.read(min(STEP, clamp - position))
        if got.shape[0] == 0:
            break
        position += got.shape[0]
    parts = []
    while True:
        got = track.read(STEP)
        if got.shape[0] == 0:
            break
        parts.append(got)
    track.close()
    return np.concatenate(parts) if parts else np.zeros(0, dtype='float32')


def fragment_tail(path, offset, discard, want):
    """Decode a fragment at `offset`, drop `discard` samples, return `want`."""
    shim = FragmentFile(path, offset)
    track = sf.SoundFile(shim)
    position = 0
    while position < discard:
        got = track.read(min(STEP, discard - position), dtype='float32')
        if got.shape[0] == 0:
            break
        position += got.shape[0]
    parts = []
    left = want
    while left > 0:
        got = track.read(min(STEP, left), dtype='float32')
        if got.shape[0] == 0:
            break
        parts.append(got)
        left -= got.shape[0]
    track.close()
    shim.close()
    return np.concatenate(parts) if parts else np.zeros(0, dtype='float32')


def main():
    for path in sys.argv[1:]:
        track = LocalDriver(path)
        sr, total = track.samplerate, track.frames
        track.close()
        clamp = sf.info(path).frames
        missing = total - clamp
        start = audio_start(path)

        print(f'\n{os.path.basename(path)}  {total / sr / 3600:.2f} h  '
              f'missing {missing:,} samples ({missing / sr:.1f} s)')
        if missing <= 0:
            print('  no tail to recover; the estimate is already exact')
            continue

        reference = oracle_tail(path, clamp)
        print(f'  oracle tail {reference.shape[0]:,} samples')

        # The clamp lands inside a frame; the fragment must begin on a boundary
        # at or before the frame containing it.
        clamp_frame = clamp // SAMPLES_PER_FRAME
        total_frames = total // SAMPLES_PER_FRAME

        for overlap in OVERLAPS:
            first_frame = clamp_frame - overlap
            if first_frame < 0:
                continue
            frames_after = total_frames - first_frame
            offset = boundary_with_frames_after(path, frames_after, start)
            if offset is None:
                print(f'  overlap {overlap:>5}: could not locate the boundary')
                continue
            discard = clamp - first_frame * SAMPLES_PER_FRAME
            candidate = fragment_tail(path, offset, discard, reference.shape[0])
            n = min(candidate.shape[0], reference.shape[0])
            if n == 0:
                print(f'  overlap {overlap:>5}: fragment produced nothing')
                continue
            bad = np.nonzero(reference[:n] != candidate[:n])[0]
            residual = (0.0 if bad.size == 0
                        else float(np.abs(reference[:n] - candidate[:n]).max()))
            print(f'  overlap {overlap:>5} frames ({overlap * SAMPLES_PER_FRAME / sr:6.2f} s): '
                  f'len {candidate.shape[0]:,}/{reference.shape[0]:,}  '
                  f'mismatched {bad.size:>9,}  max {residual:.2e}  '
                  f'{"IDENTICAL" if bad.size == 0 and n == reference.shape[0] else ""}')


if __name__ == '__main__':
    main()
