"""Does an exactly-converging fragment start *stay* exact?

The alternation measured in step1_parity/step1_sweep has period 2 in the start
frame: of two adjacent frame boundaries, one reproduces the continuous decode
bit for bit and the other leaves a ~1e-7 residual forever. A driver can find
which by comparing an overlap it can decode both ways. That is only sound if
being in phase is durable -- if a start that agrees over the overlap keeps
agreeing for the whole tail.

This decodes a long span from both adjacent boundaries and reports where, if
anywhere, each first departs from the continuous decode. Both references are
fresh decodes from the start of the file, never seeks, so nothing here depends
on mpg123's seek behaviour.

Usage:  python3 step1_durability.py <file.mp3> [span_samples]
Run from engine/.
"""

import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, '.')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from step1_converge import (FragmentFile, audio_start,   # noqa: E402
                            boundary_at_or_after, frames_from)
from step1_parity import consecutive_boundaries          # noqa: E402
from src.stream.drivers.mp3 import LocalDriver           # noqa: E402

STEP = 1 << 22


def main():
    path = sys.argv[1]
    span = int(sys.argv[2]) if len(sys.argv) > 2 else 40_000_000

    track = LocalDriver(path)
    sr, total = track.samplerate, track.frames
    track.close()
    start = audio_start(path)
    size = os.path.getsize(path)

    header = boundary_at_or_after(path, start + int((size - start) * 0.30))
    offsets = consecutive_boundaries(path, header['offset'], 2)

    print(f'{os.path.basename(path)}  span {span:,} samples '
          f'({span / sr / 3600:.2f} h)')

    for offset in offsets:
        _, after = frames_from(path, offset)
        base = total - after

        reference = LocalDriver(path)
        position = 0
        while position < base:
            got = reference.read(min(STEP, base - position))
            if got.shape[0] == 0:
                break
            position += got.shape[0]

        shim = FragmentFile(path, offset)
        candidate = sf.SoundFile(shim)

        seen = 0
        first_bad = None
        worst = 0.0
        while seen < span:
            want = min(STEP, span - seen)
            a = reference.read(want)
            b = candidate.read(want, dtype='float32')
            n = min(a.shape[0], b.shape[0])
            if n == 0:
                break
            diff = np.abs(a[:n] - b[:n])
            if seen == 0:
                diff = diff[4096:]      # decoder warm-up
            if diff.size:
                worst = max(worst, float(diff.max()))
                if first_bad is None:
                    bad = np.nonzero(diff)[0]
                    if bad.size:
                        first_bad = seen + int(bad[0]) + (4096 if seen == 0 else 0)
            seen += n

        reference.close()
        candidate.close()
        shim.close()

        where = ('never' if first_bad is None
                 else f'{first_bad:,} ({first_bad / sr:.1f} s in)')
        print(f'  byte {offset:>12,}  frame {base // 1152:>10,}  '
              f'first departure {where:>22}  max residual {worst:.2e}  '
              f'over {seen:,} samples')


if __name__ == '__main__':
    main()
