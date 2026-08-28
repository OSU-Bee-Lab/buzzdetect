"""Is a plain open bit-identical to the scanned one, chunk for chunk?

The tail-scan plan reads the body through a plain SoundFile, so the body has to
decode to exactly what today's driver decodes. A continuous read is not enough
of a test: the streamer seeks to the start of every chunk (worker.py
queue_chunk), and step1_seek.py showed mpg123's output at a given offset
depends on what was decoded before the seek. So this compares the two under the
streamer's actual access pattern.

Usage:  python3 step1_body.py <chunk_seconds> <file.mp3> [more.mp3 ...]
Run from engine/.
"""

import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, '.')

from src.stream.drivers.mp3 import LocalDriver     # noqa: E402


def main():
    chunk_seconds = float(sys.argv[1])
    for path in sys.argv[2:]:
        scanned = LocalDriver(path)
        plain = sf.SoundFile(path)
        sr = scanned.samplerate
        clamp = plain.frames
        chunk = int(chunk_seconds * sr)

        bad_chunks = 0
        worst = 0.0
        checked = 0
        position = 0
        while position < clamp:
            want = min(chunk, clamp - position)
            scanned.seek(position)
            plain.seek(position)
            a = scanned.read(want)
            b = plain.read(want, dtype='float32')
            if a.shape != b.shape or not np.array_equal(a, b):
                bad_chunks += 1
                n = min(a.shape[0], b.shape[0])
                worst = max(worst, float(np.abs(a[:n] - b[:n]).max()))
            checked += a.shape[0]
            position += want

        scanned.close()
        plain.close()
        print(f'{os.path.basename(path)}  {checked:,} samples in '
              f'{chunk_seconds:g} s chunks: '
              f'{"IDENTICAL" if bad_chunks == 0 else f"{bad_chunks} chunks differ, max {worst:.2e}"}')


if __name__ == '__main__':
    main()
