"""Does splitting a read change what libsndfile decodes from an mp3?

step1_seek.py showed that what mpg123 hands back at a given offset depends on
what was decoded before the seek. This asks the narrower and more awkward
question: with no seek at all, does reading a range as one call differ from
reading it as two?

If it does, then no two readers that segment their reads differently can be
bit-identical, and "identical to today's driver" only means anything against a
fixed access pattern -- which for buzzdetect is worker.py's seek-to-chunk-start
followed by one read of the chunk length.

The tail is not involved: every offset here is well inside the body.

Usage:  python3 step1_segmentation.py <file.mp3> [more.mp3 ...]
Run from engine/.
"""

import os
import sys

import numpy as np

sys.path.insert(0, '.')

import src.stream.drivers.mp3 as mp3     # noqa: E402
import soundfile as sf                   # noqa: E402


def scanned(path):
    shim = mp3._LengthHidingFile(path)
    return shim, sf.SoundFile(shim)


def main():
    total_split = 0
    for path in sys.argv[1:]:
        shim, track = scanned(path)
        sr, frames = track.samplerate, track.frames
        track.close()
        shim.close()

        base = frames // 3 // 1152 * 1152      # a frame-aligned starting point
        span = 1 << 18
        print(f'\n{os.path.basename(path)}  reading {span:,} samples from {base:,}')

        shim, track = scanned(path)
        track.seek(base)
        whole = track.read(span, dtype='float32')
        track.close()
        shim.close()

        for split in (1, 1151, 1152, 1153, 4096, span // 2, span // 2 + 1):
            shim, track = scanned(path)
            track.seek(base)
            first = track.read(split, dtype='float32')
            second = track.read(span - split, dtype='float32')
            track.close()
            shim.close()
            joined = np.concatenate((first, second))
            bad = np.nonzero(joined != whole)[0]
            if bad.size:
                total_split += 1
            print(f'  split at {split:>8,}: '
                  f'{"identical" if bad.size == 0 else f"{bad.size:,} differ, first at {bad[0]:,}, max {np.abs(joined - whole).max():.2e}"}')

    print(f'\n{total_split} split point(s) changed the samples')


if __name__ == '__main__':
    main()
