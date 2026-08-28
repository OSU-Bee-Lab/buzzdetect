"""Files the fast read path must decline, and decline gracefully.

The tail-scan path assumes a constant-bitrate MPEG Layer III stream whose frame
count can be worked out from its size. The corpora contain files that are none
of those things -- macOS resource forks, 2 KB fragments of recordings that died
as they started, stereo files at other bitrates -- and every one of them has to
either open on the fast path and read correctly, or fall back to the scan, and
in both cases agree with what the old driver did.

Usage:  python3 verify_degenerate.py <file.mp3> [more.mp3 ...]
Run from engine/.
"""

import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, '.')

import src.stream.drivers.mp3 as mp3            # noqa: E402
from src.stream.drivers.mp3 import LocalDriver  # noqa: E402


def old_driver(path):
    """What LocalDriver did before the tail-scan path: scan, or plain if it fails."""
    shim = None
    try:
        shim = mp3._LengthHidingFile(path)
        track = sf.SoundFile(shim)
        if 0 < track.frames < mp3._SF_COUNT_MAX:
            return shim, track
        track.close()
    except Exception:
        pass
    if shim is not None:
        shim.close()
    return None, sf.SoundFile(path)


def main():
    failures = 0
    for path in sys.argv[1:]:
        name = os.path.basename(path)
        size = os.path.getsize(path)

        try:
            shim, oracle = old_driver(path)
        except Exception as e:
            oracle = None
            reason = f'{type(e).__name__}: {e}'

        try:
            got = LocalDriver(path)
        except Exception as e:
            if oracle is None:
                print(f'  [ok  ] {name:<28} {size:>10,} B  both refuse it ({reason})')
            else:
                failures += 1
                print(f'  [FAIL] {name:<28} {size:>10,} B  old driver opened it, new one raised '
                      f'{type(e).__name__}: {e}')
                oracle.close()
            continue

        if oracle is None:
            failures += 1
            print(f'  [FAIL] {name:<28} {size:>10,} B  new driver opened a file the old one refused')
            got.close()
            continue

        path_taken = 'tail-scan' if got._layout is not None else 'scan'
        ok = got.frames == oracle.frames and got.samplerate == oracle.samplerate
        a = oracle.read(1 << 20, dtype='float32')
        b = got.read(1 << 20)
        ok = ok and a.shape == b.shape and np.array_equal(a, b)
        if ok:
            print(f'  [ok  ] {name:<28} {size:>10,} B  {path_taken:<9} '
                  f'{got.frames:>12,} frames, first {a.shape[0]:,} samples identical')
        else:
            failures += 1
            print(f'  [FAIL] {name:<28} {size:>10,} B  {path_taken:<9} '
                  f'frames {got.frames:,} vs {oracle.frames:,}, shapes {b.shape} vs {a.shape}')
        oracle.close()
        if shim is not None:
            shim.close()
        got.close()

    print(f'\n{"PASS" if failures == 0 else f"{failures} FAILURES"}')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
