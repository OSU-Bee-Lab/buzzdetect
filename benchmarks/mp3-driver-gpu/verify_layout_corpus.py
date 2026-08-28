"""Does the frame arithmetic hold on every mp3 the server has?

The driver replaces mpg123's length scan with a division: a constant-bitrate
stream's frames average exactly 144 * bitrate / samplerate bytes, so the audio
byte count divides by that to give the frame count. This checks that claim at
two levels.

Cheap, on every file: does `read_layout` accept it, and what stream shape is it?
`read_layout` already validates itself -- it predicts frame offsets across the
file and requires a real header at each, requires the last frame to end where
the audio does, and requires libsndfile's own estimate to come out exactly -- so
a file it accepts has passed all of that.

Expensive, on the files small enough to afford it (`--scan-under`): does the
predicted frame count equal what mpg123's full scan measures? That is the claim
itself, checked against the only ground truth there is.

    python3 verify_layout_corpus.py <root> [--scan-under BYTES] [--per-dir N]

Run from engine/.
"""

import argparse
import os
import sys
from collections import Counter, defaultdict

import soundfile as sf

sys.path.insert(0, '.')

import src.stream.drivers.mp3 as mp3     # noqa: E402


def shape_of(layout):
    return (f'{"MPEG1" if layout.samples_per_frame == 1152 else "MPEG2"} '
            f'{layout.bytes_per_frame:.3f} B/frame {layout.samplerate} Hz '
            f'{"mono" if layout.channels == 1 else "stereo"}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('--scan-under', type=float, default=60e6,
                        help='scan files smaller than this many bytes (0 for none)')
    parser.add_argument('--per-dir', type=int, default=0,
                        help='at most this many files per directory (0 for all)')
    args = parser.parse_args()

    shapes = Counter()
    declined = Counter()
    scanned = 0
    mismatches = []
    examples = defaultdict(str)

    for folder, _, names in os.walk(args.root):
        mp3s = sorted(n for n in names
                      if n.lower().endswith('.mp3') and not n.startswith('._'))
        if args.per_dir:
            mp3s = mp3s[:args.per_dir]
        for name in mp3s:
            path = os.path.join(folder, name)
            try:
                layout = mp3.read_layout(path)
            except Exception as e:
                declined[f'{type(e).__name__}'] += 1
                continue
            if layout is None:
                size = os.path.getsize(path)
                declined['no layout (<8 KB)' if size < 8192 else 'no layout'] += 1
                continue

            shape = shape_of(layout)
            shapes[shape] += 1
            examples[shape] = examples[shape] or path

            if args.scan_under and os.path.getsize(path) < args.scan_under:
                try:
                    driver = mp3.LocalDriver(path, layout=None)   # force the scan
                    truth = driver.frames
                    driver.close()
                except Exception as e:
                    mismatches.append((path, f'scan failed: {type(e).__name__}: {e}'))
                    continue
                scanned += 1
                if truth != layout.frames:
                    mismatches.append(
                        (path, f'predicted {layout.frames:,}, scanned {truth:,}, '
                               f'off by {layout.frames - truth:+,} '
                               f'({(layout.frames - truth) / layout.samples_per_frame:+.2f} frames)'))

    print(f'\nstream shapes accepted onto the fast path ({sum(shapes.values())} files):')
    for shape, count in shapes.most_common():
        print(f'  {count:>6}  {shape}')
        print(f'          e.g. {examples[shape]}')
    if declined:
        print(f'\ndeclined, and read the old way instead ({sum(declined.values())} files):')
        for reason, count in declined.most_common():
            print(f'  {count:>6}  {reason}')

    print(f'\nchecked against mpg123\'s own scan: {scanned} files, '
          f'{len(mismatches)} disagreed')
    for path, why in mismatches[:20]:
        print(f'  {os.path.basename(path)}: {why}')
    sys.exit(1 if mismatches else 0)


if __name__ == '__main__':
    main()
