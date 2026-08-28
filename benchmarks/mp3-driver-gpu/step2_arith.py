"""Can the true frame count be derived arithmetically instead of scanned?

libsndfile's estimate is wrong because it assumes every frame is as long as the
first, and the first is padded. But a CBR stream's frames average exactly
144 * bitrate / samplerate bytes -- the padding bit is what makes an integer
frame size track that non-integer average -- so the audio byte count divides by
it to give the frame count, and the whole mpg123 scan collapses into a
division.

That is only worth anything if it is exact. This checks it against the scan on
real files, and checks the byte offset it predicts for arbitrary frames lands
on a real header.

Usage:  python3 step2_arith.py <file.mp3> [more.mp3 ...]
Run from engine/.
"""

import os
import sys

import soundfile as sf

sys.path.insert(0, '.')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mp3_frames import find_frame, parse_header    # noqa: E402
from step1_converge import audio_start             # noqa: E402
from src.stream.drivers.mp3 import LocalDriver     # noqa: E402


def trailing_tag(path):
    """Bytes of ID3v1/APE trailer at the end of the file, if any."""
    size = os.path.getsize(path)
    with open(path, 'rb') as f:
        if size >= 128:
            f.seek(size - 128)
            if f.read(3) == b'TAG':
                return 128
        if size >= 32:
            f.seek(size - 32)
            if f.read(8) == b'APETAGEX':
                return 32
    return 0


def predict(path):
    start = audio_start(path)
    with open(path, 'rb') as f:
        f.seek(start)
        header = parse_header(f.read(8), 0)
    tag = trailing_tag(path)
    audio_bytes = os.path.getsize(path) - start - tag
    per_frame = 144.0 * header['bitrate'] * 1000 / header['samplerate']
    if header['version'] != 3:      # MPEG2/2.5 carry half a frame's samples
        per_frame /= 2
    n_frames = int(round(audio_bytes / per_frame))
    return start, header, per_frame, n_frames, audio_bytes


def frame_offset(start, per_frame, index):
    """Predicted byte offset of frame `index`, good to a byte in CBR."""
    return start + int(index * per_frame)


def main():
    for path in sys.argv[1:]:
        start, header, per_frame, n_frames, audio_bytes = predict(path)
        track = LocalDriver(path)
        scanned = track.frames
        sr = track.samplerate
        track.close()
        estimate = sf.info(path).frames
        predicted = n_frames * header['samples']

        print(f'\n{os.path.basename(path)}')
        print(f'  {header["bitrate"]} kbps {header["samplerate"]} Hz '
              f'{"mono" if header["channels"] == 1 else "stereo"}  '
              f'{per_frame:.4f} bytes/frame  audio {audio_bytes:,} bytes')
        print(f'  libsndfile estimate {estimate:>14,}')
        print(f'  scanned (oracle)    {scanned:>14,}')
        print(f'  predicted           {predicted:>14,}   '
              f'{"EXACT" if predicted == scanned else f"OFF BY {predicted - scanned:+,} samples"}')

        # Does the predicted offset of a frame actually land on a header?
        with open(path, 'rb') as f:
            worst = 0
            for fraction in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999):
                index = int((n_frames - 2) * fraction)
                guess = frame_offset(start, per_frame, index)
                f.seek(max(start, guess - 64))
                buf = f.read(512)
                found = find_frame(buf, 0, 4)
                if found is None:
                    print(f'    frame {index:>10,}: no header near {guess:,}')
                    continue
                actual = max(start, guess - 64) + found['offset']
                worst = max(worst, abs(actual - guess))
            print(f'  predicted offsets land within {worst} bytes of a real header')


if __name__ == '__main__':
    main()
