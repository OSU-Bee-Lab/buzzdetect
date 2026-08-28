"""Cut the small mp3s that tests/test_mp3_driver.py runs against.

The fixtures are excerpts of a real field recording rather than anything
synthesised, because the behaviour under test is libsndfile's and it only shows
up on the real thing. They are cut at chosen frame boundaries so that each one
exhibits a specific shape:

  truncating.mp3  starts on a padded frame, so libsndfile's length estimate
                  comes out short and the file has an unreachable tail -- the
                  case the driver exists for
  generous.mp3    starts on an unpadded frame, so the estimate comes out long
                  and there is no tail; the driver must not open a fragment
  tagged.mp3      truncating.mp3 behind the recorder's own ID3v2 tag
  stereo.mp3      a stereo excerpt at a different bitrate
  tiny.mp3        a few frames, shorter than the fragment window

Regenerating them needs the corpus; running the tests does not.

    python3 tests/make_fixtures.py <mono.mp3> <stereo.mp3>
"""

import os
import sys

import soundfile as sf

sys.path.insert(0, '.')

import src.stream.drivers.mp3 as mp3     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, 'fixtures')

MONO_FRAMES = 1914          # ~300 KB, and a tail of about three frames
STEREO_FRAMES = 240
TINY_FRAMES = 6


def frames_from(path, start, count):
    """Bytes spanning `count` frames from the boundary at `start`."""
    with open(path, 'rb') as f:
        f.seek(start)
        buf = f.read(count * 512 + (1 << 13))
    at = 0
    header = mp3._parse_header(buf, 0)
    if header is None:
        raise SystemExit(f'{path}: byte {start} is not a frame header')
    for _ in range(count):
        header = mp3._parse_header(buf, at)
        if header is None:
            raise SystemExit(f'{path}: ran out of frames at {start + at}')
        at += header['size']
    return buf[:at]


def first_frame_with(path, padding):
    """The first frame boundary whose frame has (or lacks) the padding bit.

    The padding bit is the whole story of this driver: libsndfile assumes every
    frame is as long as the first, so a fixture that starts on a padded frame
    under-reports its length and one that starts on an unpadded frame
    over-reports it.
    """
    with open(path, 'rb') as f:
        skip = mp3._id3v2_size(f)
        f.seek(skip)
        buf = f.read(1 << 16)
    header = mp3._find_frame(buf)
    at = header['offset']
    while at < len(buf) - 4:
        header = mp3._parse_header(buf, at)
        if header is None:
            break
        if ((buf[at + 2] >> 1) & 1) == padding:
            return skip + at, header
        at += header['size']
    raise SystemExit(f'{path}: no frame with padding={padding} near the start')


def id3_tag(path):
    with open(path, 'rb') as f:
        size = mp3._id3v2_size(f)
        f.seek(0)
        return f.read(size)


def write(name, payload):
    path = os.path.join(FIXTURES, name)
    with open(path, 'wb') as f:
        f.write(payload)
    return path


def describe(path):
    driver = mp3.LocalDriver(path)
    clamp = sf.info(path).frames
    line = (f'  {os.path.basename(path):<16} {os.path.getsize(path):>8,} B  '
            f'{driver.channels} ch  {driver.frames:>9,} frames  '
            f'estimate {clamp:>9,}  tail {driver.frames - clamp:>7,}  '
            f'{"tail-scan" if driver._layout is not None else "scan"}')
    driver.close()
    print(line)


def main():
    mono, stereo = sys.argv[1], sys.argv[2]
    os.makedirs(FIXTURES, exist_ok=True)

    start, _ = first_frame_with(mono, padding=1)
    body = frames_from(mono, start, MONO_FRAMES)
    write('truncating.mp3', body)
    write('tagged.mp3', id3_tag(mono) + body)
    write('tiny.mp3', frames_from(mono, start, TINY_FRAMES))

    start, _ = first_frame_with(mono, padding=0)
    write('generous.mp3', frames_from(mono, start, MONO_FRAMES))

    start, _ = first_frame_with(stereo, padding=1)
    write('stereo.mp3', frames_from(stereo, start, STEREO_FRAMES))

    print('wrote:')
    for name in sorted(os.listdir(FIXTURES)):
        describe(os.path.join(FIXTURES, name))


if __name__ == '__main__':
    main()
