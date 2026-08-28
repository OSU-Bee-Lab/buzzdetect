"""What shapes of mp3 actually exist in the corpora?

The tail-scan work has to hold for every file the analyzer will meet, not just
the 48 kbps CBR mono ones the benchmarks use. This samples files across every
experiment folder and reports the stream parameters, whether a Xing/Info header
is present (which would make libsndfile's length correct and the whole tail
problem moot), and the recorder string in the ID3 tag.

Usage:  python3 survey_corpora.py <root> [files_per_folder]
"""

import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mp3_frames import find_frame     # noqa: E402


def id3_size(head):
    if len(head) < 10 or head[:3] != b'ID3':
        return 0
    size = 0
    for byte in head[6:10]:
        size = (size << 7) | (byte & 0x7F)
    return 10 + size + (10 if head[5] & 0x10 else 0)


def describe(path):
    with open(path, 'rb') as f:
        head = f.read(1 << 16)
    skip = id3_size(head)
    if skip >= len(head):
        with open(path, 'rb') as f:
            f.seek(skip)
            buf = f.read(1 << 16)
    else:
        buf = head[skip:]
    header = find_frame(buf, 0, 4)
    if header is None:
        return None, None, None
    frame = buf[header['offset']:header['offset'] + header['size']]
    # Xing/Info sits inside the first frame's side-information padding.
    xing = b'Xing' in frame or b'Info' in frame
    shape = (f"MPEG{'1' if header['version'] == 3 else '2'} "
             f"{header['bitrate']}kbps {header['samplerate']}Hz "
             f"{'mono' if header['channels'] == 1 else 'stereo'}"
             f"{' +Xing' if xing else ''}")
    device = None
    match = re.search(rb'([A-Z][A-Za-z0-9]{1,12}-[A-Za-z0-9]{2,10})', head[:skip or 1])
    if match:
        device = match.group(1).decode('latin-1')
    return shape, device, os.path.getsize(path)


def main():
    root = sys.argv[1]
    per_folder = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    shapes = Counter()
    devices = Counter()
    for folder, _, names in os.walk(root):
        mp3s = sorted(n for n in names if n.lower().endswith('.mp3'))
        for name in mp3s[:per_folder]:
            path = os.path.join(folder, name)
            try:
                shape, device, _ = describe(path)
            except Exception as e:
                shapes[f'ERROR {type(e).__name__}'] += 1
                continue
            shapes[shape or 'no MPEG frame found'] += 1
            devices[device or '(none)'] += 1
    print('stream shapes:')
    for shape, count in shapes.most_common():
        print(f'  {count:>6}  {shape}')
    print('recorder strings:')
    for device, count in devices.most_common(12):
        print(f'  {count:>6}  {device}')


if __name__ == '__main__':
    main()
