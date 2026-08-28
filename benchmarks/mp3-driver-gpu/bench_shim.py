"""Does an mmap-backed length-hiding shim cut libsndfile's per-read Python tax?

libmpg123 issues roughly two I/O calls per MPEG frame, so a 500s chunk of
48kbps audio is ~38k readinto() calls into Python. The current shim serves each
from a BufferedReader. This compares that against serving them from a
memoryview over an mmap, which should make each call a bare memcpy.
"""
import mmap
import os
import sys
import time

import soundfile as sf

sys.path.insert(0, '.')
from src.stream.drivers.mp3 import _LengthHidingFile, _id3v2_size, _SF_COUNT_MAX


class MmapLengthHidingFile:
    """_LengthHidingFile, backed by a memoryview over an mmap.

    Same contract exactly: presents path[skip:], reports 0 only for a seek to
    the very end, so libsndfile takes its mpg123_scan() path.
    """

    def __init__(self, path):
        self._file = open(path, 'rb')
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self._view = memoryview(self._mm)
        self._skip = _id3v2_size(self._file)
        self._length = max(0, len(self._mm) - self._skip)
        self._pos = 0
        self._at_end = False

    def readinto(self, buffer):
        n = min(len(buffer), self._length - self._pos)
        if n <= 0:
            return 0
        start = self._skip + self._pos
        # memoryview slicing is a view, not a copy, so this is one memcpy with
        # no intermediate bytes object and none of BufferedReader's bookkeeping.
        buffer[:n] = self._view[start:start + n]
        self._pos += n
        return n

    def read(self, count=-1):
        if count is None or count < 0:
            count = self._length - self._pos
        n = min(count, self._length - self._pos)
        start = self._skip + self._pos
        self._pos += n
        return self._view[start:start + n].tobytes()

    def tell(self):
        if self._at_end and self._pos == self._length:
            return 0
        return self._pos

    def seek(self, offset, whence=os.SEEK_SET):
        self._at_end = (whence == os.SEEK_END and offset == 0)
        if whence == os.SEEK_END:
            self._pos = max(0, min(self._length, self._length + offset))
            return 0 if self._at_end else self._pos
        if whence == os.SEEK_SET:
            self._pos = max(0, min(self._length, offset))
        else:
            self._pos = max(0, min(self._length, self._pos + offset))
        return self._pos

    def close(self):
        self._view.release()
        self._mm.close()
        self._file.close()


def open_scanned(shim_cls, path):
    shim = shim_cls(path)
    track = sf.SoundFile(shim)
    if track.frames <= 0 or track.frames >= _SF_COUNT_MAX:
        raise RuntimeError('scan failed')
    return shim, track


def main():
    path = sys.argv[1]
    chunk_s = float(sys.argv[2]) if len(sys.argv) > 2 else 500.0

    results = {}
    for name, cls in (('BufferedReader (current)', _LengthHidingFile),
                      ('mmap + memoryview', MmapLengthHidingFile)):
        t0 = time.perf_counter()
        shim, track = open_scanned(cls, path)
        t_open = time.perf_counter() - t0
        frames, sr = track.frames, track.samplerate
        n = int(chunk_s * sr)

        reads = []
        for _ in range(3):
            track.seek(0)
            t = time.perf_counter()
            track.read(n, dtype='float32')
            reads.append(time.perf_counter() - t)
        track.close()
        shim.close()
        results[name] = (t_open, min(reads), frames, sr)

    # Plain soundfile, for the floor. Truncates, which is the whole problem,
    # but it is the speed the driver is trying to reach.
    t0 = time.perf_counter()
    plain = sf.SoundFile(path)
    t_open_plain = time.perf_counter() - t0
    n = int(chunk_s * plain.samplerate)
    plain_reads = []
    for _ in range(3):
        plain.seek(0)
        t = time.perf_counter()
        plain.read(n, dtype='float32')
        plain_reads.append(time.perf_counter() - t)
    results['plain soundfile (truncates)'] = (t_open_plain, min(plain_reads),
                                              plain.frames, plain.samplerate)
    plain.close()

    print(f'{os.path.basename(path)}   chunk {chunk_s:.0f}s\n')
    print(f"{'path':<30} {'open(scan)':>12} {'read':>10} {'read rate':>11}  frames")
    for name, (t_open, t_read, frames, sr) in results.items():
        print(f'{name:<30} {t_open:11.2f}s {t_read*1000:9.1f}ms '
              f'{chunk_s/t_read:10.0f}x  {frames:,}')

    cur = results['BufferedReader (current)']
    mm = results['mmap + memoryview']
    print(f'\nmmap vs current:  open {100*(cur[0]-mm[0])/cur[0]:+.1f}%   '
          f'read {100*(cur[1]-mm[1])/cur[1]:+.1f}% faster')
    assert cur[2] == mm[2], f'frame counts differ! {cur[2]} vs {mm[2]}'
    print(f'frame counts agree: {cur[2]:,}')


if __name__ == '__main__':
    main()
