"""Reads .mp3 files at full libsndfile speed without libsndfile's truncation.

See README.md in this directory for the full account: why libsndfile loses the
tail of a long mp3, which files are affected, the benchmarks, and the
alternatives that were tried and rejected.
"""

import os

import numpy as np
import soundfile as sf

# libmpg123 issues two reads per MPEG frame during the scan, so this buffer is
# what keeps that from becoming two syscalls per frame. Measured: 1MB buffered
# virtual IO reads slightly *faster* than letting libsndfile open the path
# itself, so the shim costs nothing on the read path.
_BUFFER_BYTES = 1 << 20

# What libsndfile reports when it has no idea how long a file is. If a scan
# somehow fails we must not present this as a real frame count.
_SF_COUNT_MAX = 0x7FFFFFFFFFFFFFFF


def _id3v2_size(fileobj):
    """Byte length of a leading ID3v2 tag, or 0 if there isn't one."""
    fileobj.seek(0)
    header = fileobj.read(10)
    if len(header) < 10 or header[:3] != b'ID3':
        return 0
    # 28-bit synchsafe integer: 7 bits per byte, high bit always clear.
    size = 0
    for byte in header[6:10]:
        size = (size << 7) | (byte & 0x7F)
    footer = 10 if header[5] & 0x10 else 0
    return 10 + size + footer


class _LengthHidingFile:
    """File-like shim that exposes path[skip:] but refuses to state its length.

    Reporting 0 for the length is what forces libsndfile down its
    mpg123_scan() path. The tag has to be skipped as well: with no length to
    work from, libsndfile cannot navigate a leading ID3v2 tag and rejects the
    file outright as "Format not recognised", so the shim presents the first
    MPEG frame at virtual offset 0.
    """

    def __init__(self, path):
        self._file = open(path, 'rb', buffering=_BUFFER_BYTES)
        self._skip = _id3v2_size(self._file)
        self._length = max(0, os.path.getsize(path) - self._skip)
        # Only a seek to the very end may answer with the lie -- libsndfile
        # calls tell() constantly for ordinary bookkeeping and mis-answering
        # those corrupts its parsing.
        self._at_end = False
        self._file.seek(self._skip)

    def read(self, count=-1):
        return self._file.read(count)

    def readinto(self, buffer):
        # Present so python-soundfile's vio_read uses the zero-copy path;
        # measured at roughly half the open-time cost of falling back to
        # read(), which matters across the scan's millions of calls.
        return self._file.readinto(buffer)

    def tell(self):
        position = self._file.tell() - self._skip
        if self._at_end and position == self._length:
            return 0
        return position

    def seek(self, offset, whence=os.SEEK_SET):
        self._at_end = (whence == os.SEEK_END and offset == 0)
        if whence == os.SEEK_END:
            position = self._file.seek(self._skip + self._length + offset, os.SEEK_SET)
            return 0 if self._at_end else position - self._skip
        if whence == os.SEEK_SET:
            return self._file.seek(offset + self._skip, os.SEEK_SET) - self._skip
        return self._file.seek(offset, whence) - self._skip

    def close(self):
        self._file.close()


class Driver:
    """soundfile.SoundFile-alike that reports and reads an mp3's true length.

    libsndfile extrapolates an mp3's length from the size of its first frame
    and then hard-clamps reads and seeks to that guess, which silently strands
    the tail of any file whose first frame is padded (measured: 301.7s lost off
    a 49.7h recording, and unreachable at the C API too, not just through
    python-soundfile). It does contain an exact scan -- mpeg_decode.c only
    reaches it when mpg123_length() fails, and that only fails when there is no
    file size to extrapolate from.

    So this is ordinary libsndfile, opened through a shim that withholds the
    file's length to force the scan. Decoding, seeking and output are all
    unchanged: reads are bit-identical to a plain soundfile open, and cost
    parity was measured on both long and short corpora. README.md has the
    numbers and the reasoning.
    """

    def __init__(self, path):
        self.path_audio = path
        self._shim = None
        self._track = self._open_scanned(path)
        if self._track is None:
            # Any file the scan can't measure is still better served by
            # libsndfile's estimate than by refusing to read it at all -- a
            # wrong length is handled downstream as a short read at EOF.
            self._close_shim()
            self._track = sf.SoundFile(path)

        self.samplerate = self._track.samplerate
        self.channels = self._track.channels
        self.frames = self._track.frames

    def _open_scanned(self, path):
        try:
            self._shim = _LengthHidingFile(path)
            track = sf.SoundFile(self._shim)
        except Exception:
            return None
        if track.frames <= 0 or track.frames >= _SF_COUNT_MAX:
            track.close()
            return None
        return track

    def _close_shim(self):
        if self._shim is not None:
            self._shim.close()
            self._shim = None

    def read(self, n_samples, dtype='float32'):
        return self._track.read(n_samples, dtype=dtype)

    def seek(self, sample_index):
        return self._track.seek(sample_index)

    def tell(self):
        return self._track.tell()

    def close(self):
        self._track.close()
        self._close_shim()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
