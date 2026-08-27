"""Reads .mp3 files at full libsndfile speed without libsndfile's truncation.

See README.md in this directory for the full account: why libsndfile loses the
tail of a long mp3, which files are affected, the benchmarks, the alternatives
that were tried and rejected, and why the read path runs in a helper process.
"""

import atexit
import multiprocessing
import os
import sys
import threading

import numpy as np
import soundfile as sf

from multiprocessing import shared_memory

# libmpg123 issues two reads per MPEG frame during the scan, so this buffer is
# what keeps that from becoming two syscalls per frame. Measured: 1MB buffered
# virtual IO reads slightly *faster* than letting libsndfile open the path
# itself, so the shim costs nothing on the read path.
_BUFFER_BYTES = 1 << 20

# What libsndfile reports when it has no idea how long a file is. If a scan
# somehow fails we must not present this as a real frame count.
_SF_COUNT_MAX = 0x7FFFFFFFFFFFFFFF

# Where the work runs: 'auto' (a helper process whenever another file is
# already open -- i.e. whenever there is contention to lose), 'always', or
# 'never' for the pre-2.0.1 in-process behaviour.
ENV_HELPERS = 'BUZZDETECT_MP3_HELPERS'

# A helper is leased for the lifetime of one open file and returned to the
# pool, so this only ever binds if something opens more files concurrently
# than there are streamers. Past it, opens fall back in-process.
MAX_HELPERS = 32

# Longest a helper may take to answer before we give up on it and reopen the
# file in-process. Generous on purpose: the first read of a file includes the
# whole mpg123 scan, which is minutes on a slow network mount.
REPLY_TIMEOUT_S = 600


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


class LocalDriver:
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

    This is the in-process implementation. `Driver` is what the driver map
    hands out, and it normally runs one of these in a helper process, because
    the shim's virtual IO crosses into Python a few million times per scan and
    so holds the GIL against every other streamer. Same file, same bytes,
    different address space.
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


# Helper process
#
# The parent never decodes an mp3 itself when it can avoid it. A helper owns a
# LocalDriver and answers open/read/seek/tell over a Pipe; sample data comes
# back through shared memory, because a 200s chunk at 44.1kHz is 35MB and
# pickling that through a 64KB pipe would cost more than the decode.

class HelperError(RuntimeError):
    """A helper process failed to answer. The caller reopens in-process."""


def _attach_shm(name):
    """Attach to the parent's buffer without registering it for cleanup.

    track=False matters: a child that registers the segment will have its
    resource tracker unlink it on exit, destroying a buffer the parent still
    owns.
    """
    try:
        return shared_memory.SharedMemory(name=name, create=False, track=False)
    except TypeError:  # track= is 3.13+
        return shared_memory.SharedMemory(name=name, create=False)


def _helper_main(conn):
    """Child entry point. One open file at a time, for its whole lifetime."""
    track = None
    shm = None
    shm_name = None
    try:
        while True:
            try:
                message = conn.recv()
            except EOFError:
                return

            op = message[0]
            try:
                if op == 'open':
                    if track is not None:
                        track.close()
                        track = None
                    track = LocalDriver(message[1])
                    conn.send(('ok', track.samplerate, track.channels, track.frames))

                elif op == 'read':
                    _, n_samples, dtype, name, _ = message
                    if shm is None or shm_name != name:
                        if shm is not None:
                            shm.close()
                        shm = _attach_shm(name)
                        shm_name = name
                    data = track.read(n_samples, dtype=dtype)
                    view = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
                    view[...] = data
                    del view
                    conn.send(('ok', data.shape[0]))

                elif op == 'seek':
                    conn.send(('ok', track.seek(message[1])))

                elif op == 'tell':
                    conn.send(('ok', track.tell()))

                elif op == 'close':
                    if track is not None:
                        track.close()
                        track = None
                    conn.send(('ok',))

                else:  # 'bye'
                    conn.send(('ok',))
                    return

            except Exception as e:
                conn.send(('err', f'{type(e).__name__}: {e}'))
    finally:
        if track is not None:
            try:
                track.close()
            except Exception:
                pass
        if shm is not None:
            shm.close()
        conn.close()


class _Helper:
    """Parent-side handle on one helper process."""

    def __init__(self):
        ctx = multiprocessing.get_context('spawn')
        self._conn, child_conn = ctx.Pipe(duplex=True)
        self._process = ctx.Process(target=_helper_main, args=(child_conn,), daemon=True)
        self._process.start()
        # The parent must drop its end or the pipe never reports EOF when the
        # child dies.
        child_conn.close()
        self._shm = None

    def alive(self):
        return self._process.is_alive()

    def call(self, *message):
        """Send one request and return the reply's payload tuple."""
        try:
            self._conn.send(message)
            if not self._conn.poll(REPLY_TIMEOUT_S):
                raise HelperError(f'helper did not answer {message[0]!r} '
                                  f'within {REPLY_TIMEOUT_S}s')
            reply = self._conn.recv()
        except HelperError:
            raise
        except Exception as e:
            raise HelperError(f'helper died during {message[0]!r}: '
                              f'{type(e).__name__}: {e}') from e
        if reply[0] != 'ok':
            # An error raised inside the child's LocalDriver, not a transport
            # failure -- the helper is still healthy.
            raise HelperError(reply[1])
        return reply[1:]

    def buffer(self, nbytes):
        """A shared buffer of at least nbytes, reused across reads."""
        if self._shm is not None and self._shm.size >= nbytes:
            return self._shm
        self._release_buffer()
        self._shm = shared_memory.SharedMemory(create=True, size=max(nbytes, 1))
        return self._shm

    def _release_buffer(self):
        if self._shm is None:
            return
        try:
            self._shm.close()
            self._shm.unlink()
        except Exception:
            pass
        self._shm = None

    def kill(self):
        self._release_buffer()
        try:
            self._conn.close()
        except Exception:
            pass
        if self._process.is_alive():
            self._process.terminate()
        self._process.join(timeout=5)

    def retire(self):
        """Ask the child to exit, then make sure it did."""
        try:
            self.call('bye')
        except HelperError:
            pass
        self.kill()


_pool_lock = threading.Lock()
_pool_idle = []
_pool_size = 0
_warned = set()


def _warn_once(key, message):
    if key in _warned:
        return
    _warned.add(key)
    # stderr rather than the worker log: drivers are built deep inside a
    # streamer and have no handle on the coordinator's logger. The desktop app
    # surfaces engine stderr in its log pane.
    print(f'WARNING: {message}', file=sys.stderr, flush=True)


def _acquire_helper():
    """An idle helper, a new one, or None if we should stay in-process."""
    global _pool_size
    with _pool_lock:
        if _pool_idle:
            return _pool_idle.pop()
        if _pool_size >= MAX_HELPERS:
            return None
        _pool_size += 1
    try:
        return _Helper()
    except Exception as e:
        with _pool_lock:
            _pool_size -= 1
        _warn_once(
            'no-helper',
            f'could not start an mp3 helper process ({type(e).__name__}: {e}); '
            'decoding mp3s in-process instead. Analyses with several streamers '
            'will be slower, because the mpg123 length scan holds the GIL.'
        )
        return None


def _release_helper(helper, healthy=True):
    global _pool_size
    if healthy and helper.alive():
        with _pool_lock:
            _pool_idle.append(helper)
        return
    helper.kill()
    with _pool_lock:
        _pool_size -= 1


@atexit.register
def _shutdown_helpers():
    with _pool_lock:
        idle, _pool_idle[:] = list(_pool_idle), []
    for helper in idle:
        helper.retire()


# How many Driver instances currently hold an open file. This is what 'auto'
# reads: a file opening while another is already open is a file whose scan
# would be contending for the GIL, and that is exactly when the helper pays
# for itself. A single-streamer run never sees a second open and so never
# spawns anything.
_live_lock = threading.Lock()
_live_open = 0


def _want_helper(contended):
    """Whether this open should go to a helper.

    `contended` is whether another file was already open when this one started
    opening -- read before the caller counts itself, or every open looks
    contended.
    """
    mode = os.environ.get(ENV_HELPERS, 'auto').strip().lower()
    if mode in ('0', 'off', 'no', 'false', 'never'):
        return False
    if mode in ('1', 'on', 'yes', 'true', 'always'):
        return True
    return contended


class Driver:
    """The mp3 reader the driver map hands out.

    Presents the driver contract (README.md) over a LocalDriver that normally
    lives in a helper process. Everything about the decode is identical --
    same libsndfile, same shim, same bytes out -- but the scan's few million
    Python-level virtual-IO callbacks happen in an interpreter that isn't
    holding up seven other streamers and an analyzer.

    A helper that dies or stops answering is not fatal: the file reopens
    in-process, seeks back to where it was, and the read is retried once.
    """

    def __init__(self, path):
        self.path_audio = path
        self._local = None
        self._helper = None
        self._position = 0

        with _live_lock:
            global _live_open
            was_contended = _live_open > 0
            _live_open += 1
        self._counted = True

        try:
            if _want_helper(was_contended):
                self._open_helper(path)
            if self._helper is None:
                self._open_local(path)
        except Exception:
            self._uncount()
            raise

    # Opening
    #
    def _open_helper(self, path):
        helper = _acquire_helper()
        if helper is None:
            return
        try:
            self.samplerate, self.channels, self.frames = helper.call('open', path)
        except HelperError as e:
            _release_helper(helper, healthy=False)
            _warn_once(
                'helper-open',
                f'mp3 helper process could not open {os.path.basename(path)} '
                f'({e}); opening it in-process instead.'
            )
            return
        self._helper = helper

    def _open_local(self, path):
        self._local = LocalDriver(path)
        self.samplerate = self._local.samplerate
        self.channels = self._local.channels
        self.frames = self._local.frames

    def _degrade(self, reason):
        """Give up on the helper and carry on in-process from where we were."""
        _warn_once(
            'helper-lost',
            f'mp3 helper process stopped answering ({reason}); '
            'reopening in-process and continuing.'
        )
        _release_helper(self._helper, healthy=False)
        self._helper = None
        self._open_local(self.path_audio)
        self._local.seek(self._position)

    # Driver contract
    #
    def read(self, n_samples, dtype='float32'):
        if self._helper is None:
            data = self._local.read(n_samples, dtype=dtype)
            self._position += data.shape[0]
            return data

        try:
            data = self._read_helper(n_samples, dtype)
        except HelperError as e:
            self._degrade(e)
            data = self._local.read(n_samples, dtype=dtype)

        self._position += data.shape[0]
        return data

    def _read_helper(self, n_samples, dtype):
        itemsize = np.dtype(dtype).itemsize
        nbytes = n_samples * max(self.channels, 1) * itemsize
        shm = self._helper.buffer(nbytes)

        (frames,) = self._helper.call('read', n_samples, dtype, shm.name, nbytes)

        shape = (frames,) if self.channels == 1 else (frames, self.channels)
        # Copy out: the buffer is reused by the next read, and a view into it
        # would keep the segment exported and block close().
        view = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        data = view.copy()
        del view
        return data

    def seek(self, sample_index):
        if self._helper is None:
            self._position = self._local.seek(sample_index)
            return self._position
        try:
            (position,) = self._helper.call('seek', sample_index)
        except HelperError as e:
            self._degrade(e)
            position = self._local.seek(sample_index)
        self._position = position
        return position

    def tell(self):
        if self._helper is None:
            return self._local.tell()
        try:
            (position,) = self._helper.call('tell')
        except HelperError as e:
            self._degrade(e)
            position = self._local.tell()
        self._position = position
        return position

    def close(self):
        if self._helper is not None:
            healthy = True
            try:
                self._helper.call('close')
            except HelperError:
                healthy = False
            _release_helper(self._helper, healthy=healthy)
            self._helper = None
        if self._local is not None:
            self._local.close()
            self._local = None
        self._uncount()

    def _uncount(self):
        if not self._counted:
            return
        self._counted = False
        with _live_lock:
            global _live_open
            _live_open -= 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
