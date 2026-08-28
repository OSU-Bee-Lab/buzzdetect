"""Reads .mp3 files at full libsndfile speed without libsndfile's truncation.

See README.md in this directory for the full account: why libsndfile loses the
tail of a long mp3, which files are affected, the benchmarks, the alternatives
that were tried and rejected, and why the read path runs in a helper process.

The body of a file is read through an ordinary `soundfile.SoundFile`, at full C
speed; only the tail libsndfile refuses to reach is decoded through a shim, and
that shim covers a couple of megabytes rather than the whole file.
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


def _trailing_tag_size(path):
    """Byte length of an ID3v1 or APE trailer, or 0 if there isn't one.

    A trailer is not audio, so it must come off the byte count before the frame
    arithmetic below divides by a frame size.
    """
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


# MPEG audio frame headers
#
# The tail is read through a shim that starts partway into the file, and
# libsndfile will only open such a thing if it begins on a real frame header --
# the same requirement the ID3 skip already exists to satisfy. So the driver has
# to be able to find one and be sure of it.

_BITRATES_V1_L3 = (None, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224,
                   256, 320, None)
_BITRATES_V2_L3 = (None, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144,
                   160, None)
_SAMPLERATES = {
    3: (44100, 48000, 32000),   # MPEG 1
    2: (22050, 24000, 16000),   # MPEG 2
    0: (11025, 12000, 8000),    # MPEG 2.5
}

# How many consecutive well-formed frames a candidate boundary must start.
# Audio data contains 0xFF bytes constantly; one plausible header proves
# nothing, but a header whose computed size lands exactly on the next header,
# three times over, is not a coincidence.
_CHAIN = 4


def _parse_header(buf, offset):
    """Decode the MPEG audio frame header at `offset`, or None if it isn't one.

    Layer III only. The other layers exist but no recorder in this project's
    corpora emits them, and accepting them would widen what counts as a valid
    boundary for no gain.
    """
    if offset + 4 > len(buf):
        return None
    b0, b1, b2, b3 = buf[offset], buf[offset + 1], buf[offset + 2], buf[offset + 3]
    if b0 != 0xFF or (b1 & 0xE0) != 0xE0:
        return None

    version = (b1 >> 3) & 0x03      # 3=MPEG1, 2=MPEG2, 0=MPEG2.5, 1=reserved
    layer = (b1 >> 1) & 0x03        # 1 == Layer III
    if version == 1 or layer != 1:
        return None

    bitrate_index = (b2 >> 4) & 0x0F
    samplerate_index = (b2 >> 2) & 0x03
    if samplerate_index == 3:
        return None
    table = _BITRATES_V1_L3 if version == 3 else _BITRATES_V2_L3
    bitrate = table[bitrate_index]
    if bitrate is None:             # 0 is "free format", 15 is reserved
        return None

    samplerate = _SAMPLERATES[version][samplerate_index]
    padding = (b2 >> 1) & 0x01
    channels = 1 if ((b3 >> 6) & 0x03) == 3 else 2
    samples = 1152 if version == 3 else 576
    size = (samples // 8) * bitrate * 1000 // samplerate + padding

    return {
        'offset': offset,
        'version': version,
        'bitrate': bitrate,
        'samplerate': samplerate,
        'channels': channels,
        'samples': samples,
        'size': size,
    }


def _chain_ok(buf, header, chain=_CHAIN):
    """Do `chain` consecutive consistent frames start here?"""
    at = header
    for _ in range(chain - 1):
        following = _parse_header(buf, at['offset'] + at['size'])
        if following is None:
            return False
        for key in ('version', 'samplerate', 'channels', 'bitrate'):
            if following[key] != header[key]:
                return False
        at = following
    return True


def _find_frame(buf, start=0, chain=_CHAIN):
    """Header of the first validated frame at or after `start` in `buf`.

    `buf` must have room for the whole chain past the boundary being looked
    for, or a real boundary near its end will be rejected. The file's last
    frames have no room for a chain, so they are checked a different way --
    see `_last_frame_offset`.
    """
    at = start
    limit = len(buf) - 4
    while at <= limit:
        index = buf.find(b'\xff', at)
        if index < 0 or index > limit:
            return None
        header = _parse_header(buf, index)
        if header is not None and _chain_ok(buf, header, chain):
            return header
        at = index + 1
    return None


# Enough bytes to hold _CHAIN frames of any bitrate this driver accepts, so a
# boundary search never fails for want of room to validate.
_SEARCH_BYTES = 1 << 13


class _Layout:
    """Where a constant-bitrate mp3's frames are, worked out by arithmetic.

    libsndfile gets an mp3's length wrong because it assumes every frame is as
    long as the first, and on these recorders the first is a padded one. But a
    CBR stream's frames average exactly 144 * bitrate / samplerate bytes -- the
    padding bit is how an integer frame size tracks that non-integer average --
    so the audio byte count divided by it gives the frame count exactly, and the
    same arithmetic gives the byte offset of any frame to within a byte or two.

    That collapses the whole mpg123 length scan into a division, and it is what
    lets the driver find the one frame boundary it needs near the end of the
    file without reading the rest of it.

    Built by `_read_layout`, which validates it and returns None if the file is
    not the shape this assumes.
    """

    def __init__(self, audio_start, header, bytes_per_frame, n_frames):
        self.audio_start = audio_start
        self.samplerate = header['samplerate']
        self.channels = header['channels']
        self.samples_per_frame = header['samples']
        self.bytes_per_frame = bytes_per_frame
        self.n_frames = n_frames
        self.frames = n_frames * header['samples']

    def _search_window(self, index):
        guess = self.audio_start + int(index * self.bytes_per_frame)
        # Frames are at least 104 bytes at the bitrates accepted here, so a
        # window of a third of a frame either side can only contain the one
        # boundary being looked for.
        margin = max(8, int(self.bytes_per_frame) // 3)
        return guess, margin, max(self.audio_start, guess - margin)

    def last_frame_ok(self, path, audio_end):
        """Does the predicted last frame exist, and end where the audio does?

        The end of the file is where a wrong frame count shows up, and it is
        also the one place a chain of following frames cannot be required --
        there are none. So the check is the other way round: a header where the
        arithmetic says the last frame starts, whose own length lands on the
        end of the audio.
        """
        guess, margin, window = self._search_window(self.n_frames - 1)
        with open(path, 'rb') as f:
            f.seek(window)
            buf = f.read(margin * 2 + 8)
        at = 0
        while True:
            index = buf.find(b'\xff', at)
            if index < 0:
                return False
            header = _parse_header(buf, index)
            if header is not None and abs(window + index - guess) <= margin:
                if abs(window + index + header['size'] - audio_end) <= 2:
                    return True
            at = index + 1

    def frame_offset(self, path, index):
        """Byte offset of frame `index`, verified against a real header.

        The arithmetic lands within a byte or two; the search around it is what
        makes the answer exact, and the validation is what makes a wrong guess
        fail loudly instead of handing libsndfile the middle of a frame.
        """
        guess, margin, window = self._search_window(index)
        with open(path, 'rb') as f:
            f.seek(window)
            buf = f.read(margin * 2 + _SEARCH_BYTES)
        header = _find_frame(buf)
        if header is None:
            return None
        offset = window + header['offset']
        if abs(offset - guess) > margin:
            return None
        return offset


def _read_layout(path):
    """The CBR layout of `path`, or None if it does not have one.

    Everything the fast read path assumes is checked here, so a file that is
    VBR, free-format, oddly framed, or simply not what it looks like drops back
    to the scan rather than being read wrongly.
    """
    size = os.path.getsize(path)
    with open(path, 'rb') as f:
        skip = _id3v2_size(f)
        f.seek(skip)
        buf = f.read(_SEARCH_BYTES)
    header = _find_frame(buf)
    if header is None:
        return None
    audio_start = skip + header['offset']

    audio_bytes = size - audio_start - _trailing_tag_size(path)
    if audio_bytes <= 0:
        return None

    # 144 = 1152 samples / 8 bits per byte. MPEG 2 and 2.5 carry half as many
    # samples per frame, and so half as many bytes.
    bytes_per_frame = 144.0 * header['bitrate'] * 1000 / header['samplerate']
    if header['version'] != 3:
        bytes_per_frame /= 2
    n_frames = int(round(audio_bytes / bytes_per_frame))
    if n_frames < 2:
        return None

    layout = _Layout(audio_start, header, bytes_per_frame, n_frames)

    # Validate. A VBR file's frames drift away from the average immediately, so
    # probing offsets across the file is what rules it out; the last frame is
    # probed separately because that is the one the tail read depends on and
    # the one an appended non-audio trailer would break.
    for fraction in (0.05, 0.25, 0.5, 0.75, 0.95):
        index = int((n_frames - 1) * fraction)
        if layout.frame_offset(path, index) is None:
            return None
    if not layout.last_frame_ok(path, audio_start + audio_bytes):
        return None
    return layout


class _LengthHidingFile:
    """File-like shim that exposes path[skip:] but refuses to state its length.

    Reporting 0 for the length is what forces libsndfile down its
    mpg123_scan() path. The tag has to be skipped as well: with no length to
    work from, libsndfile cannot navigate a leading ID3v2 tag and rejects the
    file outright as "Format not recognised", so the shim presents the first
    MPEG frame at virtual offset 0.

    `skip` names the byte to present as offset 0. Left unset it is the end of
    the ID3v2 tag, which is the whole-file case; the tail read passes the offset
    of a frame boundary near the end instead, so the scan libsndfile is being
    pushed into covers a couple of megabytes rather than gigabytes.
    """

    def __init__(self, path, skip=None):
        self._file = open(path, 'rb', buffering=_BUFFER_BYTES)
        self._skip = _id3v2_size(self._file) if skip is None else skip
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


# How the tail read is stitched onto the body
#
# Only the last fraction of a percent of a file is past libsndfile's clamp, so
# the body is read through an ordinary SoundFile at full C speed and only the
# tail comes from a shim. The seam is the delicate part.
#
# An mp3 decode that starts partway into a file is wrong for its first frame or
# two -- the bit reservoir lets a frame borrow space from its predecessors --
# and then converges. Measured on the Solar Eclipse corpus: two frames, always.
# What is not always true is that it converges *exactly*. Of two adjacent frame
# boundaries, one reproduces the continuous decode bit for bit and the other
# leaves a residual of ~2.4e-7 that never dies away, because mpg123's synthesis
# state ends up aligned differently and sums the same window in a different
# order. Which of the two is the right one is not predictable from the file --
# it was measured to alternate with the start frame, with a phase that varies
# between regions of the same file.
#
# So the driver does not predict it, it checks it: it decodes the overlap it
# has just read from the body a second time through the fragment, and keeps the
# boundary whose overlap comes back identical. That costs one extra decode of a
# few thousand samples, and it turns a coin flip into a guarantee.
#
# benchmarks/mp3-driver-gpu/STEP1_RESULT.md has the measurements.

# How far before the clamp the tail fragment begins. Large enough that the
# decoder's warm-up and the verification window both fit before the seam,
# small enough that the scan it forces stays trivial.
_OVERLAP_FRAMES = 12

# Frames of already-decoded body re-decoded through the fragment to choose the
# boundary. Convergence out of a mid-stream start was measured at two frames, so
# this is the warm-up twice over, and a mismatched boundary perturbs roughly 40%
# of samples -- enormously more evidence than the decision needs.
_VERIFY_FRAMES = 4

# The smallest window worth deciding on. A mismatched boundary perturbs roughly
# 40% of samples, so even this many of them is overwhelming evidence.
_VERIFY_MINIMUM = 64

# How much audio the driver will re-decode to build a window when the caller's
# own read did not produce one. Past this it is cheaper to say the seam went
# unchecked than to pay for checking it.
_REPLAY_LIMIT = 1 << 21


class LocalDriver:
    """soundfile.SoundFile-alike that reports and reads an mp3's true length.

    libsndfile extrapolates an mp3's length from the size of its first frame
    and then hard-clamps reads and seeks to that guess, which silently strands
    the tail of any file whose first frame is padded (measured: 301.7s lost off
    a 49.7h recording, and unreachable at the C API too, not just through
    python-soundfile). README.md tells that story in full.

    The file is opened plainly, and read plainly for as long as libsndfile is
    willing -- which is everything but the last 0.17%. The true length comes
    from arithmetic on the file's size rather than from a scan (`_Layout`), and
    the audio past the clamp is decoded through a shim that presents only the
    last couple of megabytes. So the whole-file scan and the whole-file Python
    read path that the first version of this driver paid for are both gone, and
    what replaces them is a division and one small decode at the seam.

    If any of that does not hold -- a VBR file, a stream that does not frame the
    way the arithmetic assumes, a seam that will not verify -- the driver falls
    back to `_open_scanned`, which is the original implementation: ordinary
    libsndfile opened through a shim that withholds the file's length, which
    forces mpg123's exact scan. Correct, slower, and rare.

    Reads are bit-identical to a plain soundfile open over the body, and to the
    scanned driver over the tail. `Driver` is what the driver map hands out.
    """

    def __init__(self, path):
        self.path_audio = path
        self._layout = None
        self._plain = None
        self._clamp = 0
        self._tail = None
        self._tail_shim = None
        self._tail_first = 0        # absolute sample index of the fragment's frame 0
        self._tail_overlap = None   # the boundary that verified, in frames before the seam
        self._tail_position = None  # absolute sample index the fragment will hand out next
        self._scanned = None
        self._shim = None
        self._position = 0
        self._carry = None          # fragment samples decoded but not yet handed out
        self._seek_target = 0       # where the body track was last positioned

        self._open(path)

    # Opening
    #
    def _open(self, path):
        layout = _read_layout(path)
        if layout is not None:
            try:
                plain = sf.SoundFile(path)
            except Exception:
                plain = None
            if plain is not None and self._agrees(plain, layout):
                self._layout = layout
                self._plain = plain
                self._clamp = plain.frames
                self.samplerate = layout.samplerate
                self.channels = layout.channels
                self.frames = layout.frames
                return
            if plain is not None:
                plain.close()
        self._open_scanned(path)

    @staticmethod
    def _agrees(plain, layout):
        """Do libsndfile and the arithmetic describe the same file?

        They are allowed to disagree about the length -- that disagreement is
        the whole point -- but only by the fraction of a percent the padding
        bit can account for. A larger gap means one of the two is describing
        something else, and guessing which would be worse than scanning.
        """
        if plain.samplerate != layout.samplerate or plain.channels != layout.channels:
            return False
        if plain.frames <= 0 or layout.frames <= 0:
            return False
        return abs(plain.frames - layout.frames) / layout.frames < 0.01

    def _open_scanned(self, path):
        """The original driver: force mpg123's scan and read through the shim."""
        self._scanned = self._scan(path)
        if self._scanned is None:
            # Any file the scan can't measure is still better served by
            # libsndfile's estimate than by refusing to read it at all -- a
            # wrong length is handled downstream as a short read at EOF.
            self._close_shim()
            self._scanned = sf.SoundFile(path)

        self.samplerate = self._scanned.samplerate
        self.channels = self._scanned.channels
        self.frames = self._scanned.frames

    def _scan(self, path):
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

    def _fall_back(self, reason):
        """Abandon the fast path mid-file and carry on with the scan."""
        _warn_once(
            'tail-seam',
            f'could not stitch the tail of {os.path.basename(self.path_audio)} '
            f'onto its body ({reason}); re-reading it the slow way. The audio is '
            'the same; the analysis of this file will take longer.'
        )
        self._close_tail()
        if self._plain is not None:
            self._plain.close()
            self._plain = None
        self._layout = None
        self._open_scanned(self.path_audio)
        self._scanned.seek(min(self._position, self._scanned.frames))

    # The tail
    #
    def _close_tail(self):
        if self._tail is not None:
            self._tail.close()
            self._tail = None
        if self._tail_shim is not None:
            self._tail_shim.close()
            self._tail_shim = None
        self._tail_position = None
        self._carry = None

    def _seam_boundary(self):
        """The last frame boundary at or before libsndfile's clamp."""
        per_frame = self._layout.samples_per_frame
        return (self._clamp // per_frame) * per_frame

    def _open_fragment(self, overlap):
        """A SoundFile over the file from `overlap` frames before the seam.

        Returns (shim, track, first_sample) or None. The shim withholds its
        length exactly as the whole-file one does, so libsndfile scans it -- but
        what it scans is now a couple of megabytes.
        """
        layout = self._layout
        per_frame = layout.samples_per_frame
        first_frame = self._seam_boundary() // per_frame - overlap
        if first_frame < 0:
            return None
        offset = layout.frame_offset(self.path_audio, first_frame)
        if offset is None:
            return None
        try:
            shim = _LengthHidingFile(self.path_audio, skip=offset)
            track = sf.SoundFile(shim)
        except Exception:
            return None
        if track.channels != self.channels or track.samplerate != self.samplerate:
            track.close()
            shim.close()
            return None
        return shim, track, first_frame * per_frame

    def _reference(self, body):
        """The samples just before the clamp, as the body decoded them.

        `body` is what the caller's read has already produced when that read
        crossed the seam, which for any chunk the streamer asks for is far more
        than enough. When it is not -- a read that begins a hair before the
        clamp -- the body's decode is replayed on a scratch track from the same
        seek it came from, which reproduces it exactly, because a single
        uninterrupted read is a plain function of where it started.

        Returns None when neither is possible; the window is then unknowable and
        the caller says so.
        """
        window = _VERIFY_FRAMES * self._layout.samples_per_frame

        if body is not None:
            take = min(window, body.shape[0])
            if take >= _VERIFY_MINIMUM:
                return body[body.shape[0] - take:]

        if self._seek_target is None:
            return None
        available = self._clamp - self._seek_target
        if available < _VERIFY_MINIMUM or available > _REPLAY_LIMIT:
            return None
        try:
            with sf.SoundFile(self.path_audio) as scratch:
                scratch.seek(self._seek_target)
                replay = scratch.read(available, dtype='float32')
        except Exception:
            return None
        if replay.shape[0] < _VERIFY_MINIMUM:
            return None
        return replay[-min(window, replay.shape[0]):]

    def _probe(self, track, first, window):
        """Decode the `window` samples immediately before the clamp.

        Returns those samples and whatever came out with them past the clamp.
        The reads here cover whole frames, so what follows the window is already
        the start of the tail and becomes the fragment's carry -- which is the
        point: the fragment is never asked to resume partway through a frame.
        """
        per_frame = self._layout.samples_per_frame
        start = self._clamp - window
        aligned = ((start - first) // per_frame) * per_frame
        self._skip_aligned(track, aligned)

        lead = (start - first) - aligned
        need = lead + window
        whole = -(-need // per_frame) * per_frame
        got = track.read(whole, dtype='float32')
        if got.shape[0] < need:
            raise EOFError('the fragment ended before the seam')
        return got[lead:lead + window], got[lead + window:]

    def _ensure_tail_for_seek(self):
        """Open a fragment for seeking into, without choosing between boundaries.

        Measured: seeking to a position inside the fragment returns exactly what
        seeking to the same position in the whole file returns, from either
        boundary -- an mpg123 seek resets the decoder, so which frame the
        fragment happens to start on stops mattering.
        """
        if self._tail is not None:
            return True
        for overlap in (_OVERLAP_FRAMES, _OVERLAP_FRAMES + 1):
            opened = self._open_fragment(overlap)
            if opened is None:
                continue
            shim, track, first = opened
            self._tail_shim, self._tail, self._tail_first = shim, track, first
            self._tail_overlap = overlap
            self._tail_position = None
            self._carry = None
            return True
        return False

    def _ensure_tail(self, body):
        """Open the tail fragment on the boundary that decodes as the body did.

        Of the two frame boundaries either side, one continues the body's decode
        bit for bit and the other leaves a residual of ~2e-7 that never dies
        away. Which one is not predictable from the file, so it is measured:
        both are decoded over a window the body has already produced, and the
        one that comes back identical is kept.
        """
        if self._tail is not None:
            return True

        reference = self._reference(body)
        for overlap in (_OVERLAP_FRAMES, _OVERLAP_FRAMES + 1):
            opened = self._open_fragment(overlap)
            if opened is None:
                continue
            shim, track, first = opened
            try:
                if reference is None:
                    _, carry = self._probe(track, first, 0)
                    self._adopt_tail(shim, track, first, overlap, carry)
                    _warn_once(
                        'tail-unverified',
                        'reading the tail of '
                        f'{os.path.basename(self.path_audio)} without a body read '
                        'to check the seam against, because the analysis reached '
                        'it without decoding through it. Samples may differ from '
                        'a whole-file decode in the last digit; nothing is lost '
                        'or repeated.'
                    )
                    return True

                got, carry = self._probe(track, first, reference.shape[0])
                if np.array_equal(got, reference):
                    self._adopt_tail(shim, track, first, overlap, carry)
                    return True
            except Exception:
                pass
            track.close()
            shim.close()

        return False

    def _adopt_tail(self, shim, track, first, overlap, carry):
        """Take ownership of a fragment sitting on the seam's frame boundary."""
        self._tail_shim = shim
        self._tail = track
        self._tail_first = first
        self._tail_overlap = overlap
        self._tail_position = self._clamp
        self._carry = carry

    def _skip_aligned(self, track, count):
        """Decode and throw away `count` samples, a whole number of frames.

        Every read this driver issues to a fragment covers whole frames, and
        that is not tidiness: libsndfile's mp3 output depends on where reads are
        broken. A read resumed anywhere but a frame boundary differs from an
        unbroken one by ~6e-8 from a few samples in and never converges back
        (benchmarks/mp3-driver-gpu/step1_segmentation.py). Frame-aligned reads
        are what make the fragment's decode identical to a continuous one.

        Deliberately not a seek, either: mpg123's seek leaves its synthesis
        state aligned differently again.
        """
        per_frame = self._layout.samples_per_frame
        if count % per_frame:
            raise ValueError('fragment reads must cover whole frames')
        step = per_frame * 910          # ~1M samples, still whole frames
        while count > 0:
            got = track.read(min(step, count), dtype='float32')
            if got.shape[0] == 0:
                raise EOFError('the fragment ended before the seam')
            count -= got.shape[0]

    def _read_fragment(self, n_samples):
        """`n_samples` from the fragment, out of the carry first.

        Only the seam produces a carry: the probe that lands the fragment on the
        clamp has to read whole frames to get there, so it overshoots, and what
        it overshot by is the first of the tail. Past that the fragment is read
        straight, ending wherever the caller's read ends -- which is what keeps
        the breaks in this decode in the same places as the breaks in a
        whole-file one.
        """
        carry = self._carry
        if carry is not None and carry.shape[0]:
            take = min(n_samples, carry.shape[0])
            head, self._carry = carry[:take], carry[take:]
            if take == n_samples:
                return head
            rest = self._tail.read(n_samples - take, dtype='float32')
            return head if rest.shape[0] == 0 else np.concatenate((head, rest))
        self._carry = None
        return self._tail.read(n_samples, dtype='float32')

    def _seek_tail(self, target):
        """Put the fragment's next sample at `target`.

        This is an ordinary libsndfile seek, and it needs no care at all:
        seeking to a position inside the fragment was measured to return exactly
        what seeking to the same position in the whole file returns, from either
        boundary. The boundary only matters when the tail is reached by reading
        into it, because then there is a decode in progress to continue.
        """
        self._carry = None
        self._tail.seek(target - self._tail_first)
        self._tail_position = target


    # Driver contract
    #
    def read(self, n_samples, dtype='float32', out=None):
        """Decode n_samples, optionally straight into a caller's buffer.

        `out` is what lets the helper decode directly into shared memory
        instead of into a fresh array it then copies -- one 88MB allocation
        and one 88MB memcpy saved per chunk at a 500s chunk length.
        """
        want = max(0, min(int(n_samples), self.frames - self._position))
        if self._layout is None:
            data = self._read_track(self._scanned, n_samples, dtype, out)
            self._position += data.shape[0]
            return data

        body_end = min(self._clamp, self.frames)
        n_body = max(0, min(want, body_end - self._position))
        n_tail = want - n_body

        if n_tail == 0:
            data = self._read_track(self._plain, n_body, dtype, out)
            self._position += data.shape[0]
            return data

        if n_body == 0:
            data = self._read_tail(n_tail, dtype, out)
            self._position += data.shape[0]
            return data

        if n_body < _VERIFY_MINIMUM and self._position == self._seek_target:
            # A read that begins a handful of samples before the clamp leaves
            # nothing to check a boundary against. It does not need one: nothing
            # has been decoded since the seek, so the fragment can be seeked to
            # the same place and serve the whole read, and a seek into the
            # fragment returns what a seek into the whole file returns.
            data = self._read_seeked(want, dtype, out)
            if data is not None:
                self._position += data.shape[0]
                return data

        return self._read_across(n_body, n_tail, dtype, out)

    def _read_seeked(self, want, dtype, out):
        """The whole read from the fragment, seeked to where the caller is.

        Returns None if the fragment does not reach back that far, which leaves
        the caller to stitch the read together instead.
        """
        if not self._ensure_tail_for_seek() or self._position < self._tail_first:
            return None
        try:
            self._seek_tail(self._position)
        except Exception:
            return None
        return self._hand_out(self._read_fragment(want), dtype, out)

    def _read_across(self, n_body, n_tail, dtype, out):
        """One read that crosses the clamp: body, then the seam, then the tail."""
        body = self._read_track(self._plain, n_body, dtype,
                                None if out is None else out[:n_body])
        self._position += body.shape[0]
        if body.shape[0] < n_body:
            # libsndfile stopped short of its own clamp; there is no seam to
            # cross and nothing sensible to stitch on.
            return body

        rest = None if out is None else out[n_body:n_body + n_tail]
        if self._ensure_tail(body):
            tail = self._read_tail(n_tail, dtype, rest)
        else:
            self._fall_back('neither frame boundary reproduced the body')
            tail = self._read_track(self._scanned, n_tail, dtype, rest)
        self._position += tail.shape[0]
        return self._join(out, body, tail, n_body)

    @staticmethod
    def _join(out, body, tail, n_body):
        if out is not None:
            return out[:n_body + tail.shape[0]]
        if tail.shape[0] == 0:
            return body
        return np.concatenate((body, tail))

    def _read_tail(self, n_samples, dtype, out):
        if not self._ensure_tail(None):
            self._fall_back('the tail fragment would not open')
            return self._read_track(self._scanned, n_samples, dtype, out)
        if self._tail_position != self._position:
            self._seek_tail(self._position)
        return self._hand_out(self._read_fragment(n_samples), dtype, out)

    def _hand_out(self, data, dtype, out):
        """Account for fragment samples and put them where the caller wants them.

        The fragment is always decoded as float32 and copied, rather than read
        into the caller's buffer: it is at most a few minutes of audio at the
        end of a file, so the copy the body read avoids is not worth arranging
        here.
        """
        self._tail_position += data.shape[0]
        if dtype != 'float32':
            data = data.astype(dtype)
        if out is None:
            return data
        out[:data.shape[0]] = data
        return out[:data.shape[0]]

    @staticmethod
    def _read_track(track, n_samples, dtype, out):
        if out is None:
            return track.read(n_samples, dtype=dtype)
        return track.read(n_samples, dtype=dtype, out=out)

    def seek(self, sample_index):
        target = max(0, min(int(sample_index), self.frames))
        if self._layout is None:
            self._position = self._scanned.seek(target)
            return self._position

        body_end = min(self._clamp, self.frames)
        if target <= body_end:
            self._plain.seek(target)
            self._seek_target = target
            # The boundary that continues a decode depends on the decode it is
            # continuing, so a fresh seek retires the one that was chosen for
            # the last. Re-choosing costs one small scan, and only happens on a
            # read that actually crosses the seam.
            self._close_tail()
        elif self._ensure_tail_for_seek():
            try:
                self._seek_tail(target)
            except Exception as e:
                self._fall_back(f'{type(e).__name__}: {e}')
                self._position = self._scanned.seek(target)
                return self._position
        else:
            self._fall_back('the tail fragment would not open')
            self._position = self._scanned.seek(target)
            return self._position

        self._position = target
        return self._position

    def tell(self):
        return self._position

    def close(self):
        self._close_tail()
        if self._plain is not None:
            self._plain.close()
            self._plain = None
        if self._scanned is not None:
            self._scanned.close()
            self._scanned = None
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
                    # Decode straight into the shared segment. soundfile fills
                    # `out` in place and hands back a view of just the frames it
                    # actually read, so a short read at EOF still reports its
                    # true length without a second buffer.
                    channels = track.channels
                    shape = (n_samples,) if channels == 1 else (n_samples, channels)
                    view = np.ndarray(shape, dtype=np.dtype(dtype), buffer=shm.buf)
                    data = track.read(n_samples, dtype=dtype, out=view)
                    frames = data.shape[0]
                    del view, data
                    conn.send(('ok', frames))

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
