import av
import numpy as np

_DTYPE_MAP = {
    "float32": np.float32,
    "float64": np.float64,
    "int16": np.int16,
}


def _convert_dtype(samples, dtype):
    if dtype == "int16":
        scaled = np.clip(np.round(samples.astype(np.float64) * 32768.0), -32768, 32767)
        return scaled.astype(np.int16)
    return samples.astype(_DTYPE_MAP[dtype], copy=False)


class Driver:
    """Lazily streams decoded PCM samples out of the audio track of a .mp4 file,
    soundfile.SoundFile-alike. Video, if present, is never demuxed or decoded.

    Unlike ASF/WMA (see the sibling `stream-wma` project), MP4 audio streams use
    a sample-rate time_base, so a decoded frame's `pts` *is* an exact sample
    position -- confirmed empirically by counting `frame.samples` across a full
    linear decode of the test fixture (zero drift across ~1300 frames). No
    landmark cache of "observed position <-> pts" pairs is needed as a result;
    any pts value can be trusted as an exact sample count directly.

    What MP4/AAC does share with WMA: after *any* container-level seek --
    including a seek back to pts 0 -- the first frame the decoder emits is
    measurably corrupt (its content differs from a linear decode by ~0.5 in a
    +-0.5 amplitude signal, consistent with AAC's MDCT overlap-add needing the
    previous block's tail, which a seek can't supply). The frame after that one
    resyncs exactly. Confirmed by direct experiment against the fixture: 200
    random backward-seek targets, 0 real mismatches once the first
    post-seek frame is discarded and only the second is trusted.

    That discard has a consequence a naive port of WmaFile's algorithm misses:
    because a forward seek lands at the frame boundary *at or before* the
    target, the discarded (corrupt) frame is always the one that actually
    contains the target sample -- discarding it and trusting the next frame
    always overshoots the target by construction. The fix (and what the
    landmark-cache trick in stream-wma was really doing under the hood): seek
    to one frame *before* the target's frame, so the corrupt frame is a
    throwaway one frame early and the trustworthy frame after it is the one
    that actually covers the target. Since frame size is fixed for a given
    stream, this needs at most one retry (back off by exactly the discarded
    frame's `.samples`) -- verified empirically to converge in <=2 attempts
    across 150+ random targets.

    A target inside the very first frame (sample < frame size) has no earlier
    frame to seek to and back off from -- verified that even a seek to pts 0
    corrupts frame 0 just like any other seek. The only way to get an exact
    sample 0..frame_size-1 is a decoder that has *never* been seeked, so that
    case falls back to fully reopening the container from disk (confirmed
    bit-exact across repeated fresh opens) and decoding forward from true
    position 0.
    """

    def __init__(self, path):
        self._path = path
        self._container = av.open(path)
        try:
            self._stream = self._container.streams.audio[0]
        except IndexError as exc:
            self._container.close()
            raise ValueError(f"No audio stream found in {path!r}") from exc

        self.samplerate = self._stream.rate
        self.channels = self._stream.channels
        self.frames = self._estimate_frames()

        self._decoder = self._container.decode(self._stream)
        self._resampler = self._new_resampler()

        self._position = 0
        self._decode_pos = 0
        self._buffer = np.empty((0, self.channels), dtype=np.float32)
        # Newly decoded frames land here rather than being concatenated onto
        # _buffer one at a time -- with 1024-sample AAC frames, a per-frame
        # concatenate onto a chunk-sized buffer is O(frames^2) copying (a 50s
        # chunk decodes ~2300 frames). Merged into _buffer once per _consume
        # instead.
        self._pending = []
        self._pending_samples = 0
        self._eof = False

        self._container_seek_count = 0

    def _new_resampler(self):
        return av.AudioResampler(format="fltp", layout=self._stream.layout, rate=self._stream.rate)

    def _estimate_frames(self):
        stream = self._stream
        seconds = None
        if stream.duration is not None and stream.time_base is not None:
            seconds = float(stream.duration * stream.time_base)
        elif self._container.duration is not None:
            seconds = self._container.duration / 1_000_000
        if seconds is None:
            return 0
        return int(round(seconds * self.samplerate))

    def _append_frame(self, frame):
        arr = np.ascontiguousarray(frame.to_ndarray().T).astype(np.float32, copy=False)
        self._pending.append(arr)
        self._pending_samples += arr.shape[0]
        self._decode_pos += arr.shape[0]

    def _merge_pending(self):
        if not self._pending:
            return
        parts = self._pending if self._buffer.size == 0 else [self._buffer, *self._pending]
        self._buffer = np.concatenate(parts, axis=0)
        self._pending = []
        self._pending_samples = 0

    def _flush_resampler(self):
        for out_frame in self._resampler.resample(None):
            self._append_frame(out_frame)

    def _decode_one_step(self):
        try:
            raw_frame = next(self._decoder)
        except StopIteration:
            self._flush_resampler()
            self._eof = True
            return
        for out_frame in self._resampler.resample(raw_frame):
            self._append_frame(out_frame)

    def _ensure_buffer(self, n):
        while self._buffer.shape[0] + self._pending_samples < n and not self._eof:
            self._decode_one_step()
        self._merge_pending()

    def _consume(self, n):
        if n <= 0:
            return self._buffer[:0]
        self._ensure_buffer(n)
        n = min(n, self._buffer.shape[0])
        out = self._buffer[:n]
        self._buffer = self._buffer[n:]
        self._position += n
        return out

    def _reset_decode_state(self, seek_pts):
        self._container.seek(seek_pts, stream=self._stream, backward=True)
        self._container_seek_count += 1
        self._decoder = self._container.decode(self._stream)
        self._resampler = self._new_resampler()
        self._buffer = np.empty((0, self.channels), dtype=np.float32)
        self._pending = []
        self._pending_samples = 0
        self._eof = False

    def _reopen_fresh(self):
        # The only decoder state that reproduces an exact sample 0..frame_size-1
        # is one that has never been seeked -- re-open the container from disk
        # rather than trust `container.seek(0, ...)`, which is just as corrupt
        # for its first frame as any other seek (verified empirically).
        self._container.close()
        self._container = av.open(self._path)
        self._stream = self._container.streams.audio[0]
        self._container_seek_count += 1
        self._decoder = self._container.decode(self._stream)
        self._resampler = self._new_resampler()
        self._buffer = np.empty((0, self.channels), dtype=np.float32)
        self._pending = []
        self._pending_samples = 0
        self._eof = False
        self._decode_pos = 0
        self._position = 0

    def _land_on_or_before(self, target, max_attempts=6):
        boundary = target
        for _ in range(max_attempts):
            self._reset_decode_state(boundary)
            try:
                discard = next(self._decoder)
            except StopIteration:
                return None
            try:
                good = next(self._decoder)
            except StopIteration:
                good = None
            if good is not None and good.pts is not None and good.pts <= target:
                return good
            if discard.pts is None or discard.pts <= 0:
                return None
            boundary = max(0, discard.pts - discard.samples)
        return None

    def _seek_backward(self, target):
        good = self._land_on_or_before(target)
        if good is None:
            # Target falls inside the first frame (no earlier frame to back off
            # to), or resync didn't converge within budget -- always-correct,
            # slower fallback: true start, then count all the way forward.
            self._reopen_fresh()
        else:
            self._decode_pos = good.pts
            self._position = good.pts
            for out_frame in self._resampler.resample(good):
                self._append_frame(out_frame)

        if target > self._position:
            self._consume(target - self._position)

    def seek(self, sample_index):
        if sample_index < 0:
            raise ValueError("seek target must be non-negative")
        if sample_index == self._position:
            return
        if sample_index > self._position:
            # No container seek: decode-forward-and-discard from the live
            # decoder. Required fast path for sequential seek-per-chunk
            # access patterns (the common case) -- see _container_seek_count.
            self._consume(sample_index - self._position)
            return
        self._seek_backward(sample_index)

    def read(self, n_samples, dtype="float32"):
        if n_samples < 0:
            raise ValueError("n_samples must be non-negative")
        dtype = np.dtype(dtype).name
        if dtype not in _DTYPE_MAP:
            raise ValueError(f"Unsupported dtype: {dtype!r}")
        out = _convert_dtype(self._consume(n_samples), dtype)
        if self.channels == 1:
            out = out[:, 0]
        return out

    def tell(self):
        return self._position

    def close(self):
        self._container.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
