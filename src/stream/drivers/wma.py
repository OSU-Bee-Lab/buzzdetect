import bisect

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
    """Lazily streams decoded PCM samples out of a .wma/ASF file, soundfile.SoundFile-alike.

    Sample-accurate seek/read is built on top of PyAV, whose container-level
    seeks are only accurate to "nearest preceding packet" and whose frame
    `pts` values are not sample-accurate for ASF (measured drift of ~2000-4100
    samples across a 47-frame fixture -- not a fixed offset, so it cannot be
    corrected for arithmetically). The only trustworthy position signal is a
    real count of `frame.samples` accumulated forward from a position that is
    already known exactly. Position 0 is always exact (first sample the
    decoder ever emits); every other exact position is established the same
    way: by decoding forward from an already-exact position and counting.
    A landmark cache of (exact_sample_position, frame.pts) pairs, recorded as
    frames are decoded, lets backward/cold seeks jump near a target via a
    real container seek and then decode-forward-and-discard only the small
    remainder to land on the target exactly.
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
        # _buffer one at a time -- with small ASF frames, a per-frame
        # concatenate onto a chunk-sized buffer is O(frames^2) copying.
        # Merged into _buffer once per _consume instead.
        self._pending = []
        self._pending_samples = 0
        self._eof = False

        # Implicit landmark: sample 0 is always reachable by seeking to pts 0.
        self._landmark_positions = [0]
        self._landmark_pts = [0]

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
        # Header duration is a best-effort estimate (measured ~0.1% optimistic
        # vs. truly decodable samples on the fixture) -- read()'s short-read-
        # at-EOF handling is what actually makes truncation safe, not this.
        return int(round(seconds * self.samplerate))

    def _record_landmark(self, pos, pts):
        if pts is None:
            return
        idx = bisect.bisect_left(self._landmark_positions, pos)
        if idx < len(self._landmark_positions) and self._landmark_positions[idx] == pos:
            return
        self._landmark_positions.insert(idx, pos)
        self._landmark_pts.insert(idx, pts)

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
        # Landmark position is recorded *before* this frame's output is
        # appended, so it is the exact position of that output's first
        # sample -- not derived from pts, just paired with it for lookup.
        self._record_landmark(self._decode_pos, raw_frame.pts)
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

    def _resync(self, expected_pts, max_discards=8):
        # After a container-level seek, WMA's decoder needs one packet of
        # prior context it doesn't have: the *first* frame decoded is
        # measurably corrupt (its pts matches no frame from a linear decode,
        # and its content differs) -- confirmed by direct experiment on the
        # fixture. The frame after that resyncs exactly (same pts, same
        # samples) with the frame a plain forward decode would have produced
        # at that point. So: seek to the landmark *before* the one we want,
        # discard frames until pts matches the target landmark's pts, and
        # only then trust position counting again.
        for _ in range(max_discards):
            try:
                frame = next(self._decoder)
            except StopIteration:
                self._eof = True
                return
            if frame.pts == expected_pts:
                self._record_landmark(self._decode_pos, frame.pts)
                for out_frame in self._resampler.resample(frame):
                    self._append_frame(out_frame)
                return
        # Resync didn't land where expected within budget -- fall back to
        # the one seek target known not to need warmup (true start) and
        # count all the way forward. Always correct, just slower.
        self._reset_decode_state(0)
        self._decode_pos = 0
        self._position = 0

    def _seek_backward(self, target):
        idx = bisect.bisect_right(self._landmark_positions, target) - 1
        if idx < 0:
            idx = 0

        if idx == 0:
            self._reset_decode_state(self._landmark_pts[0])
            self._decode_pos = self._landmark_positions[0]
            self._position = self._decode_pos
        else:
            self._reset_decode_state(self._landmark_pts[idx - 1])
            self._decode_pos = self._landmark_positions[idx]
            self._position = self._decode_pos
            self._resync(self._landmark_pts[idx])

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
