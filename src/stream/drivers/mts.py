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
    """Lazily streams decoded PCM samples out of the audio track of an .mts
    (MPEG-TS, typically AC3) file, soundfile.SoundFile-alike. Video, if
    present, is never demuxed or decoded.

    Unlike MP4/AAC, a container-level seek back into an AC3 stream does not
    resync cleanly: even after discarding the first post-seek frame, every
    subsequent frame differs from a linear decode by ~0.3-1% of full scale
    and never converges to zero -- confirmed empirically against the test
    fixture (50 frames sampled after a mid-stream seek, error persists at
    the same order of magnitude throughout). AC3 carries decoder state
    (e.g. bit-reservoir/gain smoothing) that a container seek can't restore,
    so no landmark-cache trick can make a seeked decoder trustworthy here.

    The only decode path confirmed bit-exact is a container opened fresh and
    decoded straight through from position 0 (same guarantee the MP4/WMA
    drivers fall back on for their unresolvable cases). This driver uses
    that path exclusively: forward seeks discard-consume from the live
    decoder (cheap, no container seek), and backward seeks reopen the
    container from disk and count forward from true position 0.
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
        self._buffer = np.empty((0, self.channels), dtype=np.float32)
        # Newly decoded frames land here rather than being concatenated onto
        # _buffer one at a time -- with small AC3 frames, a per-frame
        # concatenate onto a chunk-sized buffer is O(frames^2) copying.
        # Merged into _buffer once per _consume instead.
        self._pending = []
        self._pending_samples = 0
        self._eof = False

        self._container_reopen_count = 0

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

    def _reopen_fresh(self):
        self._container.close()
        self._container = av.open(self._path)
        self._stream = self._container.streams.audio[0]
        self._container_reopen_count += 1
        self._decoder = self._container.decode(self._stream)
        self._resampler = self._new_resampler()
        self._buffer = np.empty((0, self.channels), dtype=np.float32)
        self._pending = []
        self._pending_samples = 0
        self._eof = False
        self._position = 0

    def seek(self, sample_index):
        if sample_index < 0:
            raise ValueError("seek target must be non-negative")
        if sample_index == self._position:
            return
        if sample_index > self._position:
            # No container seek: decode-forward-and-discard from the live
            # decoder. Required fast path for sequential seek-per-chunk
            # access patterns (the common case) -- see _container_reopen_count.
            self._consume(sample_index - self._position)
            return
        # Backward seek: no AC3 container seek is trustworthy (see class
        # docstring), so the only exact path is a fresh decode from true
        # start, counting forward to the target.
        self._reopen_fresh()
        self._consume(sample_index)

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
