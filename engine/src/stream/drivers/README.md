# Audio drivers

> **Provenance.** `mp3.py` and this document were written entirely by Claude
> (Opus 5) — the diagnosis, the measurements quoted throughout, and the code —
> working under the maintainer's direction. No line of either was authored by
> a human. This note covers only those two files; `wma.py`, `mp4.py` and
> `mts.py` predate them.

Format-specific readers that buzzdetect uses instead of `soundfile` where
`soundfile` is wrong, inaccurate, or unavailable for a format.

Each driver's own class docstring holds the empirical detail for that format —
measured drift, seek behaviour, what was tried and rejected. This file covers
what they have in common, and the one story that is too large for a docstring
(mp3).

## The contract

A driver is a class named `Driver` that behaves like `soundfile.SoundFile`:

| member | meaning |
| --- | --- |
| `Driver(path)` | open the file |
| `.samplerate` | int |
| `.channels` | int |
| `.frames` | length in samples — **best effort**, see below |
| `.read(n, dtype)` | 1-D for mono, `(n, channels)` otherwise; may return fewer |
| `.seek(sample)` | absolute sample position |
| `.tell()` | current sample position |
| `.close()` | release the file |

Two properties matter more than they look:

**`.frames` is advisory.** For most compressed formats the container's stated
length is an estimate. The streamer never trusts it as ground truth — it treats
a short read as the real end of audio (`handle_bad_read` in
`src/stream/worker.py`), warns if that happens far from the expected end, and
logs it as routine if it happens near it. A driver that over-reports slightly is
safe. A driver that *under*-reports silently loses audio, because the streamer
has no way to know there was more.

**A short read means "no more audio", not "error".** Corrupt packets — dead
batteries mid-recording are the usual cause — should end the stream, not raise.
The PyAV-based drivers catch `av.FFmpegError` and treat it as EOF.

`src/stream/driver.py` defines an `AudioDriver` ABC, but it is used only as a
type annotation on `AssignFile.track` and in `build_track`'s signature. Drivers
do not subclass it; they duck-type the contract above.

## Registration and precedence

`src/stream/audio.py` builds `driver_map` from this directory: one `.py` per
extension, named for the extension, each exporting `Driver`. Dropping in
`flac.py` claims `.flac`; nothing else needs editing.

Custom drivers are registered **first**, then `soundfile`'s formats fill in
whatever is left via `setdefault`. The precedence is deliberate and load-order
independent: `soundfile` advertises support for formats it cannot read
correctly to the end, so its support is the fallback, not the default. See mp3
below for the case that motivated this.

## Why each driver exists

- **`wma.py`** — ASF `pts` values are not sample-accurate (measured drift
  2,000–4,100 samples, not a fixed offset), so position must be counted forward
  from a known-exact point. Uses a landmark cache of `(exact position, pts)`
  pairs.
- **`mp4.py`** — MP4 uses a sample-rate `time_base`, so `pts` *is* an exact
  sample position and no landmark cache is needed. But AAC emits one corrupt
  frame after any container seek, so it seeks one frame early and discards.
- **`mts.py`** — AC3 never resyncs cleanly after a container seek (error
  persists at 0.3–1% of full scale indefinitely), so the only bit-exact path is
  decoding straight through from 0. Backward seeks reopen the container.
- **`mp3.py`** — libsndfile truncates long mp3s. Below.

## mp3: why we don't just use soundfile

### The problem

libsndfile never measures an mp3's length. It extrapolates from the **first
frame**:

```
frames = filesize * samples_per_frame // size_of_first_frame
```

MPEG-1 Layer III frames alternate between padded and unpadded sizes — 156 or
157 bytes at 44.1 kHz/48 kbps, true average 156.7347 — because the frame size
isn't an integer number of bytes and the encoder carries the remainder. If the
first frame happens to be a padded one, the estimate assumes every frame is
157 bytes and comes out **0.169% short**.

libsndfile then hard-clamps every read and seek to that estimate, so the tail
becomes unreachable. This is not a python-soundfile quirk; at the C API,
`sf_seek()` one frame past the estimate returns `-1 "Internal psf_fseek()
failed"`. No libsndfile consumer in any language can get past it.

Measured on a 49.7 h field recording: 178,625.5 s reported against 178,927.2 s
actually present. **301.7 s silently discarded** — from a file with 6,849,557
valid MPEG frames and zero sync anomalies. Not a corrupt file.

Direction depends on that first frame:

| first frame | estimate | consequence |
| --- | --- | --- |
| padded (157 B) | too low | **silent data loss** at the tail |
| unpadded (156 B) | too high | short read at EOF — benign, already handled |

### Which files are affected

Only mp3s with **no Xing/Info/VBRI header in the first MPEG frame**. That
header states the true frame count, so libmpg123 reads it directly and never
extrapolates.

- Raw CBR streams straight off a field recorder generally have no such header —
  these are the at-risk files.
- Anything LAME-encoded (including anything re-encoded by ffmpeg) carries one
  and is already exact.

Beware of false negatives when checking by hand: the bytes `Info` frequently
appear inside a leading ID3v2 tag as ordinary metadata. A real header sits at a
fixed offset inside the first MPEG frame — `frame_start + 4 + 17` for mono,
`+ 32` for stereo.

Across the `Pumpkin Audio Test` corpus, all 5 field recordings were affected
(72.9 h of audio, 7.4 minutes lost), while all 75 ffmpeg-derived clips in
`five_flowers_audio` were already exact. The padding bit is not a coin flip in
practice — the Sony recorder starts on a padded frame every time.

### The fix

libsndfile already contains an exact-length scan. It just never reaches it
(`src/mpeg_decode.c`):

```c
length = mpg123_length (mh) ;
if (length <= 0 && !psf->is_pipe)
{   if ((error = mpg123_scan (mh)) != MPG123_OK)
        return error ;
    length = mpg123_length (mh) ;
}
```

`mpg123_scan()` runs only when `mpg123_length()` fails, and `mpg123_length()`
only fails when it has no file size to extrapolate from. So the driver hands
libsndfile the file through a virtual-IO shim that **refuses to report its
length**, which forces the scan branch and yields an exact frame count. The
file stays seekable; everything downstream is ordinary libsndfile.

No patched binary, no vendored library, no new dependency.

Three details are load-bearing, and all three were found by breaking them:

1. **The shim must also skip the ID3v2 tag.** With no length to work from,
   libsndfile cannot navigate a leading tag and rejects the file outright as
   `Format not recognised`. The shim presents the first MPEG frame at virtual
   offset 0.
2. **Only a seek to the very end may lie.** libsndfile calls `tell()`
   constantly for ordinary bookkeeping; mis-answering those corrupts its
   parsing.
3. **`readinto` must be implemented.** python-soundfile's `vio_read` prefers it
   and falls back to `read()` otherwise; the fallback roughly doubles scan time
   across its millions of calls.

If the scan fails or returns an implausible count, the driver falls back to a
plain `soundfile.SoundFile(path)` — a wrong length is still better than
refusing to read the file, and the streamer's short-read handling covers it.

### What it costs

The scan reads the file once, ~3.7–5.9 s per GB, because libmpg123 makes two
I/O calls per MPEG frame (a 4-byte header read plus a body skip) and each one
crosses back into Python — 13.7 M calls on the 1 GB fixture.

It is close to free in practice, for two reasons. It only runs on files that
need it: a Xing-tagged file opens in 0.3 ms and skips the scan entirely. And
1 MB-buffered virtual IO reads *faster* than letting libsndfile open the path
itself, which claws back most of the scan on files that do pay it.

Streaming the full file in 200 s chunks, no inference:

| corpus | soundfile | this driver | PyAV driver |
| --- | --- | --- | --- |
| 49.7 h, no Xing | 57.8 s, **loses 301.7 s** | 58.8 s, complete | 77.5 s, complete |
| 75 × 300 s, Xing | 7.29 s, complete | 7.23 s, complete | 10.47 s, complete |

Output is bit-identical to plain `soundfile` at every offset tested, so
adopting the driver does not shift existing results.

### Why the read path runs in a helper process

Every one of those millions of virtual-IO callbacks is a Python call, so the
scan holds the GIL for as long as it runs. On one streamer that is invisible.
On eight it is the whole analysis: the streamers stop parallelising and crawl
together, and since the ONNX fusion made the analyzer roughly 30x faster, a
streamer stall that used to hide under TensorFlow inference became the runtime.

The scan cannot be moved on its own. libsndfile fixes an mp3's length at open
and clamps every later read and seek to it, and there is no API to inject a
length it already knows — so a subprocess cannot compute the frame count and
hand the number back. The shim has to stay in place for the lifetime of the
track. What moves is therefore the whole decoder: `LocalDriver` is the
in-process reader, and `Driver` normally runs one inside a helper process,
answering open/read/seek/tell over a `Pipe` with sample data returned through
shared memory (a 200 s chunk is ~35 MB, far too much to pickle through a pipe).

Streamers stay threads and the coordinator's shared state is untouched, which
is why this shape was chosen over making the streamers themselves processes:
`Coordinator.assigned_chunks` is a plain dict under a `threading.Lock`, and
`AssignFile.track` is an open file handle that cannot cross a process boundary.

A helper is leased for the lifetime of one open file and returned to a pool. A
helper that dies or stops answering is not fatal — the file reopens in-process,
seeks back to where it was, and the read is retried once — so the worst case is
the old behaviour, not a failed analysis. `BUZZDETECT_MP3_HELPERS` selects
`auto` (the default: a helper whenever another file is already open, which is
exactly when there is contention to lose), `always`, or `never`.

Measured on beelab-files, 8 threads opening 8 distinct ~5.4 h files and reading
three 200 s chunks from each:

| | wall | slowest open | mean open |
| --- | --- | --- | --- |
| in-process (`never`) | 17.2 s | 16.8 s | 8.98 s |
| helpers (`auto`) | **5.05 s** | **4.28 s** | **3.69 s** |

End to end, the same eight files (43.4 h of audio) through a full GPU analysis
with 8 streamers: **184 s in-process, 61 s with helpers** — 3.0x, and the
per-file output is byte-identical between the two.

As analysis rates those are **850x and 2565x**, which is the useful way to read
them: the streamer-count sweep in `benchmarks/streamer-grid` measures 815–846x
for a *single* streamer and 2665–2905x for six to twelve. So eight streamers
without helpers perform like one streamer — which is precisely the claim that
the length scan serialises them on the GIL — and with helpers they perform like
the six-to-twelve-streamer plateau.

### Alternatives that were tried and rejected

- **A PyAV/ffmpeg driver.** Correct, and it reports the length to within 3
  samples, but 1.3–1.4× slower on *every* chunk it ever reads — which costs far
  more than a one-time scan. The bottleneck is not the decoder (PyAV's decode
  loop alone matches soundfile's entire read) but getting samples out of
  ~7,656 `AVFrame` objects per chunk at ~2 µs each. `np.frombuffer`,
  `memoryview` memcpy, `np.asarray` and `AudioFifo` all landed within 1% of
  each other; `to_ndarray()` was 20% worse. Threading does not help — the bit
  reservoir makes MP3 frames serially dependent. It also decodes *slightly
  differently* from libmpg123 (up to 1.6e-2 on the fixture), so it would have
  changed existing results.
- **Lying about the file size** so the estimate comes out high enough. The
  open fails: libsndfile probes the last 128 bytes for an ID3v1 tag and rejects
  the file when that read comes up empty. Zero-padding past real EOF to satisfy
  it makes libmpg123 burn its resync budget and raise
  `Unspecified internal error` from inside `read()`, discarding the audio it
  had already decoded.
- **Upgrading libsndfile.** 1.2.2 is the newest release on Homebrew,
  conda-forge and PyPI; the code path above is current upstream master.

### Caveats

The trick depends on an interaction that upstream does not document as an API —
it is a consequence of how `mpeg_decode.c` falls back, not a promised
behaviour. It is guarded by the fallback described above, and is worth a
fixture test if libsndfile is ever upgraded.

Files already analyzed under the old estimate do not self-heal: their results
are marked complete and will be skipped. Delete the corresponding
`_buzzdetect.csv` to have them redone.
