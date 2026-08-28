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
only fails when it has no file size to extrapolate from. So handing libsndfile
the file through a virtual-IO shim that **refuses to report its length** forces
the scan branch and yields an exact frame count. The file stays seekable;
everything downstream is ordinary libsndfile.

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

That is the whole of it, and for a year it was applied to the whole file. It is
now the driver's *fallback*, because doing it to the whole file costs far more
than it needs to.

### Only the tail is missing, so only the tail is scanned

Everything libsndfile clamps away is at the end — 34 s of a 5.6 h recording,
169 s of a 27.9 h one, 0.17% either way. The body is not in question at all, and
a plain `soundfile.SoundFile` reads it at full C speed. So:

- **The body is read plainly**, straight through libsndfile, with no Python in
  the read path. Measured over 894,175,796 samples read the way the streamer
  reads them: byte for byte what the scanned driver produced.
- **The length comes from arithmetic, not a scan.** A CBR stream's frames
  average exactly `144 * bitrate / samplerate` bytes — the padding bit is how an
  integer frame size tracks a non-integer average — so the audio byte count
  divided by that *is* the frame count. `_Layout` works it out and then checks
  itself: it predicts the byte offset of frames across the file and requires a
  real header at each, predicts the last frame and requires it to end where the
  audio does, and predicts libsndfile's own estimate and requires that too.
  Anything that fails takes the fallback.
- **The tail is read through the same shim, over a fragment.** It presents the
  file from a frame boundary a dozen frames before the clamp, so the scan it
  forces covers a couple of megabytes instead of gigabytes.

#### The seam

An mp3 decode that starts partway into a file is wrong for a frame or two — the
bit reservoir lets a frame borrow space from its predecessors — and then
converges. Measured across the corpus: two frames, always.

What is *not* always true is that it converges exactly. Of two adjacent frame
boundaries, one reproduces the continuous decode bit for bit and the other
leaves a residual of ~2.4e-7 that never dies away, because mpg123's synthesis
state ends up aligned differently and sums the same window in a different order.
Which of the two is right alternates with the start frame, with a phase that
varies between regions of the same file, so nothing predicts it.

So the driver measures it. It decodes a window the body has already produced
through both candidates and keeps the one that comes back identical. That costs
one extra decode of a few thousand samples, once per file, and turns a coin flip
into a check. Four of the eight corpus files want each boundary, which is the
best evidence that guessing was never going to do.

The fragment is also only ever read a whole frame at a time, because
libsndfile's mp3 output depends on where reads are broken: a read resumed
anywhere but a frame boundary differs from an unbroken one by ~6e-8 and never
converges back.

`benchmarks/mp3-driver-gpu/STEP1_RESULT.md` has all of those measurements, the
access patterns the read path is held to, and the three corners where it answers
differently from the old one on purpose.

### What it costs

Nothing, near enough. Opening a file is a division and a handful of small reads;
reading it is ordinary libsndfile; the tail costs one small scan and one small
verification at the end.

On the `Chia - Solar Eclipse` corpus — 8 files, 162.4 audio hours, 48 kbps CBR
mono, no Xing — through a full GPU analysis at 500 s chunks, 8 streamers, buffer
depth 8, one analyzer, `model_general_v3`. Rate is audio seconds over total wall
seconds, two runs of each arm, one at a time, page cache warmed first
(`benchmarks/mp3-driver-gpu/run_arms.py`):

| read path | rate | wall |
| --- | --- | --- |
| **tail scan (current)** | **3506x, 3454x** | 166.8 s, 169.3 s |
| plain soundfile, **loses 0.17%** | 3449x, 3453x | 169.2 s, 169.0 s |
| tail scan, forced into helper processes | 3408x, 3403x | 171.6 s, 171.8 s |
| whole-file scan (the driver before this) | 3002x, 2961x | 194.8 s, 197.5 s |

The whole-file scan cost ~15% of an analysis. The tail scan costs nothing
measurable: it is level with the plain reader that throws the tail away, and it
analyses more audio than that reader does. Per-file output is byte-identical
between the whole-file scan and the tail scan, in-process or through helpers,
on every file of that corpus.

### The helper process, and why it is now the exception

Every virtual-IO callback is a Python call, so a shim held over a whole file
holds the GIL for as long as the file is being read. On one streamer that is
invisible. On eight it was the whole analysis: the streamers stopped
parallelising and crawled together, and once the ONNX fusion made the analyzer
roughly 30x faster, a streamer stall that used to hide under TensorFlow
inference became the runtime.

The scan could not be moved on its own. libsndfile fixes an mp3's length at open
and clamps every later read and seek to it, and there is no API to inject a
length it already knows — so a subprocess could not compute the frame count and
hand the number back. The shim had to stay for the lifetime of the track. What
moved was therefore the whole decoder: `LocalDriver` is the in-process reader,
and `Driver` ran one inside a helper process, answering open/read/seek/tell over
a `Pipe` with sample data returned through shared memory (a 200 s chunk is
~35 MB, far too much to pickle through a pipe).

Streamers stay threads and the coordinator's shared state is untouched, which is
why this shape was chosen over making the streamers themselves processes:
`Coordinator.assigned_chunks` is a plain dict under a `threading.Lock`, and
`AssignFile.track` is an open file handle that cannot cross a process boundary.

With the body read plainly, there is nothing left holding the GIL, and the pipe
and the copy stop paying for themselves: 3462x in-process against 3402x through
helpers, on the corpus above. So `auto` now sends a file to a helper only when
that file is going to need the whole-file scan — a VBR stream, a file whose
layout will not validate — and reads everything else in-process.
`BUZZDETECT_MP3_HELPERS` still selects `always` or `never` outright.

A helper is leased for the lifetime of one open file and returned to a pool. A
helper that dies or stops answering is not fatal — the file reopens in-process,
seeks back to where it was, and the read is retried once — so the worst case is
the old behaviour, not a failed analysis.

For the record, from when the shim covered whole files: 8 threads opening 8
distinct ~5.4 h files and reading three 200 s chunks from each took 17.2 s
in-process against 5.05 s with helpers, and a full GPU analysis of those files
took 184 s against 61 s. That 3x is what the helper pool was built for, and it
is why it is still there for the files that still need it.

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
behaviour. So does the arithmetic that replaces the scan, which assumes
libsndfile extrapolates the length from the first frame. Both are checked at
runtime and both fall back rather than guess, and both are covered by
`engine/tests/test_mp3_driver.py`, which is the thing to run if libsndfile is
ever upgraded:

```
cd engine && .venv/bin/python3 tests/test_mp3_driver.py
```

A libsndfile that measured mp3 lengths properly would fail the estimate
prediction and put every file on the fallback: correct, and slower. If that ever
happens, `_agrees` is the place to notice it.

`BUZZDETECT_MP3_TAILSCAN=never` reads every file the old way — the whole-file
scan and the shim over everything. It exists so the two paths can be compared on
one corpus in one session, and as somewhere to retreat to if a file ever defeats
the layout arithmetic in a way its own validation does not catch.

Files already analyzed under the old estimate do not self-heal: their results
are marked complete and will be skipped. Delete the corresponding
`_buzzdetect.csv` to have them redone.
