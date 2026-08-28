# Plan: scan only the tail, read the body plainly

A worked proposal for removing the mp3 driver's ~10% remaining cost on GPU
analyses. Read `HANDOFF.md` first for how that cost was measured and what has
already been ruled out.

This is a plan, not a decision. **Step 1 is a feasibility gate that can kill
it**, and killing it early is a good outcome — write down the result either way.

## The idea in one paragraph

libsndfile's mp3 length is an underestimate *and* a hard clamp, so the driver
hides the file's length to force a full `mpg123_scan()`. That scan is not the
expensive part in steady state — the expensive part is that the shim keeps
libsndfile's I/O routed through Python for the file's whole life, ~38k
callbacks per chunk. But **only the tail is ever missing** (0.17%, ~34 s on a
5.6 h file). So: read the body through a *plain* `SoundFile` at full C speed up
to libsndfile's clamp, then open a second `SoundFile` over a shim that presents
only the last few megabytes, and read the remainder from that. The scan then
runs over ~2 MB instead of gigabytes, and the Python read path disappears from
everything except the last few seconds.

## Success criteria

Define these before writing code, and hold to them.

1. **Correctness is non-negotiable.** For every fixture file, the concatenated
   output must be **bit-identical** to what today's `LocalDriver` produces for
   the same file. Not "close" — identical. If it cannot be bit-identical,
   see "If the seam won't converge" below; do not quietly ship a version that
   changes results.
2. **Total frames must match** today's `LocalDriver.frames` exactly.
3. **Speed target: ≥3450x** on the Solar Eclipse corpus at chunk 500 /
   8 streamers / depth 8, against 3193x today and 3540x for plain soundfile
   (which loses the tail). Below ~3300x the change is not worth its complexity.
4. **No regression on the many-small-files shape.** Re-run the streamer grid
   (`benchmarks/streamer-grid/`) and confirm the Diel Drivers corpus does not
   get slower; that corpus is open-dominated rather than read-dominated, and it
   is where the current helper design earns its keep.

## Step 1 — the feasibility gate: does a mid-stream decode converge?

**Everything depends on this and nothing else should be built first.**

MP3 has a bit reservoir: a frame's data can live partly in the preceding
frame's slack space. So a decode started mid-stream produces wrong samples for
the first frame or two, then should converge on the same output as a
continuous decode. The whole plan rests on "then converges", and on that
convergence being *exact*.

The experiment:

- Take a fixture file. Decode it fully with today's `LocalDriver`; keep the
  samples from some sample offset `S` onward as ground truth.
- Find a valid MPEG frame boundary at a byte offset comfortably *before* the
  byte that corresponds to `S` (see Step 2 for finding boundaries).
- Open a shim presenting the file from that byte offset, decode forward.
- Align the two by content and report: **after how many samples do they become
  bit-identical, and do they ever?**

Run it at several offsets — near the start, mid-file, and just before the
clamp — and on at least one file per recorder model present in the corpora.

**Gate:** convergence must be exact and must happen within a bounded, small
number of frames (a few thousand samples). Record the worst case observed;
that number becomes the mandatory overlap in Step 3, with a generous margin.

If it converges only approximately, stop and read "If the seam won't converge".

## Step 2 — locating a frame boundary

The fragment shim must start on a real MPEG frame header or libsndfile will
reject the file ("Format not recognised" — the same failure the ID3 skip
already exists to avoid).

Do **not** compute the offset arithmetically from the bitrate. These files are
CBR, but CBR frames still vary by one byte via the padding bit, so
`frame_index * frame_size` drifts. Instead scan for the sync word (11 set bits,
`0xFF Ex`) near the target offset and **validate** the candidate: parse the
header, check version/layer/bitrate/samplerate are sane and match the file's,
then check that the *next* header lands exactly where this frame's computed
size says it should. Requiring two or three consecutive valid headers before
accepting a candidate is what separates a real boundary from audio data that
happens to contain `0xFF`.

Write this as a standalone function with its own unit test: for a fixture,
assert that every offset it returns is one that libsndfile will actually open,
and that scanning from N different starting points converges on the same
boundaries.

## Step 3 — the driver change

Keep it inside `engine/src/stream/drivers/mp3.py`. Luke's preference is that
this work stays in the Claude-authored driver rather than in engine internals
he maintains himself.

Shape (`LocalDriver` only; `Driver` and the helper pool are unchanged, and
should keep working untouched):

- `__init__`: open plainly. Record `clamp_frames = track.frames`. Determine the
  **true** frame count. Two options, and this is a real decision:
  - scan as today (correct, costs the scan once per file — but the scan is a
    per-file cost that the current design already pays, so this is not a
    regression), or
  - derive it from the tail fragment alone. Cheaper, but you must prove it
    gives exactly today's `frames` on every fixture before trusting it.
  Start with the first; it isolates the change to the read path, which is
  where the win is. Only try the second if the scan turns out to matter.
- `read()`: serve from the plain track while the position is below
  `clamp_frames`. On crossing, open the tail fragment once, seek it to the
  overlap start, decode and discard the overlap, then serve the remainder.
- `seek()`: must work across the boundary in both directions, including
  seeking backwards from the tail into the body. This is the fiddliest part
  and the most likely source of bugs — see the test plan.
- `close()`: close both tracks and both shims. Watch for the fragment being
  opened lazily and never opened at all on a short file.

Files where libsndfile's estimate is already correct (Xing-tagged, or short
files) should never open a fragment at all. Assert that in a test.

## Step 4 — test plan

Rigour here matters more than speed. The current driver is a perfect oracle:
it produces the correct full decode, so every claim below is checkable rather
than arguable.

**Unit level, fast, no GPU:**

1. *Frame-boundary finder* — as in Step 2.
2. *Bit-identity, whole file.* For each fixture, read the entire file in chunks
   through the new driver and through today's `LocalDriver`; assert
   `np.array_equal` over the whole stream and equal total length. Do this at
   several chunk sizes, including one that lands a chunk boundary exactly on
   the clamp and one that straddles it — off-by-one at the seam is the defect
   this catches.
3. *Seek matrix.* For a grid of seek targets (0, mid-body, clamp−1, clamp,
   clamp+1, mid-tail, last frame, past EOF) and each pair of consecutive
   targets, assert position and subsequent samples match the oracle. Include
   backwards seeks from the tail into the body.
4. *Short reads and EOF.* Reading past the end returns empty from both; a read
   straddling EOF returns the same short length from both. `verify_helper.py`
   is the existing pattern.
5. *Degenerate files.* A file whose estimate is already exact; a very short
   file (smaller than the fragment window); a file with an ID3v2 tag; a file
   with no tag. Each must work or fall back cleanly.
6. *Helper equivalence.* Repeat (2) with `BUZZDETECT_MP3_HELPERS=always` — the
   helper must return exactly what in-process returns, as it does today.

Fixtures: use small excerpts committed alongside the tests, not the 3 GB
corpus files, so the suite is runnable anywhere. Cut them so at least one
genuinely exhibits the truncation (verify with `clamp_test.py`).

**Integration level, on the corpora:**

7. *Output identity.* Full analysis of `Chia - Solar Eclipse` with the new
   driver; diff the per-file CSVs against a run with today's driver. Expect
   byte-identical. Anything else is a bug, not a rounding difference — the
   analyzer is deterministic for a given input.
8. *Rate.* Same corpus and settings as `HANDOFF.md` (chunk 500, 8 streamers,
   depth 8, 1 GPU analyzer) so numbers are directly comparable to the table
   there. Use `run_arm.py`; it has the `__main__` guard that a spawn-context
   helper requires.
9. *The other corpus shape.* Re-run at least the best and recommended cells of
   `benchmarks/streamer-grid` (12×1200 and 6×200) against
   `results_zerocopy.csv`, to catch a regression on open-dominated work.

**Measurement hygiene** — these all produced wrong numbers at least once during
the original investigation:

- Run one thing at a time. Microbenchmarks running alongside an analysis
  inflated a result by 60% before it was caught.
- Any harness that the driver's spawn context might re-import **must** have an
  `if __name__ == '__main__'` guard. Without it every helper re-runs the whole
  analysis and you measure several runs fighting over one GPU — it looks like a
  plausible slowdown, not like a bug.
- Fresh output directory per run; the engine skips files it has already
  analysed.
- Warm the page cache before timing, or the first run pays for cold reads.
- On this box, pin `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu` or the run dies
  on its first `FusedConv` (`docs/linux-gpu.md`).
- Report rate as **audio seconds / total wall seconds**. `local/evaluate_log.R`
  computes a steady-state rate that reads ~8% higher; do not mix them.

## If the seam won't converge

If Step 1 shows a mid-stream decode never becomes bit-identical, the plan as
written is dead, because it would silently change results. Options then, worst
to best:

- **Overlap and crossfade.** Rejected: it changes samples deliberately, and
  this project has held a hard line on decode parity (the driver was adopted
  partly *because* it is bit-identical to soundfile, and fp16 was declined over
  a 1.6e-2 drift).
- **Fragment-only for the tail, full scan for anything that reads it.** i.e.
  keep today's path for files that are actually truncated and use the plain
  path for files that are not. Cheap, safe, and it helps exactly the files that
  do not need help — probably not worth much, but measure what fraction of a
  real corpus is Xing-tagged before dismissing it.
- **Attack the callback count instead.** The shim's cost is ~38k crossings into
  Python per chunk. A C extension serving `vio_read` would remove them, at the
  cost of a build dependency in a project that currently freezes cleanly with
  PyInstaller. Weigh that seriously before starting; it is a distribution
  problem more than a coding one.

## Rough effort

Step 1 is a day at most and decides everything. Steps 2–4 are the bulk. Do not
start Step 3 before Step 1's gate has passed and its worst-case convergence
number is written down.
