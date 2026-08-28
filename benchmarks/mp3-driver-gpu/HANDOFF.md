# The mp3 driver cost ~13% of a GPU analysis. It no longer does.

Status as of 2026-08-28. The first half of this document is the investigation as
it stood on 2026-08-27; the last section is what was done about it.

## Plain English

Analyses on this machine used to run at about **3600x** (an hour of recording
analysed in a second). They dropped to about **3100x**. That was real, and here
is why.

Long mp3s lie about their length. libsndfile guesses a recording's duration from
its first frame, and on these recorders the guess is short — about 0.17%, which
is 34 seconds off a 5.6-hour file and several minutes off a 50-hour one. Worse,
it doesn't just *report* the wrong length, it *refuses to read* past it. That
audio is simply unreachable.

The mp3 driver exists to fix that, and it does: it hides the file's length from
libsndfile, which forces libsndfile to scan the file properly and find the true
end. Nothing is lost any more.

The catch was how it hid the length. It handed libsndfile a Python object to
read the file through, instead of letting libsndfile open the file itself. The
mp3 decoder asks for data roughly twice per frame of audio, which is about
**38,000 requests per chunk** — and every one of them had to bounce up into
Python. Python can only run one thread at a time, so with eight streamer threads
all bouncing, they stopped working in parallel and queued behind each other. The
GPU then sat idle waiting to be fed.

The driver worked around that by doing the reading in a **separate process**
(its own Python, its own lock), which recovered most of the loss — without it
the analysis ran at 779x instead of 3193x. But shuffling the audio between
processes cost something, and that residue was the ~13% still missing.

**None of this showed up before** because the driver was written in the
TensorFlow era, when the model itself was the slow part and the reading was free
by comparison. Making inference ~30x faster turned the reading into the
bottleneck. It was also never tested on GPU.

## The numbers, as of the investigation

Corpus: `Chia - Solar Eclipse`, 8 files, **162.4 audio hours**, 48 kbps CBR mono
44.1 kHz. Settings: chunk 500 s, 8 streamers, buffer depth 8, one GPU analyzer,
`model_general_v3`. Rate is **audio seconds / total wall seconds**.

| read path | rate | wall | starvation |
| --- | --- | --- | --- |
| plain soundfile (pre-driver behaviour, **loses 0.17%**) | 3540x | 164.9 s | 3.9 s / 55 waits |
| mp3 driver + helper, decode into shm | 3193x | 183.2 s | 16.8 s / 158 waits |
| mp3 driver + helper, double copy | 3075x | 190.1 s | 23.0 s / 221 waits |
| mp3 driver, in-process shim (`BUZZDETECT_MP3_HELPERS=never`) | 779x | 750.3 s | 596.4 s / 669 waits |

Historical reference, same corpus and settings, TensorFlow era (2025-09,
`local/gpu_tuning/logs/procwrite/chunklength_500_depth_8_streamers_8.log`):
**3599x**. Plain soundfile was 3540x — i.e. **the regression was entirely the
mp3 driver**, not the ONNX migration. Per-chunk analyzer time was essentially
unchanged: 159.4 ms against 151.9 ms.

Per-chunk read cost, single-threaded (`bench_read.py`), 500 s @ 44.1 kHz:

| | time | rate |
| --- | --- | --- |
| plain soundfile read | 472.9 ms | 1057x |
| driver, in-process shim | 553.3 ms | 904x |
| driver, helper process | 583.1 ms | 857x |
| resample 44.1k→16k (soxr HQ) | 288.2 ms | — |

A single streamer only produces ~657x, so eight of them had barely 1.5x headroom
over a 3600x GPU. There was not much slack to lose.

## What was ruled out

- **mmap-backed shim** (`bench_shim.py`). Serving libsndfile's reads from a
  memoryview over an mmap instead of a `BufferedReader`: **worse** — open +64%,
  reads +9%. `BufferedReader.readinto` is already good C. The shim's problem was
  the *number* of crossings into Python, not the cost of each, so making each
  cheaper was never going to be enough.
- **Moving the resample into the helper** (`bench_soxr_gil.py`). Would only pay
  if soxr held the GIL. It doesn't — 1.00 / 1.99 / 3.38x on 1 / 2 / 4 threads;
  the flattening at 8 is core saturation on an 8-core box, not lock contention.
- **"Just read past libsndfile's estimate"** (`clamp_test.py`). Not possible:
  reads stop dead at the estimate and `seek()` past it raises
  `Internal psf_fseek() failed`. It is a hard clamp, not a bad label.
- **Skipping the redundant pad in `OnnxModel.predict`.** Real (7.7 ms/chunk,
  5.6% of analyzer time) but worth only ~1% end-to-end, because the analyzer has
  slack — and it perturbed 366 of 7.9M outputs by 0.01 through cuDNN picking a
  different kernel on a differently-aligned buffer. Reverted.

## What was done: scan only the tail

The scan's cost is proportional to the bytes scanned, and **only the tail is
ever missing**. So the body is opened and read plainly, at full C speed with no
Python in the read path, and only the audio past libsndfile's clamp comes
through a shim — over a fragment presenting the last couple of megabytes rather
than the whole file. The length no longer needs a scan at all: a CBR stream's
frames average exactly `144 * bitrate / samplerate` bytes, so the file's size
divides straight into a frame count.

The plan is in `TAIL_SCAN_PLAN.md`, its gate and everything the gate turned up
are in **`STEP1_RESULT.md`**, and the driver's own README
(`engine/src/stream/drivers/README.md`) is where the design now lives.

Same corpus and settings, two runs of each arm, one at a time, page cache warmed
first (`run_arms.py`):

| read path | rate | wall |
| --- | --- | --- |
| **tail scan (current)** | **3506x, 3454x** | 166.8 s, 169.3 s |
| plain soundfile, **loses 0.17%** | 3449x, 3453x | 169.2 s, 169.0 s |
| tail scan, forced into helper processes | 3408x, 3403x | 171.6 s, 171.8 s |
| whole-file scan (the old driver) | 3002x, 2961x | 194.8 s, 197.5 s |

The gap is closed: the driver is now level with the plain reader that throws the
tail away, while keeping the tail, and per-file output is byte-identical to the
old driver's on every file of the corpus (`diff_results.py`).

The helper pool survives but has changed job. With nothing left holding the GIL,
a process boundary only costs a pipe and a copy, so `auto` now sends a file to a
helper **only when that file will need the whole-file scan** — a VBR stream, or
one whose layout will not validate.

Two smaller things that were open and are now moot:

- Returning the shared-memory view from `_read_helper` instead of copying it
  would have saved an 88 MB memcpy per chunk. That copy only happens on the
  helper path, which is now the exception, so the contract change it needed is
  not worth making.
- The scan itself is gone for any file with a valid layout, which was the other
  per-file cost.

## Running any of this

Everything here needs a GPU interpreter and, on this box, a pinned loader (two
cuDNN 9.x installs are visible — see `docs/linux-gpu.md`):

```
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
  python3 benchmarks/mp3-driver-gpu/run_arms.py default scan soundfile --repeat 2
```

`run_arms.py` warms the page cache, gives each arm a fresh output directory, and
runs one at a time; all three matter. `run_arm.py`, which it drives, has a
`__main__` guard that is load-bearing: the helper pool uses a spawn context, so
without it every helper re-runs the whole analysis and you measure several
concurrent runs fighting over one GPU. That mistake produced a
believable-looking 320 s "result" during the original session.

Rates from a log: `compare.py <wall_seconds|-> <log glob>` reports both the
total-audio/total-wall rate and the steady-state rate that `local/evaluate_log.R`
computes (that one drops the first chunk and measures between chunk timestamps,
so it reads ~8% higher; don't mix the two).
