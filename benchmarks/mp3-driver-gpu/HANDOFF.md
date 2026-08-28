# The mp3 driver costs ~13% of a GPU analysis

Status as of 2026-08-27. Read the plain-English summary, then the numbers.

## Plain English

Analyses on this machine used to run at about **3600x** (an hour of recording
analysed in a second). They now run at about **3100x**. That is real, and here
is why.

Long mp3s lie about their length. libsndfile guesses a recording's duration
from its first frame, and on these recorders the guess is short — about 0.17%
short, which is 34 seconds off a 5.6-hour file and several minutes off a 50-hour
one. Worse, it doesn't just *report* the wrong length, it *refuses to read* past
it. That audio is simply unreachable.

The mp3 driver exists to fix that, and it does: it hides the file's length from
libsndfile, which forces libsndfile to scan the file properly and find the true
end. Nothing is lost any more.

The catch is how it hides the length. It hands libsndfile a Python object to
read the file through, instead of letting libsndfile open the file itself. The
mp3 decoder asks for data roughly twice per frame of audio, which is about
**38,000 requests per chunk** — and every one of them now has to bounce up into
Python. Python can only run one thread at a time, so with eight streamer threads
all bouncing, they stop working in parallel and start queueing behind each
other. The GPU then sits idle waiting to be fed.

The driver already works around this by doing the reading in a **separate
process** (its own Python, its own lock), which recovers most of the loss —
without it the analysis runs at 779x instead of 3193x. But shuffling the audio
between processes costs something, and that residue is the ~13% still missing.

**None of this showed up before** because the driver was written in the
TensorFlow era, when the model itself was the slow part and the reading was
free by comparison. Making inference ~30x faster turned the reading into the
bottleneck. It was also never tested on GPU.

**What changed tonight:** the driver was copying every block of audio twice on
its way between processes. It now decodes straight into the shared buffer, one
copy instead of two. That is worth ~4% (3075x → 3193x) with byte-for-byte
identical results.

**What's left:** ~10%, and the promising fix is sketched at the bottom.

## The numbers

Corpus: `Chia - Solar Eclipse`, 8 files, **162.4 audio hours**, 48 kbps CBR
mono 44.1 kHz. Settings: chunk 500 s, 8 streamers, buffer depth 8, one GPU
analyzer, `model_general_v3`. Rate is **audio seconds / total wall seconds**.

| read path | rate | wall | starvation |
| --- | --- | --- | --- |
| plain soundfile (pre-driver behaviour, **loses 0.17%**) | **3540x** | 164.9 s | 3.9 s / 55 waits |
| mp3 driver + helper, decode into shm (**current**) | **3193x** | 183.2 s | 16.8 s / 158 waits |
| mp3 driver + helper, double copy (before tonight) | 3075x | 190.1 s | 23.0 s / 221 waits |
| mp3 driver, in-process shim (`BUZZDETECT_MP3_HELPERS=never`) | 779x | 750.3 s | 596.4 s / 669 waits |

Historical reference, same corpus and settings, TensorFlow era (2025-09,
`local/gpu_tuning/logs/procwrite/chunklength_500_depth_8_streamers_8.log`):
**3599x**. Plain soundfile today is 3540x — i.e. **the regression is entirely
the mp3 driver**, not the ONNX migration. Per-chunk analyzer time is
essentially unchanged: 159.4 ms now vs 151.9 ms then.

Per-chunk read cost, single-threaded (`bench_read.py`), 500 s @ 44.1 kHz:

| | time | rate |
| --- | --- | --- |
| plain soundfile read | 472.9 ms | 1057x |
| driver, in-process shim | 553.3 ms | 904x |
| driver, helper process | 583.1 ms | 857x |
| resample 44.1k→16k (soxr HQ) | 288.2 ms | — |

Note a single streamer only produces ~657x, so eight of them have barely 1.5x
headroom over a 3600x GPU. There is not much slack to lose.

## What was ruled out

- **mmap-backed shim** (`bench_shim.py`). Serving libsndfile's reads from a
  memoryview over an mmap instead of a `BufferedReader`: **worse** — open +64%,
  reads +9%. `BufferedReader.readinto` is already good C. The shim's problem is
  the *number* of crossings into Python, not the cost of each, so making each
  cheaper was never going to be enough.
- **Moving the resample into the helper** (`bench_soxr_gil.py`). Would only pay
  if soxr held the GIL. It doesn't — 1.00 / 1.99 / 3.38x on 1 / 2 / 4 threads;
  the flattening at 8 is core saturation on an 8-core box, not lock contention.
- **"Just read past libsndfile's estimate"** (`clamp_test.py`). Not possible:
  reads stop dead at the estimate and `seek()` past it raises
  `Internal psf_fseek() failed`. It is a hard clamp, not a bad label.
- **Skipping the redundant pad in `OnnxModel.predict`.** Real (7.7 ms/chunk,
  5.6% of analyzer time) but worth only ~1% end-to-end, because the analyzer
  has slack — and it perturbed 366 of 7.9M outputs by 0.01 through cuDNN
  picking a different kernel on a differently-aligned buffer. Reverted.

## The promising fix: scan only the tail

**A full plan with a test strategy is in `TAIL_SCAN_PLAN.md`.** The summary below
is the idea; that document is how to execute it, including the feasibility gate
that can kill it on day one.

The scan's cost is proportional to the bytes scanned, and **only the tail is
ever missing**. The shim already proves libsndfile will happily open a
file-like that begins at an arbitrary MPEG frame boundary — that is exactly
what its ID3 skip does. So:

1. Open the file **plainly** and stream it at full soundfile speed (1057x per
   thread, no Python in the read path) up to libsndfile's clamp.
2. For the missing tail, open a second `SoundFile` over a shim presenting only
   the last few MB. That scan runs over ~2 MB instead of gigabytes.
3. Read the remainder from the fragment and stitch.

This inverts the cost model: today the tax scales with file length, which is
the worst possible shape for 50-hour recordings. The target is the 3540x plain
soundfile number *with* the tail intact.

The hard part is the seam. MPEG decoders need a few frames of warm-up, so the
first samples out of the fragment are not bit-identical to a continuous decode.
Every file in both corpora is **48 kbps CBR**, where byte offset maps to frame
index exactly, so start the fragment well before the clamp, decode through the
overlap and discard it. VBR would need more care — decide whether to detect it
and fall back to the current path.

Correctness is checkable: the current driver gives a ground-truth full decode
to diff against, and `verify_helper.py` is the pattern for that.

Also still open, cheaper but smaller: the parent still copies the block out of
shared memory (`view.copy()` in `_read_helper`). The streamer resamples it
immediately and never holds it, so returning the view would save another 88 MB
memcpy per chunk — but it would hand the caller a buffer the next read
overwrites, which is a real contract change and needs thought.

## Running any of this

Everything here needs a GPU interpreter and, on this box, a pinned loader (two
cuDNN 9.x installs are visible — see `docs/linux-gpu.md`):

```
cd engine
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
  /path/to/gpu-venv/bin/python3 ../benchmarks/mp3-driver-gpu/run_arm.py \
  {helper|nohelper|soundfile} /tmp/out
```

`run_arm.py` has a `__main__` guard and it is load-bearing: the helper uses a
spawn context, so without it every helper re-runs the whole analysis and you
measure several concurrent runs fighting over one GPU. That mistake produced a
believable-looking 320 s "result" during this session.

Rates from a log: `compare.py <wall_seconds|-> <log glob>` reports both the
total-audio/total-wall rate and the steady-state rate that `local/evaluate_log.R`
computes (that one drops the first chunk and measures between chunk timestamps,
so it reads ~8% higher; don't mix the two).
