# Streamer count vs chunk length

How fast does an analysis actually go, as a function of how many streamer
threads feed the analyzer and how much audio each chunk carries?

The headline number is the **analysis rate**: audio seconds divided by wall
seconds for the whole run, start to finish. Not inference throughput — wall
time is what a user waits, and it includes interpreter startup, the inference
session build, the mp3 length scans, and the write-out. A rate of 3000x means
an hour of recording is analysed in 1.2 s.

## Running it

```
python3 run_grid.py --dry-run                 # show the corpus, run nothing
python3 run_grid.py --out results.csv         # the full grid
python3 run_grid.py --streamers 6 --chunklengths 200 --repeat 3
```

Results append to the CSV and completed cells are skipped on a re-run, so an
interrupted sweep resumes; `--force` re-runs everything.

A GPU sweep needs an interpreter whose onnxruntime carries a GPU execution
provider — `engine/.venv` is CPU-only by design, so point `--python` at one
that has `onnxruntime-gpu`:

```
python3 run_grid.py \
  --python /path/to/gpu-venv/bin/python3 \
  --ld-library-path /usr/lib/x86_64-linux-gnu
```

`--ld-library-path` is not optional on a machine carrying more than one cuDNN.
This one does (a pip `nvidia-cudnn-cu12` wheel that an `/etc/ld.so.conf.d`
entry puts ahead of the system install), and unpinned the run dies on its first
`FusedConv` with `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`. See
`engine/src/inference/onnx.py`.

## What it controls for

- **A fresh output directory per cell.** The engine skips files it has already
  analysed, so a reused output folder would measure an empty run.
- **Page cache.** The corpus is read through once before the first cell, so the
  first cell isn't charged for reading cold what every later cell gets from
  cache. `--no-warmup` opts out.
- **Enough files to keep the streamers busy.** Files are taken one per
  subfolder, round-robin across the corpus dirs, to a duration budget
  (`--target-hours`) but never fewer than the largest streamer count. With
  fewer files than streamers a cell measures idle threads, not contention.

## Failure handling

Cells that fail are recorded and the sweep continues. `status` is one of:

| status | meaning |
| --- | --- |
| `ok` | ran to completion; `rate_x` is filled in |
| `oom` | ran out of memory — the engine said so, or the kernel killed it |
| `timeout` | exceeded `--timeout`; the whole process group is killed |
| `failed` | anything else, with the reason in `note` |

Only `ok` rows get a rate: a partial run's rate depends on how far it got,
which would flatter or damn a setting arbitrarily.

Large chunk lengths are the OOM risk, and it lands in two different places.
The inference session is built for a whole chunk, so device memory scales with
`--chunklength`; separately, host memory scales with streamers × chunk × queue
depth, since every buffered chunk is resampled audio sitting in RAM. The engine
names chunk length in the first case (`WorkerInferer.process_chunk`); in the
second the kernel's OOM killer usually gets there first, which is why `classify`
reads a bare SIGKILL as an OOM rather than a crash.

## Columns

`rate_x` is the metric; the rest explain it. `audio_s` is summed from the
engine's own `file_start` progress events, so it is exact rather than estimated
from file sizes. `wall_s` is measured around the subprocess and `engine_s` is
what the engine reported for itself — the gap between them is startup.
`peak_rss_mb` is the engine process's high-water mark, which excludes the mp3
helper processes (they hold one decode buffer each; the chunks that dominate
memory are queued in the parent).

## Results

Run 2026-08-27 on beelab-files (GTX 1650 4GB, 23GB RAM, corpus on a local
NTFS-via-FUSE disk): 78 files, 314 audio hours, one GPU analyzer,
`model_general_v3`. `results.csv` is the raw data.

Analysis rate (audio seconds per wall second):

| streamers | 100s | 200s | 600s | 1200s |
| --- | --- | --- | --- | --- |
| 1 | 815x | 822x | 846x | 811x |
| 6 | 2665x | 2726x | 2810x | 2863x |
| 12 | 2676x | 2814x | 2781x | **2905x** |

Peak RSS (MB):

| streamers | 100s | 200s | 600s | 1200s |
| --- | --- | --- | --- | --- |
| 1 | 904 | 992 | 1149 | 1405 |
| 6 | 1371 | 1616 | 2526 | 4014 |
| 12 | 1787 | 2331 | 4508 | 7434 |

No cell ran out of memory, including the 12 x 1200s corner that came closest at
7.4GB.

**Streamer count is the only setting that matters, and it stops mattering at
six.** One to six is 3.3x. Six to twelve is 1%, and not monotonic — at 600s
chunks twelve streamers are *slower* than six while using 1.8x the memory.
There is no reading of this grid where more than six streamers pays.

**Chunk length barely moves the rate and dominates memory.** Across a 12x
range it buys 7% at six streamers, while peak RSS goes up 2.9x. The rate
differences across a row are small enough to be within run-to-run noise; the
memory differences are not.

The one clean effect is that longer chunks and more streamers are the same
lever: both add buffered audio. Memory tracks streamers x chunk length closely
(12 x 1200 is 8.2x the memory of 1 x 100), which is why the two settings should
be chosen together rather than separately.

At one streamer the rate is flat at ~820x whatever the chunk length, because a
single decoder thread is the ceiling and the GPU is idle most of the time. That
number is worth remembering: it is what the whole analysis collapses to if the
streamers stop running in parallel, which is exactly what the mp3 length scan
used to do to them before the helper process landed (see
`engine/src/stream/drivers/README.md` -- eight streamers without helpers
measure 850x, indistinguishable from one).

### After the tail-scan read path (2026-08-28)

The grid above was run with the mp3 driver forcing libsndfile to scan every
file in full and reading it through a Python shim for the rest of its life.
That is gone: the driver now reads the body plainly and shims only the tail
(`engine/src/stream/drivers/README.md`). Re-running the two cells the grid
recommends and the two it says are best, same corpus, same machine
(`results_tailscan.csv`):

| cell | whole-file scan | tail scan |
| --- | --- | --- |
| 12 streamers, 1200 s | 2905x | **3316x** |
| 6 streamers, 1200 s | 2863x | **3289x** |
| 12 streamers, 200 s | 2814x | **3110x** |
| 6 streamers, 200 s | 2726x | **3080x** |

10-15% everywhere, on a corpus of 78 files where opening is a much larger share
of the work than it is on the eight long files of `Chia - Solar Eclipse`. That
is the shape of corpus the whole-file scan hurt most, since its cost was paid
once per file regardless of how much of the file was read.

The conclusions above are unchanged: the shape of the grid is the same, six
streamers is still the knee, and chunk length still buys little for a lot of
memory. Only the whole surface moved up.

### What to set

**Six streamers and 200s chunks** is the recommendation: 2726x, 94% of the
best cell in the grid, for 1.6GB -- 22% of the memory the best cell needs. If
memory is genuinely free, 6 x 600s gets 97% at 2.5GB. The 12 x 1200s corner
wins the grid by 3% and costs 4.6x the memory to do it, which is not a trade
worth making on a machine that might be doing anything else.

The engine's current default is `n_analyzers * 8` streamers on GPU, so a
one-analyzer GPU run gets eight. That lands in the flat region and is not
costing anything measurable -- but it is above the knee, and on a memory-tight
machine six would be the better default.

### Caveats

One run per cell, so single-digit-percent differences between adjacent cells
are not resolvable -- `--repeat` exists for when that matters. All of it is one
GPU analyzer on one card; the streamer knee should be expected to move with
analyzer count, since what the streamers are feeding is what sets how fast they
need to be.
