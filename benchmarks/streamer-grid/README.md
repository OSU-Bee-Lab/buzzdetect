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
