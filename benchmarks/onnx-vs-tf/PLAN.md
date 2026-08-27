# Benchmark: ONNX vs TensorFlow on CUDA

Instructions for an agent. Target machine: Ubuntu with an NVIDIA GPU, CUDA and
cuDNN already installed. Run everything from the `buzzdetect` checkout.

> **Revised 2026-08-27** after auditing the repo against this plan on the target
> machine. The v1 plan was written from a laptop and got several things wrong
> about this box. Changes are marked **[revised]**; the reasoning is kept, not
> just the conclusion, so a future reader can tell which constraints are real.

## What this is for

`model_general_v3` ships as both an ONNX graph and a TensorFlow SavedModel, and
`DualRuntimeModel` will run either (`engine/src/inference/models.py`). The
shipped desktop app carries only the ONNX half. There is a suspicion that ONNX
is ~30% slower than TensorFlow on CUDA. If that reproduces, it justifies
building a way for power users to bring their own TensorFlow to the app; if it
doesn't, the idea gets dropped.

You are measuring only. Change no engine code — the benchmark scripts live
outside the repo and only call into it. Report a number and the evidence for it.

Note what is actually being compared: the two runtimes use **different YAMNet
trunks** — `yamnet_onnx` vs `yamnet_k2` — and the trunk is nearly all of the
compute (the model itself is one dense layer over the embedding). So this is a
YAMNet-under-CUDA comparison, not a probe comparison. Say so in the report.

**[revised] There is a third arm.** `models/model_general_v3_onnx/` already
holds `model_combined.onnx`: a single graph taking a raw 1-D waveform
(`input_1: ['unk__409']`) to `[N, 13]` class scores, with the log-mel front end
**fused into the graph**. This matters because the two ONNX paths differ from
TensorFlow in a way the v1 plan didn't account for:

| arm | model | front end | trunk + head |
|---|---|---|---|
| `tensorflow` | `model_general_v3` | in the Keras graph, on the GPU | GPU |
| `onnx` | `model_general_v3` | **NumPy, on the CPU** (`embedders/yamnet_onnx/features.py`) | onnxruntime |
| `onnx_fused` | `model_general_v3_onnx` | in the ONNX graph | onnxruntime |

So if `onnx` loses to `tensorflow`, the CPU-side featurisation is a prime
suspect, and `onnx_fused` is the control that isolates it. That is a much
cheaper fix than a TensorFlow path in the app, so establish it before
concluding anything.

## 0. [revised] Facts about this machine, already checked

Don't re-derive these.

- **GPU**: GeForce GTX 1650, 4 GB, driver 580.173.02 (CUDA 13 capable). This is
  the workstation the ~30% suspicion came from — it is not underpowered for the
  question, and the 4 GB is only a constraint in that TensorFlow must be given
  `set_memory_growth` or it takes the whole card.
- **Typical throughput**: ~3000 audio-seconds per wall second, with
  `--n_streamers 12`. Use 12.
- **No system CUDA 12.** `/usr/local/` has only `cuda-11.8`, too old for
  TF ≥ 2.16, and `ldconfig` finds `libcudart.so.12` only inside an unrelated
  project's venv. System cuDNN 9 *is* present, but that isn't enough on its own.

## 1. Environment

Neither existing requirements file works here: `requirements.txt` pins CPU
`onnxruntime`, and `requirements-onnx-cuda.txt` has no TensorFlow. You need one
venv with both.

```
cd engine
uv venv --python 3.12 .venv-bench
VIRTUAL_ENV=.venv-bench uv pip install \
    "tensorflow[and-cuda]>=2.16" "keras>=3.14" \
    "onnxruntime-gpu[cuda,cudnn]==1.26.0" \
    numpy pandas soundfile soxr av matplotlib onnx
```

**[revised] Expect this to take a long time — 4+ GB of downloads.** You are
installing two independent CUDA stacks; `tensorflow[and-cuda]` and
`onnxruntime-gpu[cuda,cudnn]` each pull their own `nvidia-*` wheels, several of
them 400 MB–1 GB. It is download-bound, not stuck. Run it in the background.

**[revised] The v1 alternative is dead.** v1 suggested trying plain
`tensorflow` (no `[and-cuda]`) so TF uses the system CUDA/cuDNN, with wheels
only from onnxruntime. There is no system CUDA 12 here (see §0), so that path
fails. The wheel route is the only one.

The `onnxruntime-gpu` pin is deliberate — see the comment in
`requirements-onnx-cuda.txt`. If pip cannot find a consistent set, **stop and
report the conflict** rather than downgrading `onnxruntime-gpu` to the 1.27
line or dropping the NVIDIA wheels — either would silently change what is being
measured.

**[revised] If it does conflict, there is a proven fallback.** The repo's own
`environment/` conda env already holds TensorFlow 2.20.0 and
`onnxruntime-gpu` 1.22.2 side by side on Python 3.13, and both reach the GPU
today:

```
TF GPUs: [PhysicalDevice('/physical_device:GPU:0')]
ort providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

It proves coexistence is possible. Prefer `.venv-bench` with the 1.26 pin so
you measure the version that ships; fall back to `environment/` only if
resolution fails, and say in the report that the ORT version differs.

## 2. Prove both runtimes actually reach the GPU

This is the step that makes or breaks the result. A CPU-bound run looks
completely normal in the logs. Do not skip it, and do not proceed if either
check fails.

```
python buzzdetect_cli.py --probe_gpu
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

The first must include `CUDAExecutionProvider` in its JSON. The second must
print a non-empty list. If TF prints an empty list, its own CUDA/cuDNN
resolution is the problem — check `nvidia-smi` and TF's stderr, which names the
library it failed to `dlopen`.

Record `nvidia-smi` output (driver, GPU model), `tf.__version__`,
`onnxruntime.__version__`, and the Python version in the report.

## 3. [revised] Audio

Corpus: `/media/server storage/experiments/Luke - Diel Drivers/` — 473 mp3
files across 16 date directories, ~5.8 h each (44.1 kHz mono), ~2750 h total.
Plenty; the constraint is choosing how much.

**One directory is not enough, and not for the reason you'd guess.** Each date
subdirectory holds exactly **one** file, and `analyze.py:350` queues one
`AssignFile` per file — so a single-file corpus leaves 11 of 12 streamers idle,
and §5's remedy for a starved analyzer (raise `--n_streamers`) has nothing to
bite on.

At ~3000×, 90 files ≈ 520 h of audio ≈ 10 minutes of wall clock per run. Build
a flat symlink corpus (`build_corpus.py`); `search_dir` uses `os.walk`, which
lists symlinked files normally. Index-prefix the basenames — they are not
unique across date directories, and results are keyed on the path under
`--dir_audio`.

## 4. The runs

**[revised] Three arms, not two**, and the two end-to-end figures are each
compromised in a different direction (§5), so **the targeted inference
benchmark in §4a is the primary measurement**, not a nice-to-have.

### 4a. Targeted inference benchmark (primary)

`bench_inference.py`. Decode one 200 s chunk once, then time the model over
~30 repeats after ~5 warmups, in a **separate process per arm** — the two
runtimes would otherwise contend for the 4 GB card. Report median.

Four things it must get right or the number is fiction:

1. **Force TensorFlow to sync.** Keras eager ops return once the work is
   *enqueued* on the CUDA stream, not when it completes. Timing without
   materialising the result (`np.asarray`) measures dispatch, not compute.
   onnxruntime's `run()` is already synchronous.
2. **Time the TF path both ways** — with and without that `np.asarray`. The
   difference sizes the confound in §5 and is worth reporting on its own.
3. **Warm up.** cuDNN autotune and the first CUDA context land on iteration 1.
4. **Same input array, same batch shape** for every arm — 200 s at
   `framehop_prop 1` is ~208 frames, matching production.

Break out, per arm: whole `predict`, trunk alone (`embedder.embed`), head alone
(`predict_embeddings`). For the `onnx` arm additionally split
`features.waveform_to_patches` (CPU) from the trunk session — that split is the
whole reason the third arm exists.

`model_general_v3_onnx` is a plain `BaseModel`, not a `DualRuntimeModel`: it has
no `.runtime`, and it inherits `uses_tensorflow = True` despite being pure
onnxruntime. Don't trust either attribute. It also declares
`embeddername = 'yamnet_k2'`, so **loading it imports TensorFlow** even though
it never runs a TF op — set `set_memory_growth` in that arm too, or an idle TF
takes the card.

### 4b. End-to-end runs (secondary)

Same everything except the arm. Use a **separate `--dir_out` per run**:
`src/stream/worker.py:62` skips any file whose completed results already exist,
so reusing the directory would make the second run finish instantly and look
like a total victory. **This applies per repeat, not just per arm.**

```
BUZZDETECT_RUNTIME=onnx python buzzdetect_cli.py \
  --modelname model_general_v3 \
  --dir_audio <CORPUS> \
  --dir_out ../local/bench/onnx-1 \
  --analyzers_gpu 1 --analyzers_cpu 0 \
  --n_streamers 12 \
  --chunklength 200 \
  --framehop_prop 1 \
  --verbosity_print PROGRESS --verbosity_log DEBUG --log_progress true \
  < /dev/null
```

Then `BUZZDETECT_RUNTIME=tensorflow` (same model), and
`--modelname model_general_v3_onnx` for the fused arm. Redirect stdin from
`/dev/null`: the manifest prompt only fires on drift, and fresh output
directories won't trigger it, but `interrupt.py` watches stdin when it isn't a
tty and should be given a clean EOF.

`_pick_runtime` raises rather than falling back, so an env var that can't be
honoured fails loudly — good. Confirm from the log that each run used the
runtime you asked for (the analyzer logs `processing on GPU`; the embedder in
use distinguishes them).

Run each arm **at least twice** and report all timings. Discard nothing, but
note which was first — the first run of either pays for cold file cache.

## 5. Reading the numbers

Two figures per run, both from the log file that lands in `--dir_out`
(`analyze.py:139`):

- **`Total analysis time: Ns`** — end of the run.
- **Per-chunk `rate:`** in the PROGRESS lines — audio seconds per wall second.
  Take the median across chunks, skipping the first few per worker.

**[revised] Both end-to-end figures are biased, in opposite directions.** v1
said to lead with the rate. Don't — lead with §4a, and use these two as a
bracket around it.

- **`rate:` favours TensorFlow.** `report_rate` stops `timer_analysis` and
  restarts it (`inference/worker.py:87`) before the loop blocks on
  `get_analyze()`, so the wait for the next chunk is inside the timed span —
  *and* the TF result is still an unmaterialised `tf.Tensor` at that point
  (`assignments.py:39`). It is only forced to host memory later, in another
  thread, by `np.asarray` in `write/worker.py:69`. So the TF analyzer can stop
  its timer before the GPU has finished, and the sync cost lands where nothing
  times it. ONNX pays that cost inside the timer.
- **`Total analysis time` penalises TensorFlow.** It is immune to the above —
  the run cannot end until the writer has drained every chunk, so all GPU work
  is inside the wall clock — but it includes startup, and TF's import plus CUDA
  context is much slower than onnxruntime's.

**The rate is only meaningful if the GPU was fed.** `timer_analysis` includes
any wait for the next assignment, so a starved analyzer reports a low rate that
has nothing to do with inference. That is what `BUFFER BOTTLENECK` DEBUG lines
mark (`src/inference/worker.py:89`). **[revised]** Note the threshold is 10 ms
(`worker.py:116`), which at ~3000× is ~15% of a chunk's GPU time — so a run can
look clean and still be meaningfully starved. Watch `nvidia-smi dmon` during a
run as a second opinion. If bottleneck lines appear, raise `--n_streamers`
(16, then 24) and rerun every arm.

### [revised] Parity cannot be checked from the result CSVs

v1 asked for the max absolute difference between output CSVs, against the
5.6e-05 agreement checked at export. That is not measurable there:
`write/formatting.py:33` does `np.array(results).round(digits_results)`, and
`model_general_v3` sets `digits_results = 2`. Every difference would come out
0.00 or 0.01.

Instead, have the microbenchmark dump each arm's **unrounded** prediction array
to `.npy` and compare those at full precision — max abs difference, mean, and
top-class agreement — pairwise across all three arms. Keep a CSV comparison of
the full runs as a coarse sanity check only, and label it as such.

A parity failure invalidates the timing question and is the more important
finding.

## 6. Report

Write findings to `local/ONNX_vs_TF_results.md`:

- environment (versions, driver, GPU)
- **[revised]** the §4a microbenchmark table: per arm, median whole-`predict`,
  trunk, head; plus the ONNX featurise/trunk split; plus the TF
  synced-vs-unsynced gap
- both end-to-end figures for each run, all repeats, with the §5 bias caveat
  attached to each
- median rate per arm and the ratios
- bottleneck lines seen, and what `--n_streamers` it took to clear them
- **[revised]** three-way full-precision parity from the `.npy` dumps
- a one-line verdict: does ~30% reproduce?

**[revised] If ONNX does lose badly**, the ordered list of things to try is now:

1. **The fused graph** (`onnx_fused`). If it closes the gap, the CPU-side NumPy
   front end was the cause and the fix is already built.
2. Opset and conversion settings in
   `buzzdetect-training/tools/export_onnx.py`, and IO binding.
3. A TensorFlow path in the app — last, not first.

### [revised] Before recommending the fused graph, check macOS

A win on CUDA does not automatically transfer:

- **Windows + CUDA: yes, almost certainly.** Same `CUDAExecutionProvider`, same
  graph, same kernels.
- **macOS + CoreML: uncertain, possibly a regression.** The CoreML EP
  partitions a graph by op support and runs unsupported regions on the CPU,
  with a transfer at each boundary. MobileNet convs are well supported; a
  log-mel front end (STFT/FFT, log, mel matmul) is much less likely to be, and
  an unsupported region *in the middle* of the graph is the bad case — today's
  arrangement, where featurisation happens once in NumPy and the trunk is one
  unbroken CoreML partition, may well be better. There is also the fp16/Neural
  Engine path to lose, worth ~3.5× (`src/inference/onnx.py:45-47`).

Dump the fused graph's op types and opsets as evidence, and state plainly that
the macOS question needs measuring on a Mac. If the fused graph ships, it
should be *alongside* `model_general_v3`, not instead of it —
`shipped-models.txt` lists model names, so both can ship and the platform can
pick.

## Scripts

Outside the repo (nothing here modifies engine code):

| script | does |
|---|---|
| `build_corpus.py` | flat symlink corpus of N files |
| `bench_inference.py` | §4a, one arm per invocation, dumps JSON + `.npy` |
| `compare_npy.py` | full-precision parity between two arms |
| `run_endtoend.sh` | §4b, three arms × two repeats, then a summary table |
| `compare_parity.py` | coarse CSV comparison of full runs |
| `inspect_onnx.py` | op/opset inventory, for the CoreML question |
| `run_all.sh` | all of the above in order; hard-fails on §2 |

## Cleanup

`local/bench/` and `engine/.venv-bench` are yours to delete. Leave
`engine/.venv`, `engine/environment/`, `engine/models/`, and anything under
`engine/audio_in/` alone. **[revised]** The symlink corpus in
`local/bench/corpus` points into `/media/server storage/` — delete the
directory of links, and never anything they resolve to.
