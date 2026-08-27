# ONNX vs TensorFlow on CUDA — results

Run 2026-08-27 on `beelab-files`. Branch `bench/onnx-vs-tf`. Scripts in this
directory; raw outputs in the gitignored `local/bench/`.

**Verdict: the ~30% suspicion does not reproduce as stated, and the cause is
not the one anyone expected.** ONNX as shipped is 3.4x slower than TensorFlow,
not 30%. Almost all of that is a CPU-side spectrogram that has nothing to do
with onnxruntime. What remains after fixing it is a *shared* problem — two
convolution layers that run ~100x off hardware peak in onnxruntime, in every
ONNX graph we have.

## Environment

| | |
|---|---|
| GPU | NVIDIA GeForce GTX 1650, 4 GB, driver 580.173.02 |
| CPU | 8 cores |
| Python | 3.12.3 (`engine/.venv-bench`) |
| tensorflow | 2.21.0 |
| keras | 3.15.1 |
| onnxruntime-gpu | 1.26.0 (the shipped pin) |
| onnx | 1.22.0 |

Both runtimes were confirmed on the GPU before measuring: `--probe_gpu`
reported `CUDAExecutionProvider`, and `tf.config.list_physical_devices('GPU')`
reported the card.

## 1. Inference benchmark (primary)

One 200 s chunk (209 YAMNet frames), median of 30 repeats after 5 warmups, one
process per arm. Rates are audio-seconds per wall-second.

| arm | whole `predict` | trunk | head | init | rate | vs TF |
|---|---|---|---|---|---|---|
| `tensorflow` (`yamnet_k2`) | **52.6 ms** | 52.4 ms | 2.4 ms | 2.35 s | 3806 | 1.00x |
| `onnx_fused` (`model_combined.onnx`) | **67.7 ms** | *(fused)* | *(fused)* | 0.23 s | 2955 | 1.29x slower |
| `onnx` as shipped (`yamnet_onnx`) | **177.0 ms** | 177.3 ms | 0.3 ms | 0.38 s | 1130 | 3.37x slower |

The shipped ONNX path splits as:

```
onnx_featurize_cpu   147.6 ms   (83%)  NumPy log-mel front end, on the CPU
onnx_trunk_only       42.4 ms   (17%)  MobileNet in onnxruntime, on the GPU
```

**The runtime was never the problem.** onnxruntime's trunk (42.4 ms) is faster
than TensorFlow's entire embed (52.4 ms). The shipped path loses because it
computes the spectrogram in NumPy on the CPU, which `yamnet_k2` does inside the
graph on the GPU.

## 2. Numerical parity

Full precision, from the unrounded `.npy` dumps — **not** from the result CSVs,
which `write/formatting.py:33` rounds to `digits_results` (2 for this model) and
which therefore cannot resolve anything below 0.01.

| pair | max abs diff | mean abs diff | top class agrees |
|---|---|---|---|
| onnx vs tensorflow | 2.5e-05 | 4.56e-06 | 100.0000% |
| onnx_fused vs tensorflow | 2.15e-05 | 1.17e-06 | 100.0000% |
| onnx_fused vs onnx | 2.26e-05 | 4.45e-06 | 100.0000% |

Tighter than the 5.6e-05 claimed at export. Parity is not in question, so the
timing comparison is valid.

## 3. Where the time actually goes (onnxruntime per-node profiler)

Profiling the fused graph, 200 s input, everything on `CUDAExecutionProvider`
(no CPU partition):

| op | ms/run | % |
|---|---|---|
| **Conv** | **54.88** | **73.6%** |
| Mul (STFT windowing) | 8.52 | 11.4% |
| GlobalAveragePool | 2.37 | 3.2% |
| Gemm (classifier head) | 1.64 | 2.2% |
| FusedMatMul (**the DFT**) | **0.83** | **1.1%** |
| everything else | ~5.5 | ~7% |

And the Conv time is two layers:

```
layer2/depthwise_conv    29.87 ms   40.1%
layer1/conv              20.69 ms   27.8%
the other 25 convs        ~4.3 ms    ~6%
```

The standalone trunk has the identical pathology, so this is **not** an export
artifact:

| layer | fused graph | standalone `yamnet.onnx` |
|---|---|---|
| layer2 depthwise | 29.87 ms | 29.03 ms |
| layer1 conv | 20.69 ms | 18.58 ms |
| other 25 convs | ~4.3 ms | ~3.3 ms |

For scale: layer 1 is roughly 0.18 GFLOP. On a ~2.9 TFLOP card that is well
under a millisecond of arithmetic. It is taking 20 ms — about two orders of
magnitude off peak. Depthwise convolutions are a known weak spot for cuDNN;
TensorFlow ships its own hand-tuned depthwise kernels, which is the most likely
reason it wins at all.

### Hypothesis tested and REJECTED: the missing FFT

The op inventory shows no `STFT` or `DFT` op — the spectrogram is a dense
matrix multiply against a `[2, 257, 512]` constant (`cst_rfft_512__52`), where
TensorFlow's `tf.signal.stft` uses a real FFT. A dense DFT is O(N^2) against
the FFT's O(N log N), so this looked like the whole story.

**It is not.** That MatMul costs **0.83 ms — 1.1% of the run.** The naive DFT
is entirely affordable on a GPU at this size. Do not spend effort re-exporting
for an FFT; it would buy at most a fraction of a millisecond.

*(This corrects §2 of the first draft of NEXT_STEPS.md, which recommended
exactly that.)*

## 4. CUDA provider tuning — no effect

`providers_for()` (`src/inference/onnx.py:84`) passes `{}` today, so cuDNN
algorithm selection and tensor layout are at their defaults. Depthwise convs
are the classic case where those defaults are wrong, so this looked promising.
It isn't:

| config | median | vs default |
|---|---|---|
| default (what ships) | 65.9 ms | 1.00x |
| `cudnn_conv_algo_search=EXHAUSTIVE` | 65.9 ms | 1.00x |
| `cudnn_conv_algo_search=HEURISTIC` | 66.0 ms | 1.00x |
| `cudnn_conv_use_max_workspace=1` | 65.9 ms | 1.00x |
| `prefer_nhwc=1` | 66.5 ms | 0.99x |
| nhwc + EXHAUSTIVE | 66.8 ms | 0.99x |
| nhwc + EXHAUSTIVE + workspace | 66.7 ms | 0.99x |

Every knob is within noise, and NHWC is marginally *worse*. Provider options do
not reach whatever is wrong here.

One lead surfaced from the session logs, not yet chased:

```
13 Memcpy nodes are added to the graph tf2onnx_tf2onnx for CUDAExecutionProvider.
It might have negative impact on performance (including unable to run CUDA graph).
```

## 5. The async-dispatch confound — real in principle, worthless in practice

`report_rate` stops `timer_analysis` (`inference/worker.py:87`) before the TF
result is materialised, since `np.asarray` only happens later in another thread
(`write/worker.py:69`). In theory that lets the TF analyzer stop its timer
before the GPU has finished, flattering its per-chunk `rate:`.

Measured, by timing `predict` with and without forcing `np.asarray`:

| arm | synced | unsynced | gap |
|---|---|---|---|
| tensorflow | 52.6 ms | 52.9 ms | none (noise) |
| onnx | 177.0 ms | 177.3 ms | none |
| onnx_fused | 67.7 ms | 67.9 ms | none |

**No measurable bias.** The end-to-end `rate:` figures need no discount. Worth
recording so nobody re-derives the worry.

## 6. End-to-end runs — started, deliberately abandoned

Six runs were planned (3 arms x 2 repeats, 90-file symlink corpus, ~520 h of
audio). Stopped during run 1 by decision, once profiling redirected the work.

Run 1 (`onnx`, 12 streamers) was ~74% through after 31 minutes, and produced
one result worth keeping:

| | |
|---|---|
| `BUFFER BOTTLENECK` lines | 1,854 |
| median wait | 0.10 s (p90 0.20 s, max 7.40 s) |
| total analyzer idle | 217 s of ~1,850 s (**12%**) |
| effective throughput | ~756 audio-s/wall-s |
| same path, in isolation | 1,129 audio-s/wall-s |

The analyzer was idle only 12% of the time yet ran at 67% of its solo speed.
That gap is **CPU contention, not starvation**: the shipped ONNX path computes
its spectrogram in the analyzer thread, competing with 12 decoder threads for 8
cores. So the CPU front end costs twice — 147.6 ms per chunk directly, and
cores stolen from the streamers that feed it.

Note that §5 of the plan prescribes raising `--n_streamers` when bottleneck
lines appear. **On this machine that is the wrong move** — 12 streamers on 8
cores is already oversubscribed, and more would add contention, not decode
capacity.

**Unanswered, and it matters:** whether real-world throughput is capped by audio
decoding rather than inference. If it is, the fused model delivers far less
than its 2.6x to actual users, and TensorFlow would deliver almost nothing on
top. Worth resuming for `tensorflow` and `onnx_fused` before committing to any
plan that rests on inference speed. Estimated ~40 min per run on this hardware.

## 7. The trap that nearly produced a false positive

The first ONNX run silently fell back to the CPU:

```
CUDNN_BACKEND_TENSOR_DESCRIPTOR cudnnFinalize failed
cudnn_status: CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH
... Falling back to ['CPUExecutionProvider'] and retrying.
```

This box has cuDNN **9.25.0.15** installed system-wide (found by `ldconfig`) and
**9.24.0.43** as a wheel in the venv. onnxruntime loaded cuDNN's sub-libraries
from both. It caught the failure, switched to CPU, and carried on.

- Cost: the trunk timed **214.7 ms** instead of **42.4 ms** — a 5x error.
- Direction: it would have *confirmed* the 30% suspicion.
- **`get_providers()` still reported `CUDAExecutionProvider` throughout.** The
  plan's §2 GPU check does not catch this.

Fix is `LD_LIBRARY_PATH` pointing at the venv's own `nvidia/*/lib`, which the
runner scripts now do (commit `d3d9d0a`). TensorFlow is immune because it
preloads its own copies — which is also why the fused arm looked healthy before
the fix: it imports TensorFlow for its declared embedder and inherited TF's
resolution.

## What to do about it

1. **Ship the fused ONNX model.** 177 ms -> 67.7 ms for every user, no size
   cost (14 MB replaces 56 KB + a 13 MB embedder), no new dependency, and it
   loads 10x faster than TensorFlow. Two small blockers in NEXT_STEPS §1.
2. **Chase the two convolutions.** ~50 ms of every ONNX run, in every graph.
   If they can be brought anywhere near hardware peak, ONNX beats TensorFlow
   outright and the whole question closes. This is the open thread.
3. **Do not re-export for an FFT** (§3).
4. **Do not tune CUDA provider options** (§4).
5. **Do not build a TensorFlow distribution path yet.** It buys 1.29x over the
   fused model, and item 2 may erase even that.
