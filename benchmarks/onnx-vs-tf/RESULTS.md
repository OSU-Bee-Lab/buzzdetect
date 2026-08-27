# ONNX vs TensorFlow on CUDA — results

Run 2026-08-27 on `beelab-files`. Branch `bench/onnx-vs-tf`. Scripts in this
directory; raw outputs in the gitignored `local/bench/`.

**Verdict: the ~30% suspicion does not reproduce as stated, and the cause is
not the one anyone expected.** ONNX as shipped is 3.4x slower than TensorFlow,
not 30%. Almost all of that is a CPU-side spectrogram that has nothing to do
with onnxruntime. What remained after fixing it was a 1.29x gap, and §8 closes
it: fusing Conv+Relu takes the fused graph to **49.1 ms against TensorFlow's
52.6 ms**, bit-exact. **The ONNX path wins outright.**

> **§3's per-layer attribution below is wrong, and §8 explains why.** The "two
> convolution layers ~100x off peak" are a profiling artifact. Timed properly
> they take 0.48 ms and 0.87 ms. The section is kept as written because the
> mistake is instructive and the rejected hypotheses in §4 remain valid.

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

> **Superseded by §8.** Everything in this section that attributes cost to a
> *particular node* is an artifact of the profiler. The totals are fine.

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
magnitude off peak. Even judged as memory-bound (depthwise convolutions have
poor arithmetic intensity), layer 2 moves ~82 MB in 29 ms — about 2.8 GB/s
against the card's ~192 GB/s. Slow by any measure.

**And that reasoning was right, which is exactly why the measurement should
have been doubted.** A number two orders of magnitude off hardware is more
often a broken measurement than a broken kernel. The tell was in this table
already: the per-node kernel times sum to 54.9 ms against a 41 ms wall clock.
The parts exceed the whole. See §8.

**But this is not why TensorFlow wins, and an earlier draft of this document
said it was.** That draft blamed TensorFlow's hand-tuned depthwise kernels. Our
own numbers refute it: onnxruntime's trunk alone is **42.4 ms**, while
TensorFlow's *entire* embed — front end and trunk together — is **52.4 ms**.
TensorFlow cannot be beating onnxruntime at convolutions when its whole
pipeline costs more than onnxruntime's convolutions do. The likelier reading is
that these layers are slow under *both* runtimes: a property of YAMNet's shape
on this GPU, not an onnxruntime defect.

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

## 4. Tuning attempts — four levers, none of them moved it

`providers_for()` (`src/inference/onnx.py:84`) passes `{}` today, so cuDNN
algorithm selection and tensor layout sit at their defaults.

**Provider options.** Depthwise convs are the classic case where those defaults
are wrong. They aren't:

| config | median | vs default |
|---|---|---|
| default (what ships) | 65.9 ms | 1.00x |
| `cudnn_conv_algo_search=EXHAUSTIVE` | 65.9 ms | 1.00x |
| `cudnn_conv_algo_search=HEURISTIC` | 66.0 ms | 1.00x |
| `cudnn_conv_use_max_workspace=1` | 65.9 ms | 1.00x |
| `prefer_nhwc=1` | 66.5 ms | 0.99x |
| nhwc + EXHAUSTIVE | 66.8 ms | 0.99x |
| nhwc + EXHAUSTIVE + workspace | 66.7 ms | 0.99x |

Everything within noise; NHWC marginally worse.

**Batch size.** Everything else here runs 209 frames at once (200 s at
framehop 1). If cuDNN were picking a bad kernel at that shape, `--chunklength`
would be a user-facing fix. Per-frame cost on the standalone trunk:

| frames | 1 | 8 | 32 | 64 | 128 | **209** | 256 | 418 | 836 |
|---|---|---|---|---|---|---|---|---|---|
| ms/frame | 0.976 | 0.328 | 0.213 | 0.207 | 0.203 | **0.206** | 0.205 | 0.232 | 0.203 |

Flat from 32 frames upward — the convolutions scale linearly, so they are
uniformly slow rather than badly shaped. `--chunklength 200` is already fine
and is not a tuning knob for this.

**fp16.** The last lever on convolution throughput, and it is actively harmful
here:

| | time | rate |
|---|---|---|
| fp32 (current) | 42.3 ms | 4741 audio-s/wall-s |
| fp16 | 81.0 ms | 2478 audio-s/wall-s |

**0.52x — half the speed** — plus 0.0108 max absolute error on the embeddings.
The GTX 16xx Turing dies have a crippled fp16 rate and this is one of them.
(Says nothing about CoreML's fp16/ANE path on macOS: different silicon,
reportedly ~3.5x the other way.)

**CPU fallback and host/device copies.** onnxruntime warns that it inserts
`13 Memcpy nodes` into this graph. Profiled, they cost **0.000 ms** — the
optimizer removes them. 59 of 147 nodes do sit on `CPUExecutionProvider`, but
all are tiny shape arithmetic (`Shape`, `Concat`, `Split`, `Cast`) totalling
**0.618 ms/run — 0.9%**. Not a factor.

**Where that leaves it.** Splitting the fused graph's profile by op class:
convolution-side work (Conv/Relu/Pool/Gemm) is 56.2 ms, and everything else —
the entire STFT front end — is 13.0 ms (18.8%). So the front end isn't the
problem either. The ONNX path looks close to its practical floor under
onnxruntime 1.26 on this GPU.

Untested, both larger lifts than a config change: a newer onnxruntime (1.27+,
CUDA 13 — this box's 580 driver supports it), and the TensorRT execution
provider.


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

## 8. The 1.29x, closed — it was the Relus, not the convolutions

`profile_trunk.py`'s per-node kernel times are not trustworthy. Enabling
profiling makes onnxruntime record CUDA events around every kernel, which
serialises the stream and bills queue wait to whichever node is doing the
waiting — always one of the first large ones. That is how layer 1 and layer 2
came to be charged 18.6 and 29.0 ms, and why the column sums to 54.9 ms against
a 41 ms wall clock.

Timed honestly — rebuilt as standalone graphs on their real shapes, inputs
device-resident via `IOBinding`, outputs synchronised explicitly
(`isolate_convs.py`):

| layer | profiler said | actually |
|---|---|---|
| layer 2 depthwise, `[209,32,48,32]` | 29.0 ms | **0.87 ms** |
| layer 1 conv, `[209,1,96,64]` | 18.6 ms | **0.48 ms** |

*(Device-resident inputs are not optional here. Bound from the host, an 80 MB
tensor's PCIe copy dwarfs the kernel: layer 3, which costs well under a
millisecond, "took" 19.7 ms. The trunk control in that script exists to catch
exactly this — it must reproduce the known ~41 ms or nothing else is credible.)*

`sum_of_parts.py` then rebuilds **every** node of ORT's own optimized graph and
times it the same way:

| op | nodes | standalone |
|---|---|---|
| Conv | 27 | 32.34 ms |
| Relu | 27 | **10.34 ms** |
| everything else | 3 | 0.25 ms |
| **sum of parts** | | **42.93 ms** |
| **whole trunk** | | **41.00 ms** |

The parts add up to the whole. **There is no pathology**: cost is flat across
the trunk and the convolutions are near what this card can do. The largest
single node is `Conv__289` at 4.45 ms; layer 1 is not in the top ten.

What the flat profile does expose is the Relus — 27 standalone passes that
re-read and re-write an entire activation tensor to apply a `max()`. Layer 2's
alone moves 82 MB. tf2onnx emits them as separate nodes, and ORT's
`ConvActivationFusion` is not registered for the CUDA EP, so nothing folds
them. Handing the graph `com.microsoft.FusedConv` nodes directly
(`try_fusedconv.py`, `bench_fusedconv_endtoend.py`):

| arm | time | vs TF |
|---|---|---|
| TensorFlow (`yamnet_k2`) | 52.6 ms | 1.00x |
| `onnx_fused`, as exported | 67.8 ms | 1.29x slower |
| **`onnx_fused` + FusedConv** | **49.1 ms** | **0.93x — faster** |

Standalone trunk: 41.7 ms -> 23.2 ms, **1.80x**. Both are **bit-exact** (max
abs diff 0.0), since the rewrite only changes which kernel applies the `max()`.
The saving exceeds the Relus' own 10.3 ms because fusion also eliminates the
intermediate write.

### Also ruled out in the course of this

| lever | result |
|---|---|
| Dynamic batch dim (`unk__360`) | static batch, and free-dimension override: both within noise (`try_static_batch.py`) |
| Asymmetric padding on layer 1 | symmetric and unpadded variants identical (`isolate_convs.py`) |
| Layer 1's single input channel | 3 channels is *slower* in wall time, 2.4x better per FLOP |
| Folding depthwise BN by hand | 26 nodes removed, bit-exact, **1.00x** — ORT's optimizer already does it (`fold_depthwise_bn.py`) |
| GPU clocks / thermals | boosting correctly: 1875 MHz, P0, 99% util, 61 W of 75 W |

That last one also explains why `bisect_trunk.py`'s prefix marginals disagreed
with both the profiler and `sum_of_parts.py`: ORT re-optimizes each prefix, so
a cut placed between a Conv and its Mul/Add yields a *different* graph, not a
smaller one. Prefix bisection is only valid across fusion boundaries.

## What to do about it

1. **Ship the fused ONNX model.** 177 ms -> 67.7 ms for every user, no size
   cost (14 MB replaces 56 KB + a 13 MB embedder), no new dependency, and it
   loads 10x faster than TensorFlow. Two small blockers in NEXT_STEPS §1.
2. **Apply the FusedConv rewrite at export** (§8). 67.8 ms -> 49.1 ms,
   bit-exact, no new dependency, no runtime cost — it is a graph rewrite over a
   build artifact, so it belongs in `tools/onnxify_model.py` beside the export
   it corrects.
3. **Do not ship or source TensorFlow for CUDA users.** With §8 the ONNX path
   is *faster* than TensorFlow (0.93x). NEXT_STEPS §5 is retired.
4. **Do not chase the convolutions further** (§8). They are not slow; the
   premise was a measurement artifact. A newer onnxruntime and TensorRT were
   the next candidates and neither is now worth its cost.
5. **Do not re-export for an FFT** (§3) — the DFT costs 0.83 ms.
6. **Do not switch to fp16 on NVIDIA** (§4) — it halves throughput on this
   class of card.
7. **Finish the end-to-end runs before sizing any of this** (§6). If real
   throughput is decode-bound on 8 cores, a 1.38x on inference may not reach
   users at all.

## Method note

Three separate attributions of the trunk's 41 ms disagreed with each other, and
the first two were wrong:

- **Per-node profiler** — inflated, and inflated *non-uniformly*. Its own total
  exceeding the wall clock was the available tell.
- **Prefix bisection** — invalidated by re-optimization at each cut.
- **Standalone-node sum, validated against a whole-graph control** — the only
  one that reconciled, and only because the control was there to check it.

Any per-node GPU timing here should carry a whole-graph control that reproduces
a known number. Both wrong methods looked entirely plausible in isolation.
