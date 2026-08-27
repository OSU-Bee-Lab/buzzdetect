# ONNX conv tuning — handoff brief

Self-contained: everything a fresh session needs. Companion documents are
`RESULTS.md` (the full benchmark this came out of) and `NEXT_STEPS.md` (the
product decisions that depend on it). Branch `bench/onnx-vs-tf`.

## The issue in one paragraph

Running YAMNet through onnxruntime on CUDA, **two convolution layers consume
roughly 50 ms of every ~68 ms call**, while the other 25 convolutions together
take about 4 ms. Layer 1 is ~0.18 GFLOP, which on this card is well under a
millisecond of arithmetic; judged as memory-bound it moves ~82 MB in 29 ms,
about 2.8 GB/s against ~192 GB/s available. It is one to two orders of
magnitude off the hardware by either measure. Closing this would make the ONNX
path beat TensorFlow outright and settle a shipping decision.

## Why it matters

| path | time (200 s audio, 209 frames) | vs TensorFlow |
|---|---|---|
| TensorFlow (`yamnet_k2`) | 52.6 ms | 1.00x |
| ONNX fused (`model_combined.onnx`) | 67.7 ms | 1.29x slower |
| ONNX as shipped (`yamnet_onnx`) | 177.0 ms | 3.37x slower |

The shipped path's 177 ms is a separate, already-diagnosed problem (its log-mel
front end runs in NumPy on the CPU, 147.6 ms of the 177). Switching to the
fused graph fixes that and is tracked in `NEXT_STEPS.md` §1. **This document is
only about the remaining 1.29x**, which is entirely the convolutions.

If it can't be closed, the fallback is shipping or sourcing TensorFlow for
CUDA users — expensive and awkward (see `NEXT_STEPS.md` §5). So a negative
result here is still decision-relevant.

## Reproduce it

```
cd engine
NVLIB=$(ls -d .venv-bench/lib/python3.12/site-packages/nvidia/*/lib \
        | xargs -I{} readlink -f {} | tr '\n' ':')
LD_LIBRARY_PATH="$NVLIB" .venv-bench/bin/python \
    ../benchmarks/onnx-vs-tf/profile_trunk.py
```

**The `LD_LIBRARY_PATH` is mandatory.** This box has cuDNN 9.25 system-wide and
9.24 as a venv wheel; onnxruntime mixes their sub-libraries, fails with
`CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`, and **silently falls back to the
CPU while still reporting `CUDAExecutionProvider` from `get_providers()`**. A
CPU-bound run looks entirely normal and is ~5x slower. See `RESULTS.md` §7.

If `engine/.venv-bench` is gone, rebuild it per `PLAN.md` §1 (expect 4+ GB of
downloads). Only `onnxruntime-gpu==1.26.0` and `onnx` are needed for the
tuning work; TensorFlow is only there for the comparison arms.

Expected output — the pathology, on the standalone trunk:

```
yamnet_core_1/layer2_depthwise_conv_1/depthwise   Conv   29.03 ms  51.9%
Conv__169  (layer1)                               Conv   18.58 ms  33.2%
...the other 25 convs                                     ~3.3 ms
total 55.91 ms/run   (wall clock without profiling: 42.3 ms)
```

`profile_fused.py` shows the same two layers in `model_combined.onnx`, so this
is **not** an export artifact — it is present in every ONNX graph in the repo.

## Environment

| | |
|---|---|
| GPU | GTX 1650 (TU117, Turing, compute 7.5), 4 GB, driver 580.173.02 |
| CPU | 8 cores |
| onnxruntime-gpu | 1.26.0 (CUDA 12), the version that ships |
| cuDNN | 9.24 wheel in venv; 9.25 system-wide (see the trap above) |
| Python | 3.12.3, `engine/.venv-bench` |

Shapes: layer 1 is a 3x3 stride-2 conv, `[209,1,96,64] -> [209,32,48,32]`, one
input channel. Layer 2 is a 3x3 **depthwise** conv on `[209,32,48,32]`.

## Ruled out — do not repeat these

| lever | result |
|---|---|
| `cudnn_conv_algo_search` (EXHAUSTIVE / HEURISTIC) | within noise of 65.9 ms |
| `cudnn_conv_use_max_workspace=1` | within noise |
| `prefer_nhwc=1` (alone and combined) | 66.5–66.8 ms — marginally **worse** |
| Batch size (1 → 836 frames) | per-frame cost flat from 32 up; 209 is fine |
| fp16 (`onnxconverter_common.float16`) | **0.52x — half speed**, +0.0108 error |
| The `13 Memcpy nodes` warning | costs 0.000 ms; optimizer removes them |
| CPU-resident nodes | 59 of 147, all shape arithmetic, 0.618 ms total (0.9%) |

Scripts for each: `sweep_cuda_opts.py`, `sweep_batch.py`, `try_fp16.py`,
`find_cpu_nodes.py`.

**On fp16:** the GTX 16xx dies have a crippled fp16 rate. This result is
specific to this card and says nothing about CoreML's fp16/Neural Engine path
on macOS, which is reportedly ~3.5x in the other direction.

## Also worth knowing before theorising

**TensorFlow is probably not faster at these convolutions.** An earlier draft
assumed its hand-tuned depthwise kernels explained the gap. The numbers refute
it: onnxruntime's trunk *alone* is 42.4 ms while TensorFlow's *entire* embed —
front end plus trunk — is 52.4 ms. So these layers are plausibly slow under
both, and the 1.29x may live somewhere other than raw conv throughput. Worth
confirming with a TensorFlow-side per-layer profile (`tf.profiler`) before
assuming there is headroom to recover at all.

**The front end is not the problem.** Convolution-side work is 56.2 ms of the
fused graph; the whole STFT front end is 13.0 ms (18.8%). The DFT — computed
as a dense matmul rather than an FFT — costs 0.83 ms, 1.1%. An earlier
hypothesis that this was the story is dead; do not re-export chasing an FFT.

## Next steps, best first

1. **TensorFlow per-layer profile.** Cheapest way to learn whether there is any
   headroom. If TF is also ~30 ms on layer 2, the ceiling is the GPU and this
   whole thread closes with a clear negative — which is a useful answer.
2. **A newer onnxruntime.** 1.27+ is built against CUDA 13; this box's 580
   driver supports it. Conv kernel selection changes between releases. Cheap to
   test in a throwaway venv. Note the shipped pin is 1.26 for driver-coverage
   reasons (`requirements-onnx-cuda.txt`), so a win here is an argument to
   revisit that pin, not a free change.
3. **TensorRT execution provider.** The most likely real win — TensorRT does
   its own kernel selection and fuses aggressively. Costs: another large
   dependency, per-machine engine build time on first run, and it would apply
   only to NVIDIA. Worth prototyping before committing.
4. **Isolate the two layers.** Extract them into a minimal ONNX graph and time
   them against a hand-written CUDA/cuDNN call and against PyTorch's conv on
   the same shapes. That separates "onnxruntime chooses badly" from "this shape
   is simply bad on Turing" — the single most informative experiment, if the
   cheaper ones are inconclusive.
5. **Question the shape.** Layer 1 has one input channel, which is an awkward
   case for cuDNN. If YAMNet's front end emitted a different layout — or the
   patches were laid out `[N,96,64,1]` — the first conv might land on a better
   kernel. This changes the model, so it is last, and it would need re-export
   plus a parity check.

## Ground rules

- Change no engine code. These scripts call into `engine/` from outside.
- Verify every result is actually on the GPU (see the cuDNN trap).
- Full-precision parity for any change that touches numerics: use
  `compare_npy.py`, not the result CSVs, which are rounded to 2 decimals by
  `write/formatting.py:33` and cannot resolve the 2.5e-05 the arms agree to.
