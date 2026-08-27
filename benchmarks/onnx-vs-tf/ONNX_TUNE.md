# ONNX conv tuning — closed

**Resolved 2026-08-27. The premise was wrong and the gap is closed.** Full
write-up in `RESULTS.md` §8; this file is kept for the record of what the
question was and how it was answered.

## What this document used to say

That two convolution layers consumed ~50 ms of every ~68 ms call while the
other 25 took ~4 ms, running one to two orders of magnitude off hardware peak,
and that closing this would decide whether the ONNX path could ship.

## What was actually true

**There were no pathological convolutions.** `profile_trunk.py`'s per-node
kernel times are an artifact: enabling profiling makes onnxruntime record CUDA
events around every kernel, serialising the stream and charging queue wait to
whichever node is waiting — always one of the first large ones.

The tell was present in the original numbers and went unread: **the per-node
times summed to 54.9 ms against a 41 ms wall clock.** The parts exceeded the
whole, so at least one of them was not measuring what it claimed.

Timed as standalone graphs on their real shapes, with inputs device-resident
and outputs synchronised (`isolate_convs.py`):

| layer | profiler said | actually |
|---|---|---|
| layer 2 depthwise | 29.0 ms | **0.87 ms** |
| layer 1 conv | 18.6 ms | **0.48 ms** |

`sum_of_parts.py` rebuilds every node of ORT's optimized graph the same way and
reconciles: 27 Conv = 32.3 ms, 27 Relu = 10.3 ms, sum 42.9 ms against the 41 ms
whole. Cost is flat across the trunk.

## What closed the gap

The Relus. 27 standalone passes that re-read and re-write a whole activation
tensor to apply a `max()` — layer 2's alone moves 82 MB. tf2onnx emits them as
separate nodes and ORT's `ConvActivationFusion` is not registered for the CUDA
EP, so nothing folds them. Rewriting each Conv+Relu to a
`com.microsoft.FusedConv` node:

| arm | time | vs TF |
|---|---|---|
| TensorFlow (`yamnet_k2`) | 52.6 ms | 1.00x |
| `onnx_fused`, as exported | 67.8 ms | 1.29x slower |
| **`onnx_fused` + FusedConv** | **49.1 ms** | **0.93x — faster** |

Bit-exact (max abs diff 0.0). Standalone trunk 41.7 -> 23.2 ms, 1.80x.

**The ONNX path now beats TensorFlow**, which retires the fallback of shipping
or sourcing TensorFlow for CUDA users (`NEXT_STEPS.md` §5).

## The five proposed next steps, in hindsight

1. ~~TensorFlow per-layer profile~~ — unnecessary. The premise it would have
   tested did not exist, and a TF-side profiler would have been just as
   vulnerable to the same artifact.
2. ~~A newer onnxruntime~~ — not needed; nothing was wrong with 1.26. The
   shipped pin can stay.
3. ~~TensorRT~~ — not needed. Would have been a large dependency bought against
   a problem that wasn't there.
4. **Isolate the two layers** — this was the right call, and listed fourth. It
   is what produced the answer, in about twenty minutes. It was ranked last
   among the cheap options because the profiler's numbers were trusted; the
   lesson is that the experiment which *checks the premise* should be ranked
   above the ones that build on it.
5. ~~Question the shape~~ — the shapes are fine.

## Ruled out along the way

| lever | result |
|---|---|
| Dynamic batch dim (`unk__360`) | static batch and free-dimension override both within noise (`try_static_batch.py`) |
| Asymmetric padding on layer 1 | symmetric and unpadded variants identical |
| Layer 1's single input channel | 3 channels slower in wall time |
| Folding depthwise BN by hand | 26 nodes removed, bit-exact, **1.00x** — ORT already does it (`fold_depthwise_bn.py`) |
| GPU clocks / thermals | boosting correctly: 1875 MHz, P0, 99% util, 61 W of 75 W |
| Prefix bisection of the trunk | invalid — ORT re-optimizes each prefix, so a cut inside a fusion group yields a different graph (`bisect_trunk.py`) |

Plus everything in the original table, which still holds: `cudnn_conv_algo_search`,
`cudnn_conv_use_max_workspace`, `prefer_nhwc`, batch size, fp16 (0.52x on this
card), the Memcpy warning, and CPU-resident nodes.

## Method note

Any per-node GPU timing needs a whole-graph control that reproduces a known
wall clock. Two plausible-looking attributions here were both wrong, in
different directions, and only the one carrying a control reconciled.

Measure with inputs device-resident (`IOBinding` + `synchronize_outputs()`).
Bound from the host, an 80 MB tensor's PCIe copy dwarfs the kernel: layer 3
appeared to take 19.7 ms when it takes well under a millisecond.

## Still live

The `LD_LIBRARY_PATH` trap is unchanged and still bites — see `RESULTS.md` §7.
This box has cuDNN 9.25 system-wide and 9.24 in the venv; onnxruntime mixes
their sub-libraries and **silently falls back to the CPU while still reporting
`CUDAExecutionProvider`**. Every script here must be run as:

```
cd engine
NVLIB=$(ls -d .venv-bench/lib/python3.12/site-packages/nvidia/*/lib \
        | xargs -I{} readlink -f {} | tr '\n' ':')
LD_LIBRARY_PATH="$NVLIB" .venv-bench/bin/python ../benchmarks/onnx-vs-tf/<script>.py
```

## What remains to do

Land the rewrite at export time — it belongs in `engine/tools/onnxify_model.py`
next to the export it corrects, not at runtime, since the `.onnx` files are
build artifacts. `bench_fusedconv_endtoend.py::fuse_conv_relu` is the whole
implementation, ~20 lines. Then re-run the parity check in `RESULTS.md` §2.

Note the scripts here deliberately change no engine code, per the original
ground rules; that rule is why this is a hand-off rather than a patch.
