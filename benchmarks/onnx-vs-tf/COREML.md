# CoreML: the placement question, answered

Measured 2026-08-27 on Apple M1 (8 cores, 16 GB), macOS 26.5.2, onnxruntime
1.29.0, against a waveform-in fused graph (YAMNet front end + trunk +
`yamnet_medium_general` head, 18 classes) built by the training repo's
`tools/export_onnx.py`. 200 s of audio per call, median of 15-20 after warmup.

`HANDOFF.md` §4 asked one question -- does CoreML take `com.microsoft.FusedConv`,
or does it decline the op and shatter the graph? The answer turned out to
depend on something the handoff did not ask about, so read the shape rows
first.

## The short version

1. **Input shape decides everything.** With a dynamic waveform length the
   shipped CoreML configuration (`ModelFormat=MLProgram`) **cannot run the
   fused graph at all** -- CoreML's MIL rejects unbounded dimensions. Fix the
   input length and the same graph runs end to end on CoreML in two partitions.
2. **With a fixed input length, FusedConv is free on CoreML** -- 68.5 ms fused
   against 69.1 ms unfused, i.e. no difference, and no shift of nodes onto the
   CPU. Outcome 1 in the handoff's table: **bake the fusion in at export.**
3. The regression the handoff feared is real, but only under the CoreML
   *default* (`NeuralNetwork`) format, which buzzdetect deliberately does not
   use. There, CoreML declines FusedConv, all 27 land on the CPU, and the graph
   goes 143.7 -> 300.1 ms.

## Placement tables

Node counts are node-*runs* over two profiled runs, so halve them for nodes.
An entire CoreML partition profiles as one node, which is why the CoreML
column is small when things are going well.

### Dynamic input length (`waveform: [unk]`)

| config | arm | time | placement |
|---|---|---|---|
| CoreML default (NeuralNetwork, ALL) | as exported | 154.1 ms | CPU=148 CoreML=18 (9 partitions) |
| CoreML default (NeuralNetwork, ALL) | fused | 339.9 ms | CPU=202 CoreML=18 — **FusedConv=54 on CPU** |
| CoreML MLProgram, CPUAndGPU (shipped) | either | — | **fails to run**: `Concat` error; MIL reports `input_1 has unbounded dimension which is not supported` |
| CPU EP | as exported | 240.4 ms | CPU=240 |
| CPU EP | fused | 241.1 ms | CPU=240 |

Parity on the CPU EP is BIT-EXACT, as the handoff predicted. On the CoreML
default path the two arms differ by 1.58e-2 — that is not the rewrite, it is
the Neural Engine running the unfused arm in fp16 while the fused arm's convs
fall back to fp32 on the CPU.

### Fixed input length (`waveform: [3200000]`)

| config | arm | time | max abs diff vs CPU fp32 | placement |
|---|---|---|---|---|
| CoreML MLProgram, CPUAndGPU (**shipped**) | as exported | **68.5 ms** | 4.5e-06 | CoreML=4 CPU=2 (`Log` only) |
| CoreML MLProgram, CPUAndGPU (**shipped**) | **fused** | **69.1 ms** | 4.5e-06 | CoreML=4 CPU=2 (`Log` only) |
| CoreML MLProgram, ALL | as exported | 69.1 ms | 4.5e-06 | CoreML=4 CPU=2 |
| CoreML MLProgram, ALL | fused | 68.9 ms | 4.5e-06 | CoreML=4 CPU=2 |
| CoreML default (NeuralNetwork, ALL) | as exported | 143.7 ms | 1.6e-02 | CPU=8 CoreML=6 |
| CoreML default (NeuralNetwork, ALL) | fused | 300.1 ms | 2.7e-03 | CPU=62 CoreML=8 — **FusedConv=54 on CPU** |
| CPU EP | as exported | 230.8 ms | 0 | CPU=104 |
| CPU EP | fused | 238.3 ms | 0 | CPU=104 |

Two nodes on the CPU in the good rows is the whole graph minus `Log`, which
CoreML does not implement. That is one boundary crossing, not a shattering.

## What this means for the shipped artifact

- **Fuse at export.** No macOS regression under the configuration buzzdetect
  actually requests, a 1.38x win on CUDA (`RESULTS.md` §8). The decision tree
  in `HANDOFF.md` §4 does not need to be entered.
- **The engine must fix the input length before creating the session.** This is
  not an optimisation, it is the difference between CoreML running and CoreML
  failing. `onnxruntime.tools.onnx_model_utils.make_dim_param_fixed` costs
  ~40 ms on this 14 MB graph, and session creation at a new length is ~0.5 s
  including CoreML's compile — cheap enough to do once per worker.
- Chunks shorter than the session's length (the tail of every file) have to be
  zero-padded up to it and their surplus frames dropped.

## Against what ships today

Today's macOS path computes the log-mel front end in NumPy on the CPU and runs
only the trunk on CoreML:

| path | time per 200 s chunk |
|---|---|
| NumPy front end (`embedders/yamnet_onnx/features.py`) | 118.2 ms |
| YAMNet trunk, CoreML MLProgram CPUAndGPU, 209 patches | 50.6 ms |
| head | <1 ms |
| **total, as shipped** | **~170 ms** |
| **fused single graph, CoreML MLProgram CPUAndGPU** | **68.5 ms** |

**2.5x on macOS**, and the front end's 118 ms of CPU stops competing with the
decoder threads for cores (`RESULTS.md` §6).

## §5.1: the unverified 3.5x for fp16 / the Neural Engine

It is real, and it is bigger than the fp32 GPU path — but only for the
convolutional trunk, so end to end it is worth about half the quoted number.

YAMNet trunk alone, 209 patches, fp16 via `onnxconverter_common.float16`:

| config | time | max abs diff on embeddings |
|---|---|---|
| CPU fp32 | 180.9 ms | 0 |
| CoreML MLProgram CPUAndGPU fp32 (shipped) | 49.5 ms | 4.3e-06 |
| CoreML MLProgram ALL fp32 | 49.5 ms | 4.3e-06 |
| **CoreML MLProgram ALL fp16** | **15.1 ms** | 1.9e-02 |
| CoreML NeuralNetwork ALL, fp32 file (ANE runs it in fp16 anyway) | 13.9 ms | 1.4e-02 |
| CoreML NeuralNetwork ALL, fp16 file | 197.1 ms | 1.8e-02 |

**3.3x** for the trunk, against the fp32 MLProgram path. So the ~3.5x in
buzzdetect's notes was not folklore.

On the whole fused graph the win halves, because the log-mel front end stays in
fp32 and becomes the larger share of what is left:

| config | time | max abs diff on predictions | top-class agreement |
|---|---|---|---|
| CPU fp32 | 241.5 ms | 0 | 1.0000 |
| CoreML MLProgram CPUAndGPU fp32 (shipped) | 71.2 ms | 4.3e-06 | 1.0000 |
| **CoreML MLProgram ALL fp16** | **38.2 ms** | 1.7e-02 | 0.9952 |
| CoreML MLProgram CPUAndGPU fp16 | 68.1 ms | 1.2e-02 | 0.9952 |

**1.86x end to end**, at 1.7e-02 on the predictions — two orders of magnitude
past the 1e-4 parity budget the models are validated against, and one frame in
209 changes its top class. That is the same trade `BUZZDETECT_GPU_FP16` already
describes, so the switch keeps its meaning; it is not something to turn on by
default.

Note `MLComputeUnits=ALL` is what reaches the Neural Engine. `CPUAndGPU` with
an fp16 file gains nothing (68.1 vs 71.2 ms) — Metal was already running fp32
at full rate.

Two mechanical notes for anyone redoing this:

- Converting an already-fused graph to fp16 produces a model onnxruntime will
  not load: the converter does not know `com.microsoft.FusedConv`'s type
  schema. Convert first, then fuse — or block the front end and re-fuse after.
- The converter also trips over the explicit `Cast`-to-float in the framing
  code, so the front-end nodes have to go in `node_block_list` regardless.

## Reproducing

`verify_fusion.py` answers the fusion question but always asks for
CoreML bare, which is the NeuralNetwork path buzzdetect does not use.
`verify_coreml.py` beside it runs the same three measurements across the
CoreML configurations that matter, including the shipped one.

```
python verify_coreml.py path/to/model.onnx
```

---

## §5.6: is throughput decode-bound?

Not on this machine, at this speed, on local files. 48 copies of the 300 s
fixture — four hours of 44.1 kHz FLAC — through `buzzdetect_cli.py` with one
CoreML analyzer:

| streamers | wall | vs real time |
|---|---|---|
| 1 | 11.4 s | 1267x |
| **2** | **8.9 s** | **1614x** |
| 4 | 9.2 s | 1565x |
| 8 | 10.2 s | 1412x |

No `BUFFER BOTTLENECK` was reported at any setting, so the analyzer never
waited more than 10 ms for an assignment. Two streamers is 21% better than the
default of one; past that the decoder threads start taking cores off the
analyzer and it gets slower again, which is the contention `RESULTS.md` §6
describes, just much milder here.

Read this narrowly. Forty-eight copies of one file on a local SSD is the
friendliest possible decode: everything after the first read is in the page
cache, FLAC is cheap to decode, and there is no network. The Linux box's
finding — twelve decoder threads and an analyzer contending for eight cores —
is about a different shape of workload. What this does show is that the
inference side is no longer the thing to optimise on an Apple machine: four
hours of audio in nine seconds.

## Still unmeasured

**The fused graph has not been run on CUDA.** `RESULTS.md` §8 measured
FusedConv, and `fold_depthwise_bn.py` measured the batchnorm fold, but on the
old `model_combined.onnx` and separately. The graph that now ships has both
rewrites and a pinned input length, and no one has timed that combination on a
GPU. The rewrites are the same ones, so the expectation is the same 1.38x or
better — but it is an expectation, not a measurement.
