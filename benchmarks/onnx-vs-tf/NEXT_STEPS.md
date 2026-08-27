# buzzdetect: next steps

Written 2026-08-27, out of the ONNX-vs-TensorFlow benchmark
(`benchmarks/onnx-vs-tf/` on the `bench/onnx-vs-tf` branch). Ordered by
value-for-effort, best first.

Measured on a GTX 1650, 200 s of audio per call, median of 30:

| path | time | vs TensorFlow |
|---|---|---|
| TensorFlow (`yamnet_k2`) | 52.6 ms | 1.00x |
| ONNX fused (`model_combined.onnx`) | 67.7 ms | 1.29x slower |
| **ONNX as shipped today** | **177.0 ms** | **3.37x slower** |

All three agree numerically to 2.5e-05, top class for top class.

---

## 1. Ship the fused ONNX model  (biggest win, no downside)

The shipped path spends **147.6 ms of its 177 ms computing the spectrogram in
NumPy on the CPU**. Its actual GPU work is 42.4 ms -- faster than TensorFlow's
whole 52.4 ms. The runtime was never the problem.

`models/model_general_v3_onnx/model_combined.onnx` already does the whole thing
in one graph and is 2.6x faster than what ships. No size cost: 14 MB, replacing
`model.onnx` (56 KB) plus the `yamnet_onnx` embedder (13 MB). Loads in 0.23 s
against TensorFlow's 2.35 s.

Two small blockers:

- `scripts/build-engine.mjs:208` requires a file named exactly `model.onnx`;
  this directory has `model_combined.onnx`.
- Its `model.py` declares `embeddername = 'yamnet_k2'`, and `BaseModel.__init__`
  imports that module even with `initialize=False`. `embedders/yamnet_k2/
  embedder.py` imports TensorFlow at module scope, which the frozen sidecar
  excludes -- so it would fail to load in a shipped build despite never running
  a TensorFlow op. It needs an embedder whose class attributes exist without
  importing TensorFlow.

## 2. Chase the two convolutions  (the open thread)

**Supersedes the previous §2, which recommended re-exporting with a real FFT.
Profiling disproved that: the dense-matmul DFT costs 0.83 ms, 1.1% of the run.
Do not spend effort there.**

The real cost is two convolution layers, and they are slow in **every** ONNX
graph we have -- the fused one and the standalone trunk alike, so it is not an
export artifact:

| layer | fused graph | standalone `yamnet.onnx` |
|---|---|---|
| layer2 depthwise conv | 29.87 ms | 29.03 ms |
| layer1 conv | 20.69 ms | 18.58 ms |
| the other 25 convs | ~4.3 ms | ~3.3 ms |

Convolutions are 73.6% of the fused graph's runtime and those two are 68% of
it. Layer 1 is about 0.18 GFLOP; on a ~2.9 TFLOP card that is well under a
millisecond of arithmetic, so it is running roughly **two orders of magnitude
off peak**. Depthwise convolutions are a known cuDNN weak spot, and TensorFlow
ships its own hand-tuned depthwise kernels -- most likely the entire reason it
wins.

**If these can be brought near hardware peak, ONNX beats TensorFlow outright
and every question below closes.** That is why this is the top item.

Already ruled out:

- **CUDA provider options.** `cudnn_conv_algo_search` (EXHAUSTIVE and
  HEURISTIC), `cudnn_conv_use_max_workspace`, and `prefer_nhwc`, alone and
  combined: all within noise of the 65.9 ms default, and NHWC is marginally
  worse. See RESULTS §4.

Not yet tried:

- The `13 Memcpy nodes are added to the graph` warning onnxruntime emits for
  this graph -- it explicitly says it may prevent CUDA graph capture.
- fp16. `BUZZDETECT_GPU_FP16` exists but only reaches CoreML today
  (`fp16_supported`); Turing does packed fp16 at 2x fp32.
- Batch size. Everything here runs 209 frames at once; a different batch may
  land on better kernels.
- Per-layer timing on the **TensorFlow** side, to confirm it really is fast on
  these same two layers rather than slow everywhere else.
- A newer onnxruntime (1.27+, CUDA 13 -- this box's 580 driver supports it),
  or the TensorRT execution provider.

## 3. Model shipping shape  (aspirational, from Luke)

Standardise every model directory on **`model.keras` + `model.onnx`**, where
the ONNX half is the *fused* graph (front end included). Today it's uneven:

| model | TensorFlow | ONNX |
|---|---|---|
| `model_general_v3` | `saved_model.pb` | `model.onnx` (trunk only) |
| `model_general_v3_onnx` | -- | `model_combined.onnx` (fused) |
| `yamnet_large_general` | `model.keras` | -- |

That makes `DualRuntimeModel` uniform, removes the separate `_onnx` model
directory, and means a newly trained model is shippable as soon as it's
exported. Requires the export tool to emit a fused graph by default (see §2).

## 4. Runtime preference

`_pick_runtime` defaults to `auto`, which prefers ONNX. Flipping it to prefer
TensorFlow **must not be a global flip**: on macOS TensorFlow gets no GPU at
all, while ONNX gets CoreML and its Neural Engine path is worth ~3.5x. So the
rule would have to be "prefer TensorFlow only where a CUDA GPU is present",
which is platform-aware logic that doesn't exist yet.

If §2 works, this becomes moot -- don't build it before then.

## 5. Bring-your-own interpreter  (only if §2 fails to close the gap)

`resolve_engine` (`src-tauri/src/lib.rs:45`) has two shapes: frozen sidecar, or
`engine/.venv` + `buzzdetect_cli.py`. The second only resolves in dev mode, so
today "run the GUI against your own TensorFlow" means installing Node, Rust and
the Tauri toolchain.

A third shape -- an explicit interpreter path from a setting, checked before
the sidecar -- would let a *packaged* app use a user's own venv, give the UI a
place to expose the runtime choice, and give the app somewhere to fix the cuDNN
path (§6). Roughly fifteen lines.

## 6. Guard against the silent CPU fallback

If a machine has cuDNN installed system-wide *and* in the Python environment,
onnxruntime can load mismatched halves, fail, and quietly fall back to the CPU
while still reporting `CUDAExecutionProvider`. Cost here: 214.7 ms instead of
42.4 ms, a 5x slowdown that looks completely normal in the logs.

`start_analysis` already sets `LD_LIBRARY_PATH` for the bundled-CUDA build, so
shipped builds are covered. Anything running against a user-supplied
environment (§5) needs the same treatment. A startup warning when the timings
look CPU-shaped would also be cheap insurance.
