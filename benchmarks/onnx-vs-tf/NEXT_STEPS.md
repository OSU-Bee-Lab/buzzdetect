# buzzdetect: next steps

Written 2026-08-27, out of the ONNX-vs-TensorFlow benchmark
(`benchmarks/onnx-vs-tf/` on the `bench/onnx-vs-tf` branch). Ordered by
value-for-effort, best first.

Measured on a GTX 1650, 200 s of audio per call, median of 30:

| path | time | vs TensorFlow |
|---|---|---|
| TensorFlow (`yamnet_k2`) | 52.6 ms | 1.00x |
| **ONNX fused + FusedConv** | **49.1 ms** | **0.93x — faster** |
| ONNX fused (`model_combined.onnx`) | 67.7 ms | 1.29x slower |
| **ONNX as shipped today** | **177.0 ms** | **3.37x slower** |

All agree numerically to 2.5e-05, top class for top class; the FusedConv row is
bit-exact against the row above it.

**Updated 2026-08-27, after §2 closed.** §2 asked whether two slow convolutions
could be rescued. They were never slow — the per-node profiler was
misattributing (`RESULTS.md` §8). The real saving was fusing Conv+Relu, which
puts ONNX ahead of TensorFlow and retires §4 and §5 below.

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

## 2. Fuse Conv+Relu at export  (CLOSED — do this, it is ~20 lines)

**This section previously asked how to rescue two convolutions running two
orders of magnitude off peak. There were no such convolutions.** The per-node
profiler was charging them other nodes' queue wait; timed properly they take
0.48 ms and 0.87 ms, and cost is flat across the trunk. `RESULTS.md` §8 has
the full accounting. (That in turn supersedes the §2 before it, which
recommended re-exporting with a real FFT — the dense-matmul DFT costs 0.83 ms.
This section has now been wrong twice; the current version is the first one
whose measurement carried a control.)

The actual saving is the **Relus**: 27 standalone nodes that re-read and
re-write a whole activation tensor to apply a `max()`. tf2onnx emits them
separately, and ORT's `ConvActivationFusion` is not registered for the CUDA EP,
so nothing folds them. Rewriting each Conv+Relu pair into a
`com.microsoft.FusedConv` node:

| arm | time | vs TF |
|---|---|---|
| `onnx_fused`, as exported | 67.8 ms | 1.29x slower |
| **`onnx_fused` + FusedConv** | **49.1 ms** | **0.93x — faster** |

Bit-exact. Standalone trunk 41.7 -> 23.2 ms, 1.80x.

**What to do:** put the rewrite in `engine/tools/onnxify_model.py`, beside the
export it corrects — the `.onnx` files are build artifacts, so this belongs at
export time, not at runtime.
`benchmarks/onnx-vs-tf/bench_fusedconv_endtoend.py::fuse_conv_relu` is the
entire implementation. Fold it into §1's re-export so the two ship together,
then re-run the §2 parity check in `RESULTS.md`.

**Caveats before shipping it:**

- `FusedConv` is a `com.microsoft` contrib op, not standard ONNX. It is part of
  onnxruntime proper (not a separate package), but it does pin the artifact to
  onnxruntime as the runtime. That is already true in practice.
- The CPU EP is fine: 343.1 ms as exported vs 339.2 ms fused, i.e. no change
  (it already applies this fusion itself). **CoreML is the one still
  unmeasured** — it may decline the contrib op and partition the graph around
  it, which would be a real regression on the platform with the most to lose.
  Measure on macOS before shipping.
- Fuse only where the Conv's output has exactly one consumer, as the reference
  implementation does; otherwise the rewrite drops a value someone else reads.

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

**Moot: §2 worked.** ONNX is now the faster path on CUDA as well as on macOS,
so there is no reason to prefer TensorFlow anywhere, and no platform-aware
logic to build. Don't build it.

## 5. Bring-your-own interpreter  (motivation retired; the §6 use stands)

**§2 closed, so the reason this section existed is gone** — nobody needs to
bring their own TensorFlow to get GPU speed, because ONNX is now faster. What
survives is the smaller, unrelated use: somewhere for a packaged app to fix the
cuDNN library path (§6). Judge it on that alone, which is a much weaker case.

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
