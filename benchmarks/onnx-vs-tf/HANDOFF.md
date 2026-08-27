# Fuse Conv+Relu at ONNX export — handoff

**For:** an agent working in the buzzdetect *training* repo (TensorFlow/Keras),
on the macOS machine, with access to this repo too.

**Ask:** add one graph rewrite to the TF→ONNX export script, then answer the
questions that can only be answered on macOS. The rewrite is ~20 lines and is
written for you (`fuse_conv_relu.py`); the measuring is the real work.

*A copy of this was also handed over on a USB stick. If the two disagree, this
one is current.*

---

## 1. What this is and why

On CUDA, the buzzdetect ONNX path was 1.29x slower than TensorFlow (67.7 ms vs
52.6 ms per 200 s chunk). The cause was believed to be two pathological
convolution layers. **It wasn't** — that was a profiler artifact, and the
convolutions are fine. The real cost was the **Relus**.

tf2onnx emits every Relu as its own graph node. onnxruntime has a pass
(`ConvActivationFusion`) that folds a Relu into the preceding convolution, but
it is registered for the **CPU** execution provider and **not for CUDA**. So on
a GPU each Relu becomes a separate kernel that reads an entire activation
tensor out of memory and writes it back just to apply a `max()`. YAMNet at 209
frames has 27 of these, moving up to 82 MB apiece.

Rewriting each `Conv`→`Relu` pair into onnxruntime's `com.microsoft.FusedConv`
node lets the convolution kernel apply the activation to data it is already
holding, and the intermediate tensor is never written:

| arm | time | vs TF |
|---|---|---|
| TensorFlow (`yamnet_k2`) | 52.6 ms | 1.00x |
| ONNX fused graph, as exported | 67.8 ms | 1.29x slower |
| **ONNX fused graph + FusedConv** | **49.1 ms** | **0.93x — faster** |

Measured on a GTX 1650 against `model_general_v3_onnx/model_combined.onnx`, the
full waveform→predictions graph, on a real 200 s mp3. **Bit-exact** — max abs
diff 0.0 on the `(209, 13)` predictions, not merely within tolerance, because
the rewrite changes which kernel applies the `max()` and not the arithmetic.

`RESULTS.md` §8 in this directory has the evidence rather than the conclusion.

---

## 2. What to implement

You have the script that exports the TF/Keras models to ONNX and lands them in
buzzdetect. Add the rewrite as the **last step before writing the `.onnx`**,
after any export and after any `onnxsim`/checker pass you already run.

```python
from fuse_conv_relu import fuse_conv_relu   # vendored from this directory

model, n_fused = fuse_conv_relu(model)      # an onnx.ModelProto
onnx.checker.check_model(model)
onnx.save(model, out_path)
print(f'fused {n_fused} Conv+Relu pairs')
```

Or as a post-processing CLI step, if that fits the pipeline better:

```
python fuse_conv_relu.py model.onnx model.onnx
```

`fuse_conv_relu.py` depends on **`onnx` only** — no TensorFlow, no
onnxruntime, nothing from the buzzdetect engine. Vendor it, or lift the
function; it is one self-contained pass.

**Three things not to get wrong:**

- **It must run at export, not at load.** The `.onnx` files are build
  artifacts. (If §4 makes CoreML a problem, this changes — see the decision
  tree there.)
- **Only fuse when the Conv's output has exactly one consumer.** The
  reference implementation checks this. YAMNet has no branching convolutions,
  but MobileNet-v2/v3 and ResNet backbones do, and fusing across a skip
  connection deletes a tensor another node still reads. If the training repo
  ever exports a different backbone, this check is what keeps it correct.
- **`n_fused == 0` is a signal, not a no-op.** It means the pattern didn't
  match — a different activation (Relu6/HardSwish/Clip), or the graph already
  fused. Fail loudly rather than silently shipping an unfused model.

---

## 3. Verifying it (do this on every export)

`verify_fusion.py` in this directory is standalone (`onnx`, `onnxruntime`,
`numpy`). It reports parity, **EP placement**, and time for each provider:

```
pip install onnx onnxruntime numpy
python verify_fusion.py path/to/model.onnx
python verify_fusion.py path/to/unfused.onnx path/to/fused.onnx
```

Expected on the machine this was developed on (Linux, CUDA):

```
--- CUDAExecutionProvider ---
  parity        BIT-EXACT
  placement as exported   294 node-runs   CUDA=176  CPU=118
  placement fused         240 node-runs   CUDA=122  CPU=118
  as exported       67.5 ms
  fused             48.9 ms   1.38x

--- CPUExecutionProvider ---
  parity        BIT-EXACT
  placement as exported   242 node-runs   CPU=242
  placement fused         242 node-runs   CPU=242
  as exported      338.9 ms
  fused            336.2 ms   1.01x
```

**Parity must read BIT-EXACT.** If it doesn't, the rewrite is wrong — do not
loosen a tolerance to make it pass.

The CPU EP showing ~1.00x is correct and expected: it already applies this
fusion internally, so there is nothing left to win there. The point of the CPU
row is that it does not *regress*.

---

## 4. THE OPEN QUESTION — CoreML. This is why it's your machine.

`FusedConv` is a `com.microsoft` **contrib op**, not standard ONNX. It ships
inside onnxruntime itself (no extra package), but only some execution providers
implement it. CUDA and CPU do. **Nobody has ever run it on CoreML.**

The risk is not that it's slow — it's **graph partitioning**. If CoreML
declines the op, onnxruntime splits the graph and hands those nodes to the CPU.
A model that was one CoreML partition can become dozens of fragments with a
tensor copy at every boundary. That is *slower than never fusing at all*, and
macOS is the platform with the most to lose (CoreML's Neural Engine path is
reportedly worth ~3.5x, though see §5.1 — that number is itself unverified).

**A speedup number will not show you this.** That is why `verify_fusion.py`
prints the placement table. Run:

```
python verify_fusion.py path/to/model.onnx --ep CoreMLExecutionProvider
```

and read the `placement` rows. Three outcomes, three different answers:

| what you see | meaning | what to do |
|---|---|---|
| `fused` keeps roughly the same CoreML node count, time flat or better | CoreML takes FusedConv, or declines it harmlessly | **Ship it at export.** Done. |
| `fused` shifts a large number of nodes onto CPU, time regresses | CoreML declines the op and the graph shatters | **Do not bake it in.** Go to the decision tree below. |
| CoreML takes almost nothing in *either* graph | CoreML isn't really running here | Fix that first — this test says nothing until it is. Note `get_available_providers()` reports what onnxruntime was *compiled* with, so it will happily claim CoreML on a machine where nothing runs on it. The placement table is the ground truth. |

### If CoreML regresses — decision tree

Do not bake FusedConv into the shipped `.onnx`. Options, best first:

1. **Rewrite at session creation, only for the CUDA EP.** The pass is
   milliseconds on a ~150-node graph, so doing it at load costs nothing real.
   This lives in buzzdetect (`engine/src/inference/onnx.py`, near
   `make_session`/`providers_for`), not in the training repo — so the training
   repo change becomes *nothing*, and this handoff reduces to the measurement.
   Cleanest outcome if CoreML says no.
2. **Emit two artifacts** — `model.onnx` and `model.cuda.onnx`. Simple, but
   doubles the shipped size and adds a selection rule to the engine and the
   packaging script. Only if 1 proves awkward.
3. **Fuse a standard op instead.** `Conv`+`Clip` is standard ONNX where
   `Relu6` applies; there is no standard fused Conv+Relu. Probably a dead end,
   but worth ten minutes before accepting 2.

Whichever way it goes, **write down the placement tables** — this question has
never been answered and the answer should not have to be rediscovered.

---

## 5. Other hanging ends, while you're on that machine

These are open items from the same investigation. Some are squarely yours; some
just happen to need macOS or a fresh look. Ordered by value.

### 5.1 CoreML fp16 / Neural Engine — an unverified 3.5x

`RESULTS.md` §4 measured fp16 on the GTX 1650 at **0.52x — half speed** (the
GTX 16xx dies have a crippled fp16 rate). That result is card-specific and says
nothing about macOS. The often-repeated claim that CoreML's fp16/ANE path is
worth **~3.5x** appears in the notes with **no measurement behind it**.

`BUZZDETECT_GPU_FP16` exists in the engine but only reaches CoreML
(`fp16_supported`). Convert with `onnxconverter_common.float16` and measure —
speed *and* max abs error on the predictions. If 3.5x is real it is a bigger
win than everything else in this document combined; if it's folklore, several
downstream decisions rest on it and should stop.

### 5.2 Does the fused model actually work through the engine?

Everything so far calls the ONNX graph **directly**. Nobody has run a fused
model through `buzzdetect_cli.py` end to end and compared the output CSVs.
Do that once — it is the only test that exercises the real path.

Compare at **full precision from `.npy` dumps, not the CSVs**:
`write/formatting.py:33` rounds results to `digits_results` (2 for these
models) and cannot resolve anything below 0.01. `compare_npy.py` in this
directory does it properly.

### 5.3 Two blockers on shipping the fused graph at all

Independent of this work, and they will bite whoever ships next
(`NEXT_STEPS.md` §1):

- `scripts/build-engine.mjs:208` requires a file named exactly **`model.onnx`**.
  The fused model directory has `model_combined.onnx`. If the training repo
  controls the output filename, **just name it `model.onnx`** and this blocker
  disappears — worth doing while you're in the export script.
- The fused model's `model.py` declares `embeddername = 'yamnet_k2'`, and
  `BaseModel.__init__` imports that module even with `initialize=False`.
  `embedders/yamnet_k2/embedder.py` imports TensorFlow at module scope, which
  the frozen sidecar excludes — so it would fail to load in a shipped build
  despite never running a TensorFlow op. Needs an embedder whose class
  attributes exist without importing TensorFlow.

### 5.4 Model shipping shape (`NEXT_STEPS.md` §3, from Luke)

Standardise every model directory on **`model.keras` + `model.onnx`**, where
the ONNX half is the *fused* graph (front end included, waveform in). Today
it's uneven — `model_general_v3` has `saved_model.pb` + a trunk-only
`model.onnx`, `model_general_v3_onnx` has `model_combined.onnx`,
`yamnet_large_general` has no ONNX at all. This is squarely the training repo's
call and would make a newly trained model shippable the moment it's exported.

Note the provenance of `model_combined.onnx` is **not reproducible from this
repo** — there's no script here that builds it. If the training repo's exporter
is what produced it, good; if not, that gap needs closing before anyone relies
on regenerating it.

### 5.5 The silent-fallback trap (read before trusting any number)

On the Linux box, a cuDNN version mismatch made onnxruntime **fall back to the
CPU while still reporting `CUDAExecutionProvider` from `get_providers()`**. A
5x slowdown that looked completely normal, and it nearly produced a false
positive (`RESULTS.md` §7).

**CoreML has the same shape of failure** and it is the one you're testing. Never
trust `get_providers()` or `get_available_providers()` as evidence that a
provider ran — both report what onnxruntime was *compiled* with. Use the
placement table.

### 5.6 Throughput may be decode-bound anyway (`RESULTS.md` §6)

A partial end-to-end run showed the analyzer idle only 12% of the time yet
running at 67% of its solo speed — CPU contention between 12 decoder threads
and the analyzer on 8 cores. **Nobody knows whether real-world throughput is
capped by audio decoding rather than inference.** If it is, a 1.38x on
inference may deliver very little to actual users. Worth knowing before anyone
sizes this work. Not a macOS question, but it's the one that decides whether
any of this matters.

---

## 6. Files

Everything here is in `benchmarks/onnx-vs-tf/`.

**Standalone — copy these into the training repo.** They need only
`pip install onnx onnxruntime numpy`; no TensorFlow, no engine imports:

| file | what it is |
|---|---|
| `fuse_conv_relu.py` | the rewrite. Import it or run as CLI. **This is the deliverable.** |
| `verify_fusion.py` | parity + EP placement + timing, for any `.onnx` |

**Evidence and context — read, don't run.** These import from `engine/` and
expect `engine/` as the working directory with a CUDA venv:

| file | what it establishes |
|---|---|
| `RESULTS.md` §8 | the whole story: how the two "pathological" convolutions turned out not to exist, and how the Relus were found |
| `isolate_convs.py` | the two convolutions take 0.48 and 0.87 ms, not 18 and 28.5. Shows the IOBinding + `synchronize_outputs()` technique, with a whole-graph control that must reproduce a known wall clock |
| `sum_of_parts.py` | rebuilds every node of onnxruntime's own optimized graph; parts (42.9 ms) reconcile with the whole (41.0 ms), exposing the 27 Relus at 10.3 ms |
| `try_fusedconv.py` | the fix on the standalone trunk: 41.7 -> 23.2 ms |
| `bench_fusedconv_endtoend.py` | the fix on the full waveform->predictions graph with real audio: 67.8 -> 49.1 ms. `fuse_conv_relu.py` is this script's function, made standalone |
| `compare_npy.py` | full-precision parity. Use this, not the result CSVs, which round to 2 decimals |

**Negative results — don't repeat these:**

| file | result |
|---|---|
| `fold_depthwise_bn.py` | hand-folding the depthwise batchnorm: 26 nodes removed, bit-exact, **1.00x**. onnxruntime already does it |
| `try_static_batch.py` | the dynamic batch dimension is not the problem |
| `bisect_trunk.py` | **a method that does not work.** Prefix bisection is invalid here: onnxruntime re-optimizes each prefix, so a cut inside a fusion group yields a different graph. Kept as a warning |

---

## 7. Definition of done

1. The export script fuses, and says how many pairs it fused.
2. `verify_fusion.py` reports **BIT-EXACT** on every EP available to you.
3. **The CoreML placement question in §4 is answered and written down** — this
   is the part only you can do.
4. If CoreML regresses, the §4 decision tree is followed rather than shipping
   a macOS regression to win on CUDA.
5. One real `buzzdetect_cli.py` run against a fused model, compared at full
   precision (§5.2).
