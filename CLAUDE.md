# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Tauri 2 + SvelteKit desktop GUI wrapped around `engine/`, a **git subtree** of the [buzzdetect](https://github.com/OSU-Bee-Lab/buzzdetect) bioacoustics pipeline. The GUI lets a user pick a model, an input audio folder, and an output folder, then runs the Python engine as a subprocess and renders its live progress as a file tree with progress bars.

Two codebases live side by side and only talk to each other through a subprocess boundary (Rust spawns Python, Python prints structured JSON lines on stdout):

- **Frontend**: `src/` (SvelteKit, Svelte 5 runes) + `src-tauri/` (Rust/Tauri commands)
- **Engine**: `engine/` — a standalone Python package with its own venv, CLI, and models, tracked as a git subtree of upstream buzzdetect. It can be run and tested completely independently of the Tauri app via `buzzdetect_cli.py`.

## Workflow

Commit as you go — after each self-contained change, not batched at the end of a session.

The user verifies all UI elements themselves (this sandbox has no headless browser/Playwright tooling and Tauri's webview needs a display that isn't available here). Don't claim to have visually verified UI changes — confirm via `npm run check` / `npm run build` and describe what to check manually instead.

## Commands

Frontend (run from repo root):
```
npm run dev            # vite dev server (also what `tauri dev` drives via beforeDevCommand)
npm run build           # vite build -> ../build (frontendDist)
npm run check            # svelte-kit sync + svelte-check
npx tauri dev            # run the full desktop app in dev mode
npx tauri build           # produce a bundled desktop app
npm run build:engine       # freeze the engine into a sidecar (see Releases)
```

Engine (run from `engine/`, using its own venv):
```
uv venv --python 3.13 .venv && uv pip install -r requirements.txt   # one-time setup
.venv/bin/python3 buzzdetect_cli.py --modelname <name> --dir_audio <dir> --dir_out <dir>
```
There is no test suite in this repo currently. The closest thing to one lives
in the other repo: buzzdetect-training's `tools/export_onnx.py` re-exports a
model's graph and refuses to write it if it stops matching the Keras model it
came from.

## Releases

`npm version <x.y.z>` (which syncs `tauri.conf.json`/`Cargo.toml`/`Cargo.lock`
via `scripts/sync-version.mjs`), then push the tag. `.github/workflows/release.yml`
builds installers for macOS arm64, macOS x86_64, Windows and Linux and attaches
them to a **draft** release, which you publish by hand.

There is one bundled-CUDA variant, Windows only, and it ships as a portable zip
rather than an installer: its ~2.7GB of NVIDIA runtime passes both makensis's
~2GiB ceiling and GitHub's 2GiB release-asset limit. That size is a fixed cost
-- nvprune only accepts static libraries, not the prebuilt `.so`/`.dll` the
wheels ship, and cuDNN's sub-libraries aren't safely separable -- so the
packaging bends around it. There is no Linux equivalent for the same reason;
Linux GPU users install CUDA themselves and the ordinary build finds it.

No universal macOS build is possible: onnxruntime ships no universal2 wheel, so
PyInstaller can only freeze the engine for the architecture it's running on.
That's why the matrix has two separate macOS jobs.

## Gotchas / non-obvious behavior

- **Subprocess boundary, not a library call.** `src-tauri/src/lib.rs::start_analysis` spawns the engine as a child process. `resolve_engine` picks between two shapes: the frozen PyInstaller sidecar next to the app executable (shipped builds), or `engine/.venv` + `buzzdetect_cli.py` (a checkout). Either way the child's working directory must contain `models/` and `src/stream/drivers/`, because `engine/src/config.py` addresses both by relative path — that's the whole reason `scripts/build-engine.mjs` assembles `src-tauri/engine-payload/` rather than freezing them into the binary. Only one analysis runs at a time (`AnalysisState` mutex). Stopping is a request, not a kill: `cancel_analysis` writes `STOP` on the child's stdin (which is why that pipe is kept open after the manifest prompt's `y`), the engine takes that as the coordinator's early-exit path, and the logs keep flowing while the workers wind down and report themselves out. Signals are only the escalation, and only once the engine has gone quiet — `engine/src/pipeline/interrupt.py` is the engine end of that, where a signal, or a second stop request, is the impatient path. The engine's own stdout/stderr lines prefixed `BDPROGRESS ` (`PROGRESS_MARKER`) are JSON and get re-emitted as Tauri's `engine-progress` event; everything else becomes `engine-log`. `emit_progress()` in `engine/src/pipeline/progress_json.py` is the single source of truth for that wire format — if you add a new progress event kind, update both it and the frontend's `run` store (`src/lib/progress.svelte.ts`) together.
- **Manifest locking is duplicated logic, not shared code.** An output folder records the settings (model, output classes) that fixed its result schema in `buzzdetect_manifest.json`, and both sides independently refuse to let a run drift from it: Python via `reconcile_with_manifest`/`pipeline/manifest.py` (prompts y/N on stdin — Rust's `start_analysis` always auto-answers `y` since there's no attached terminal), Svelte via `checkManifest()` in `+page.svelte` (locks classes in the UI, blocks on model mismatch). Changing the manifest schema means updating both.
- **`engine/` is a git subtree, not a copy.** It tracks `upstream` = github.com/OSU-Bee-Lab/buzzdetect. Pull upstream work with `git subtree pull --prefix=engine upstream main --squash`; push work the other way with `git subtree push --prefix=engine upstream <branch>`. Keep the fork delta small — anything that isn't GUI-specific (new stream drivers, streaming fixes, format support) belongs upstream, where it comes back for free on the next pull. The current delta is `src/pipeline/progress_json.py` and its `emit_progress()` call sites, `src/pipeline/interrupt.py` and its wiring in `src/analyze.py`, `search_dir` in `src/utils.py` being a generator, driver precedence in `src/stream/audio.py`, `requirements.txt`, and the ONNX-only inference path (`src/inference/`, `buzzdetect.spec`, the requirements files, and the removal of `embedders/`). `engine/` carries its own `.gitignore` files — don't re-add engine rules to the root one.
- **A model is one ONNX graph, waveform in, predictions out.** There is no embedder plugin, no NumPy front end and no TensorFlow anywhere in the engine: `models/<name>/model.onnx` contains the log-mel front end, the embedding trunk and the classifier head, fused into a single graph by buzzdetect-training's `tools/export_onnx.py`. A model directory is that file, an optional `model.fp16.onnx`, a `config_model.json` naming the classes, and a `model.py` that is nothing but class attributes. The `.onnx` files are build artifacts: retrain a model or change the front end and you re-export, you don't derive anything at runtime. `shipped-models.txt` is the list of names that get bundled; everything else in `engine/models/` stays local.
- **The session is built at one fixed input length, and that is load-bearing.** `make_session` pins the graph's free `samples` dimension via onnxruntime's `add_free_dimension_override_by_name`, and `OnnxModel.predict` zero-pads every chunk up to it and drops the frames the padding produced. This is not a tuning choice: CoreML's MLProgram format cannot compile a graph with an unbounded dimension, so a dynamic session fails outright on macOS — and a fixed one lets CoreML constant-fold the front end's shape arithmetic and swallow the whole graph. `OnnxModel.n_frames` is what decides how much of the output is real; it reproduces the front end's framing arithmetic including a float32 reciprocal that makes it wrong by one frame if you do it in float64. `benchmarks/onnx-vs-tf/COREML.md` has the measurements.
- **The shipped build is the same engine, minus the NVIDIA runtime.** The CUDA variant is the same engine with `onnxruntime-gpu` and the NVIDIA runtime wheels; those libraries are deliberately kept *out* of the frozen binary (`strip_nvidia` in `engine/buzzdetect.spec`) and shipped as loose files in `engine-payload/nvidia/`, because a 2.5GB single file is more than makensis will bundle. `start_analysis` puts that directory on the child's `LD_LIBRARY_PATH`/`PATH` — if you move it, move both ends.
- **GPU availability is two questions, answered in two places.** Whether the *build* has a GPU execution provider is decided at build time and written to `engine-payload/gpu-providers.json`; whether *this machine* can actually run one can only be found by trying, because `ort.get_available_providers()` reports what onnxruntime was compiled with and says the same thing on a CUDA workstation and a laptop with no NVIDIA hardware. So the default builds carry `onnxruntime-gpu` *without* its `[cuda,cudnn]` extras -- the CUDA provider, none of the 2.7GB runtime -- and `probe_gpu` (`engine/src/inference/onnx.py`, reached via `buzzdetect_cli.py --probe_gpu`) builds a real session **and runs one inference through it** to see which provider survives. Both halves are load-bearing: a provider can initialise, take the graph, name itself in `get_providers()` and still fail on its first kernel. The case that forced this is a Linux box with two cuDNN 9.x installs on the loader path (a pip `nvidia-cudnn-cu12` wheel that an `/etc/ld.so.conf.d` entry put ahead of the system one) -- cuDNN 9 resolves its sub-libraries by soname, so it paired a core from one install with a sub-library from the other and died with `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH` on the first `FusedConv`, minutes into the analysis. `gpu_status` in `src-tauri/src/lib.rs` runs that once at startup and the frontend shows a spinner until it answers. Two builds skip the probe because they have nothing to discover: the bundled-CUDA one (`engine-payload/nvidia` exists) and any CoreML-only one (CoreML ships with macOS). If you add a GPU provider, decide which of those two groups it's in.
- `engine/src/gui/` is upstream's legacy CustomTkinter GUI, superseded here by the Tauri frontend; don't extend it. Same for `engine/docs/`, `engine/audio_in/`, and `engine/buzzdetect_gui.py` — they arrive with the subtree and are excluded at bundle time rather than deleted, since deleting them would conflict on every pull.
