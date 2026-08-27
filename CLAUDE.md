# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Tauri 2 + SvelteKit desktop GUI wrapped around `engine/`, the [buzzdetect](https://github.com/OSU-Bee-Lab/buzzdetect) bioacoustics pipeline. Both live in that one repo — there is no longer a separate engine repo or a fork relationship. The GUI lets a user pick a model, an input audio folder, and an output folder, then runs the Python engine as a subprocess and renders its live progress as a file tree with progress bars.

Two codebases live side by side and only talk to each other through a subprocess boundary (Rust spawns Python, Python prints structured JSON lines on stdout):

- **Frontend**: `src/` (SvelteKit, Svelte 5 runes) + `src-tauri/` (Rust/Tauri commands)
- **Engine**: `engine/` — a standalone Python package with its own venv, CLI, and models, tracked as ordinary files in this repo. It can be run and tested completely independently of the Tauri app via `buzzdetect_cli.py`.

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
There is no test suite in this repo currently. The closest things to one are
`engine/embedders/yamnet_onnx/BUILD.py` and `engine/tools/onnxify_model.py`,
which re-export their ONNX artifacts and fail if they stop matching the
TensorFlow originals.

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

- **Subprocess boundary, not a library call.** `src-tauri/src/lib.rs::start_analysis` spawns the engine as a child process. `resolve_engine` picks between two shapes: the frozen PyInstaller sidecar next to the app executable (shipped builds), or `engine/.venv` + `buzzdetect_cli.py` (a checkout). Either way the child's working directory must contain `models/`, `embedders/` and `src/stream/drivers/`, because `engine/src/config.py` addresses all three by relative path — that's the whole reason `scripts/build-engine.mjs` assembles `src-tauri/engine-payload/` rather than freezing them into the binary. Only one analysis runs at a time (`AnalysisState` mutex). Stopping is a request, not a kill: `cancel_analysis` writes `STOP` on the child's stdin (which is why that pipe is kept open after the manifest prompt's `y`), the engine takes that as the coordinator's early-exit path, and the logs keep flowing while the workers wind down and report themselves out. Signals are only the escalation, and only once the engine has gone quiet — `engine/src/pipeline/interrupt.py` is the engine end of that, where a signal, or a second stop request, is the impatient path. The engine's own stdout/stderr lines prefixed `BDPROGRESS ` (`PROGRESS_MARKER`) are JSON and get re-emitted as Tauri's `engine-progress` event; everything else becomes `engine-log`. `emit_progress()` in `engine/src/pipeline/progress_json.py` is the single source of truth for that wire format — if you add a new progress event kind, update both it and the frontend's `run` store (`src/lib/progress.svelte.ts`) together.
- **Manifest locking is duplicated logic, not shared code.** An output folder records the settings (model, output classes) that fixed its result schema in `buzzdetect_manifest.json`, and both sides independently refuse to let a run drift from it: Python via `reconcile_with_manifest`/`pipeline/manifest.py` (prompts y/N on stdin — Rust's `start_analysis` always auto-answers `y` since there's no attached terminal), Svelte via `checkManifest()` in `+page.svelte` (locks classes in the UI, blocks on model mismatch). Changing the manifest schema means updating both.
- **`engine/` is just tracked files — edit it directly.** It began as a git subtree of a separate upstream repo, and the history still shows that import (`fbc5ee8 Squashed 'engine/' content`), but **that relationship is over**: the GUI and the engine are one project in one repo, `origin` is the only remote, and there is no `upstream` to sync with. **Do not run `git subtree pull/push`** — there is nothing on the other end. Nor is there a "fork delta" to keep small; changes under `engine/` are ordinary commits like any other. `engine/` carries its own `.gitignore` files — don't re-add engine rules to the root one.
- **Two branch layouts coexist right now.** `main` (and `origin/main`) still holds the *engine-only* layout at the repo root — `buzzdetect_cli.py`, `src/`, `embedders/`, `models/` — from before the merge. The unified layout, where that tree lives under `engine/` beside `src-tauri/` and `src/`, is on `gui-overhaul` and branches cut from it. So `git ls-tree main` looks nothing like your working tree, and diffing against `main` is meaningless. Branch from `gui-overhaul`.
- **The engine that ships is not the engine you develop against.** Shipped builds run the ONNX path only: `models/` at the repo root (not `engine/models/`) is the list of models that get bundled, they run through `embedders/yamnet_onnx`, and `engine/requirements-onnx.txt` — no TensorFlow — is what's frozen. `engine/models/` and the TensorFlow `embedders/yamnet` stay for local work, training and the ONNX export tools. If you change YAMNet's front end or retrain a model, re-run the export scripts; the `.onnx` files are build artifacts, not derived at runtime. The CUDA variant is the same engine with `onnxruntime-gpu` and the NVIDIA runtime wheels; those libraries are deliberately kept *out* of the frozen binary (`strip_nvidia` in `engine/buzzdetect.spec`) and shipped as loose files in `engine-payload/nvidia/`, because a 2.5GB single file is more than makensis will bundle. `start_analysis` puts that directory on the child's `LD_LIBRARY_PATH`/`PATH` — if you move it, move both ends.
- **GPU availability is two questions, answered in two places.** Whether the *build* has a GPU execution provider is decided at build time and written to `engine-payload/gpu-providers.json`; whether *this machine* can actually run one can only be found by trying, because `ort.get_available_providers()` reports what onnxruntime was compiled with and says the same thing on a CUDA workstation and a laptop with no NVIDIA hardware. So the default builds carry `onnxruntime-gpu` *without* its `[cuda,cudnn]` extras -- the CUDA provider, none of the 2.7GB runtime -- and `probe_gpu` (`engine/src/inference/onnx.py`, reached via `buzzdetect_cli.py --probe_gpu`) builds a real session to see which provider survives. `gpu_status` in `src-tauri/src/lib.rs` runs that once at startup and the frontend shows a spinner until it answers. Two builds skip the probe because they have nothing to discover: the bundled-CUDA one (`engine-payload/nvidia` exists) and any CoreML-only one (CoreML ships with macOS). If you add a GPU provider, decide which of those two groups it's in.
- `engine/src/gui/` is the legacy CustomTkinter GUI, superseded by the Tauri frontend; don't extend it. Same for `engine/docs/`, `engine/audio_in/`, and `engine/buzzdetect_gui.py` — they are excluded at bundle time rather than deleted. That was originally to avoid conflicting on every subtree pull; with the repos merged, deleting them is now merely a decision nobody has made, so leave them alone unless asked.
