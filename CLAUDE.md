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
There is no test suite in this repo currently. The closest things to one are
`engine/embedders/yamnet_onnx/BUILD.py` and `engine/tools/onnxify_model.py`,
which re-export their ONNX artifacts and fail if they stop matching the
TensorFlow originals.

## Releases

`npm version <x.y.z>` (which syncs `tauri.conf.json`/`Cargo.toml`/`Cargo.lock`
via `scripts/sync-version.mjs`), then push the tag. `.github/workflows/release.yml`
builds installers for macOS arm64, macOS x86_64, Windows and Linux and attaches
them to a **draft** release, which you publish by hand.

No universal macOS build is possible: onnxruntime ships no universal2 wheel, so
PyInstaller can only freeze the engine for the architecture it's running on.
That's why the matrix has two separate macOS jobs.

## Gotchas / non-obvious behavior

- **Subprocess boundary, not a library call.** `src-tauri/src/lib.rs::start_analysis` spawns the engine as a child process. `resolve_engine` picks between two shapes: the frozen PyInstaller sidecar next to the app executable (shipped builds), or `engine/.venv` + `buzzdetect_cli.py` (a checkout). Either way the child's working directory must contain `models/`, `embedders/` and `src/stream/drivers/`, because `engine/src/config.py` addresses all three by relative path — that's the whole reason `scripts/build-engine.mjs` assembles `src-tauri/engine-payload/` rather than freezing them into the binary. Only one analysis runs at a time (`AnalysisState` mutex). The engine's own stdout/stderr lines prefixed `BDPROGRESS ` (`PROGRESS_MARKER`) are JSON and get re-emitted as Tauri's `engine-progress` event; everything else becomes `engine-log`. `emit_progress()` in `engine/src/pipeline/progress_json.py` is the single source of truth for that wire format — if you add a new progress event kind, update both it and the frontend's `run` store (`src/lib/progress.svelte.ts`) together.
- **Manifest locking is duplicated logic, not shared code.** An output folder records the settings (model, output classes) that fixed its result schema in `buzzdetect_manifest.json`, and both sides independently refuse to let a run drift from it: Python via `reconcile_with_manifest`/`pipeline/manifest.py` (prompts y/N on stdin — Rust's `start_analysis` always auto-answers `y` since there's no attached terminal), Svelte via `checkManifest()` in `+page.svelte` (locks classes in the UI, blocks on model mismatch). Changing the manifest schema means updating both.
- **`engine/` is a git subtree, not a copy.** It tracks `upstream` = github.com/OSU-Bee-Lab/buzzdetect. Pull upstream work with `git subtree pull --prefix=engine upstream main --squash`; push work the other way with `git subtree push --prefix=engine upstream <branch>`. Keep the fork delta small — anything that isn't GUI-specific (new stream drivers, streaming fixes, format support) belongs upstream, where it comes back for free on the next pull. The current delta is `src/pipeline/progress_json.py` and its `emit_progress()` call sites, `search_dir` in `src/utils.py` being a generator, driver precedence in `src/stream/audio.py`, and `requirements.txt`. `engine/` carries its own `.gitignore` files — don't re-add engine rules to the root one.
- **The engine that ships is not the engine you develop against.** Shipped builds run the ONNX path only: `models/` at the repo root (not `engine/models/`) is the list of models that get bundled, they run through `embedders/yamnet_onnx`, and `engine/requirements-onnx.txt` — no TensorFlow — is what's frozen. `engine/models/` and the TensorFlow `embedders/yamnet` stay for local work, training and the ONNX export tools. If you change YAMNet's front end or retrain a model, re-run the export scripts; the `.onnx` files are build artifacts, not derived at runtime. The CUDA variant is the same engine with `onnxruntime-gpu` and the NVIDIA runtime wheels; those libraries are deliberately kept *out* of the frozen binary (`strip_nvidia` in `engine/buzzdetect.spec`) and shipped as loose files in `engine-payload/nvidia/`, because a 2.5GB single file is more than makensis will bundle. `start_analysis` puts that directory on the child's `LD_LIBRARY_PATH`/`PATH` — if you move it, move both ends.
- `engine/src/gui/` is upstream's legacy CustomTkinter GUI, superseded here by the Tauri frontend; don't extend it. Same for `engine/docs/`, `engine/audio_in/`, and `engine/buzzdetect_gui.py` — they arrive with the subtree and are excluded at bundle time rather than deleted, since deleting them would conflict on every pull.
