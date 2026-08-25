# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Tauri 2 + SvelteKit desktop GUI wrapped around `engine/`, a forked-in copy of the [buzzdetect](https://github.com/OSU-Bee-Lab/buzzdetect) bioacoustics pipeline. The GUI lets a user pick a model, an input audio folder, and an output folder, then runs the Python engine as a subprocess and renders its live progress as a file tree with progress bars.

Two codebases live side by side and only talk to each other through a subprocess boundary (Rust spawns Python, Python prints structured JSON lines on stdout):

- **Frontend**: `src/` (SvelteKit, Svelte 5 runes) + `src-tauri/` (Rust/Tauri commands)
- **Engine**: `engine/` — a standalone Python package with its own venv, CLI, and models. Treat it as a vendored/forked dependency: it can be run and tested completely independently of the Tauri app via `buzzdetect_cli.py`.

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
```

Engine (run from `engine/`, using its own venv):
```
uv venv --python 3.13 .venv && uv pip install -r requirements.txt   # one-time setup
.venv/bin/python3 buzzdetect_cli.py --modelname <name> --dir_audio <dir> --dir_out <dir>
```
There is no test suite in this repo currently.

## Gotchas / non-obvious behavior

- **Subprocess boundary, not a library call.** `src-tauri/src/lib.rs::start_analysis` spawns `buzzdetect_cli.py` as a child process (fixed interpreter at `engine/.venv/bin/python3`; `resolve_engine_dir` picks bundled-resource vs. `../engine` dev path). Only one analysis runs at a time (`AnalysisState` mutex). The engine's own stdout/stderr lines prefixed `BDPROGRESS ` (`PROGRESS_MARKER`) are JSON and get re-emitted as Tauri's `engine-progress` event; everything else becomes `engine-log`. `emit_progress()` in `engine/src/pipeline/progress_json.py` is the single source of truth for that wire format — if you add a new progress event kind, update both it and the frontend's `run` store (`src/lib/progress.svelte.ts`) together.
- **Manifest locking is duplicated logic, not shared code.** An output folder records the settings (model, output classes, framehop) that fixed its result schema in `buzzdetect_manifest.json`, and both sides independently refuse to let a run drift from it: Python via `reconcile_with_manifest`/`pipeline/manifest.py` (prompts y/N on stdin — Rust's `start_analysis` always auto-answers `y` since there's no attached terminal), Svelte via `checkManifest()` in `+page.svelte` (locks classes/framehop in the UI, blocks on model mismatch). Changing the manifest schema means updating both.
- `engine/` is a vendored fork of github.com/OSU-Bee-Lab/buzzdetect — runnable and testable standalone via `buzzdetect_cli.py`, independent of the Tauri app. `engine/src/gui/` is that upstream project's legacy CustomTkinter GUI, superseded here by the Tauri frontend; don't extend it.
