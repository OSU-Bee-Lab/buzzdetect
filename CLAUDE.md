# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Tauri 2 + SvelteKit desktop GUI wrapped around `engine/`, a forked-in copy of the [buzzdetect](https://github.com/OSU-Bee-Lab/buzzdetect) bioacoustics pipeline. The GUI lets a user pick a model, an input audio folder, and an output folder, then runs the Python engine as a subprocess and renders its live progress as a file tree with progress bars.

Two codebases live side by side and only talk to each other through a subprocess boundary (Rust spawns Python, Python prints structured JSON lines on stdout):

- **Frontend**: `src/` (SvelteKit, Svelte 5 runes) + `src-tauri/` (Rust/Tauri commands)
- **Engine**: `engine/` — a standalone Python package with its own venv, CLI, and models. Treat it as a vendored/forked dependency: it can be run and tested completely independently of the Tauri app via `buzzdetect_cli.py`.

## Workflow

Commit as you go — after each self-contained change, not batched at the end of a session.

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

## Architecture

### Process boundary (src-tauri/src/lib.rs)

- `resolve_engine_dir` picks the bundled `engine/` resource dir in a built app, or `../engine` next to the project root in dev. The Python interpreter is always `engine/.venv/bin/python3` — fixed, not user-configurable, since `engine/` ships its own venv.
- `start_analysis` spawns `buzzdetect_cli.py` as a child process with piped stdio, translating the `AnalysisSettings` struct (from the frontend) into CLI flags. Only one analysis can run at a time (`AnalysisState` mutex).
- The engine's stdout/stderr are read line-by-line in background threads (`spawn_line_reader`). Lines prefixed `BDPROGRESS ` (see `PROGRESS_MARKER`) are parsed as JSON and re-emitted as a Tauri `engine-progress` event; everything else becomes an `engine-log` event. A separate poller thread emits `engine-exit` once the child exits.
- `cancel_analysis` kills the child process directly.
- `read_manifest` / `list_models` / `get_model_classes` read straight from the filesystem under `engine/models/<modelname>/` — no engine process involved.

### Manifest locking (schema safety)

An output folder can accumulate results across multiple runs. `buzzdetect_manifest.json` in `dir_out` records the settings (model, output classes, framehop) that determine the result schema. Both sides enforce the same rule independently:
- Python: `engine/src/pipeline/manifest.py` + `reconcile_with_manifest` in `buzzdetect_cli.py` (prompts y/N on stdin if settings conflict; `start_analysis` in Rust always answers `y` since there's no attached terminal).
- Svelte: `checkManifest()` in `src/routes/+page.svelte` reads the manifest via the `read_manifest` command and locks the classes/framehop fields in the UI to match once the selected model matches the manifest's. A selected model that doesn't match the manifest's is surfaced as a blocking error (`modelMismatch`) rather than silently overridden, so the user can't pick settings that would fail engine-side reconciliation.

Keep these two implementations in sync if the manifest schema changes.

### Progress protocol

`engine/src/pipeline/progress_json.py`'s `emit_progress(event, **fields)` is the single source of truth for the wire format; every progress event goes through it. Event kinds currently include `manifest` / `manifest_done` (file discovery), `file_skip`, and per-chunk progress used to drive the frontend's file tree and progress bars. `src/lib/progress.svelte.ts` (`run` store) is the frontend counterpart — it owns the running/error state, builds the nested directory tree from flat file paths, and computes aggregate seconds-done/seconds-total. If you add a new progress event kind, update both sides together.

### Engine internals (engine/src/)

Pipeline is producer/consumer across threads, coordinated by `pipeline/coordination.py::Coordinator`:
- `stream/` — streamer workers read audio files in chunks (format-specific drivers in `stream/drivers/`) and enqueue them.
- `inference/` — analyzer workers (CPU and/or GPU, count set by `analyzers_cpu`/`analyzers_gpu`) run the model (`inference/models.py::load_model`) over queued chunks.
- `write/` — writer worker accumulates per-file results and writes CSVs (`_buzzpart.csv` while partial, renamed to `_buzzdetect.csv` — see `config.py` suffixes — once a file is fully covered).
- `pipeline/logger.py` + `loglevels.py` — central logging thread; console and log-file verbosity are configured independently (`verbosity_print` vs `verbosity_log`).
- `analyze.py::Analyzer` wires all of the above together and is the entry point used by both `buzzdetect_cli.py` and (historically) `src/gui/` — a legacy CustomTkinter GUI now superseded by the Tauri frontend in this repo.
- Models live in `engine/models/<modelname>/`, each with `model.py`, `model.keras`, and `config_model.json` (holds the `classes` list read by `get_model_classes`). A directory only counts as a model if it has a `model.py` (see `list_models` in `src-tauri/src/lib.rs`).

### Frontend structure (src/)

- `src/routes/+page.svelte` — the entire UI: settings panel (model/dirs/advanced params) + run panel (progress tree + log). No routing beyond this single page.
- `src/lib/settings.svelte.ts` — persisted user settings (Svelte 5 `$state` rune-based store).
- `src/lib/progress.svelte.ts` — `run` store: ingests `engine-progress`/`engine-log`/`engine-exit` events into UI-ready state (file tree, rates, done/total seconds).
- `src/lib/DirRow.svelte` — recursive directory row component for the progress tree.
- Built with `@sveltejs/adapter-static`, since Tauri needs a static frontend bundle.
