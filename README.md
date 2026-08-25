# buzzdetect2 — desktop app for buzzdetect

[![Latest release](https://img.shields.io/github/v/release/LukeHearon/buzzdetect-2)](https://github.com/LukeHearon/buzzdetect-2/releases/latest)

buzzdetect2 is a desktop front end for [buzzdetect](https://github.com/OSU-Bee-Lab/buzzdetect), the OSU Bee Lab's bioacoustics classifier. Point it at a folder of audio, pick a model, pick somewhere for the results to land, and press Analyze. You get a CSV per audio file giving each model class's activation over time.

The whole analysis engine ships inside the app. There is no Python to install, no environment to build, no dependencies to chase — download it, open it, run it.

### Warning: vibes

The desktop app around buzzdetect is built pretty much entirely via LLMs; I don't write the Rust or the Svelte it's made of. What it drives is buzzdetect itself — the same streaming code, the same trained models.

One thing does differ, and it's worth stating plainly. So the app can ship without a 1GB TensorFlow install, the models are converted to ONNX ahead of time and run through onnxruntime. That conversion is checked at every step against the TensorFlow original: log-mel features, embeddings, and classifier outputs all agree to within float32 rounding (~1e-5), and on test audio the two produce identical result files. The conversion tools re-run those checks on every export and fail rather than ship a mismatch.

## Install

[![Download for macOS (Apple Silicon)](https://img.shields.io/badge/Download-macOS%20Apple%20Silicon-000?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/LukeHearon/buzzdetect-2/releases/latest/download/buzzdetect2-macOS-AppleSilicon.dmg) [![Download for macOS (Intel)](https://img.shields.io/badge/Download-macOS%20Intel-555?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/LukeHearon/buzzdetect-2/releases/latest/download/buzzdetect2-macOS-Intel.dmg) [![Download for Windows](https://img.shields.io/badge/Download-Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)](https://github.com/LukeHearon/buzzdetect-2/releases/latest/download/buzzdetect2-Windows.exe) [![Download for Linux (.deb)](https://img.shields.io/badge/Download-.deb-FCC624?style=for-the-badge&logo=linux&logoColor=black)](https://github.com/LukeHearon/buzzdetect-2/releases/latest/download/buzzdetect2-Linux.deb)

Pick the right macOS build: **Apple Silicon** for M1 and later, **Intel** for pre-2020 Macs. There's no universal build — the machine learning runtime the engine uses isn't published in a form that allows one.

Mac users will have to fight against macOS to open buzzdetect2. More on that [below](#macos-woes).

On Debian/Ubuntu, use the `.deb`:

``` sh
sudo apt install ./buzzdetect2-Linux.deb
```

On other distros, grab the [AppImage](https://github.com/LukeHearon/buzzdetect-2/releases/latest/download/buzzdetect2-Linux.AppImage) instead.

## Overview

Three settings get you running: a **model**, an **audio directory** to read, and an **output directory** to write into. Everything else has a working default and lives behind Advanced.

Analysis is *resumable and idempotent*. Results are written incrementally as each chunk finishes, so an interrupted run isn't lost work — start it again and it picks up from where the partial results stop. Files that already have complete results are skipped outright. The output folder records the settings that determined its result schema, and both the app and the engine refuse to write results into it under settings that don't match, so a folder can't end up holding two incompatible kinds of CSV.

## Features

- **A real progress view.** Your audio directory is drawn as a file tree with a progress bar per file and per folder, filling in as chunks complete. Files that get skipped say why — already analyzed, too small, name conflict.
- **Honest time estimates.** A rolling analysis rate (as a multiple of realtime), how much audio is left, and an ETA — all weighted by actual audio duration rather than file count, so a folder of unevenly sized recordings still estimates sensibly.
- **Pick your output classes.** Write out every class the model knows, or just the ones you care about.
- **Tunable throughput.** Chunk length, number of parallel analyzers, number of concurrent file readers, and stream buffer depth are all adjustable when the defaults don't suit your machine.
- **Stop whenever.** Cancelling leaves completed chunks on disk, ready to resume.
- **A log you can actually read**, with separate verbosity for the console pane and the log file.
- **Broad format support.** mp3, mp4, mts, wma, wav, flac and everything else libsndfile handles — including recorder files with corrupt tails, which are read up to the damage rather than thrown out.
- **Settings persist** between launches, so reopening the app puts you back where you were.

## macOS woes

buzzdetect2 is not code-signed, because I cannot afford the developer licenses. Because of this, macOS helpfully pitches a tantrum when you try to use this tool. Here's how to get past the worst of it:

1.  Install buzzdetect2 to `/Applications` (drag it there from the DMG).
2.  Open buzzdetect2. You'll see a "buzzdetect2.app" Not Opened popup.
    - Click "Done"
3.  Open System Settings → Privacy & Security, scroll to the Security section near the bottom. You should see ""buzzdetect2.app" was blocked to protect your Mac".
    - Click "Open Anyway"
4.  In one last bid to stop you from working, macOS will throw up a dialogue titled "Open "buzzdetect2.app"?"
    - Click "Open Anyway"

You will only have to do this once. Thank goodness.

Windows does something similar but less elaborate: SmartScreen shows "Windows protected your PC", and you click "More info" → "Run anyway".

## Building it yourself

You need [Node](https://nodejs.org/), a [Rust toolchain](https://rustup.rs/), and [uv](https://docs.astral.sh/uv/).

``` sh
npm install
npm run build:engine     # freezes the Python engine into a sidecar binary
npx tauri build
```

To work on it without freezing the engine each time, set up `engine/.venv` (`cd engine && uv venv --python 3.13 .venv && uv pip install -r requirements.txt`) and run `npx tauri dev` — the app falls back to running the engine from source when no sidecar is present.

`engine/` is a [git subtree](https://www.atlassian.com/git/tutorials/git-subtree) of upstream buzzdetect, so fixes flow in with `git subtree pull --prefix=engine upstream main --squash`.
