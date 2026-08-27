# buzzdetect — desktop app

[![Latest release](https://img.shields.io/github/v/release/OSU-Bee-Lab/buzzdetect)](https://github.com/OSU-Bee-Lab/buzzdetect/releases/latest)

A desktop front end for [buzzdetect](https://github.com/OSU-Bee-Lab/buzzdetect), the OSU Bee Lab's bioacoustics classifier. Point it at a folder of audio, pick a model, pick somewhere for the results to land, and press Analyze. You get a CSV per audio file giving each model class's activation over time.

The whole analysis engine ships inside the app. There is no Python to install, no environment to build, no dependencies to chase — download it, open it, run it.

### Warning: vibes

The desktop app around buzzdetect is built pretty much entirely via LLMs; I don't write the Rust or the Svelte it's made of. What it drives is buzzdetect itself — the same streaming code, the same trained models.

One thing does differ, and it's worth stating plainly. So the app can ship without a 1GB TensorFlow install, the models are converted to ONNX ahead of time and run through onnxruntime. That conversion is checked at every step against the TensorFlow original: log-mel features, embeddings, and classifier outputs all agree to within float32 rounding (~1e-5), and on test audio the two produce identical result files. The conversion tools re-run those checks on every export and fail rather than ship a mismatch.

## Install

[![Download for macOS (Apple Silicon)](https://img.shields.io/badge/Download-macOS%20Apple%20Silicon-000?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/OSU-Bee-Lab/buzzdetect/releases/latest/download/buzzdetect-macOS-AppleSilicon.dmg) [![Download for macOS (Intel)](https://img.shields.io/badge/Download-macOS%20Intel-555?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/OSU-Bee-Lab/buzzdetect/releases/latest/download/buzzdetect-macOS-Intel.dmg) [![Download for Windows](https://img.shields.io/badge/Download-Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)](https://github.com/OSU-Bee-Lab/buzzdetect/releases/latest/download/buzzdetect-Windows.exe) [![Download for Linux (.deb)](https://img.shields.io/badge/Download-.deb-FCC624?style=for-the-badge&logo=linux&logoColor=black)](https://github.com/OSU-Bee-Lab/buzzdetect/releases/latest/download/buzzdetect-Linux.deb)

Pick the right macOS build: **Apple Silicon** for M1 and later, **Intel** for pre-2020 Macs. There's no universal build — the machine learning runtime the engine uses isn't published in a form that allows one.

Mac users will have to fight against macOS to open buzzdetect. More on that [below](#macos-woes).

On Debian/Ubuntu, use the `.deb`:

``` sh
sudo apt install ./buzzdetect-Linux.deb
```

On other distros, grab the [AppImage](https://github.com/OSU-Bee-Lab/buzzdetect/releases/latest/download/buzzdetect-Linux.AppImage) instead.

### NVIDIA GPU builds

If you have an NVIDIA card on Windows, there's a separate CUDA build that runs inference on the GPU: [Windows CUDA](https://github.com/OSU-Bee-Lab/buzzdetect/releases/latest/download/buzzdetect-Windows-CUDA.zip). It bundles the CUDA runtime, so you need a recent NVIDIA driver but no system CUDA install. Turing (GTX 16-series, RTX 20-series) and newer are supported. It ships as a portable zip rather than an installer — unpack it anywhere and run `buzzdetect-cuda.exe` — because the bundled runtime is larger than the installer format can take.

On Linux there's no separate CUDA build for the same size reason. Install CUDA 12 and cuDNN 9 yourself and the ordinary `.deb`/AppImage finds them.

It's about a gigabyte, which is why it's separate rather than the default. Set GPU analyzers to 1 and CPU analyzers to 0 in Advanced settings to use the card. The GPU analyzers setting only appears in builds whose engine can actually use one, and if a CUDA build can't reach your card it says so in the log and falls back to the CPU rather than pretending.

On Apple Silicon the regular macOS build already has a GPU option — it runs the model on the GPU through CoreML, at full float32 precision, for a bit over 2x end to end. Nothing extra to install.

The CUDA build installs alongside the regular one rather than replacing it.

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

buzzdetect is not code-signed, because I cannot afford the developer licenses. Because of this, macOS helpfully pitches a tantrum when you try to use this tool.

Install it to `/Applications` (drag it there from the DMG), then open it. You'll get one of two complaints, and they need different answers.

**"buzzdetect.app" is damaged and can't be opened.** Nothing is damaged — this is what macOS says about an unsigned app it has quarantined, and there's no button that gets past it. Remove the quarantine flag in Terminal:

``` sh
xattr -dr com.apple.quarantine /Applications/buzzdetect.app
```

Then open it normally.

**"buzzdetect.app" Not Opened.** The milder version, which you can click through:

1.  Click "Done".
2.  Open System Settings → Privacy & Security, scroll to the Security section near the bottom. You should see ""buzzdetect.app" was blocked to protect your Mac".
    - Click "Open Anyway"
3.  In one last bid to stop you from working, macOS will throw up a dialogue titled "Open "buzzdetect.app"?"
    - Click "Open Anyway"

Either way, you only have to do it once. Thank goodness.

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
