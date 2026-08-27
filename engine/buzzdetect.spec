# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the frozen buzzdetect engine.

Produces a single `buzzdetect-engine` executable that the desktop app spawns
as a sidecar, so users don't need Python or a venv. Named for the app rather
than after it: Tauri installs an externalBin beside the app executable, which
is itself named buzzdetect, so a sidecar called buzzdetect would be the same
file. Build it through
`node scripts/build-engine.mjs` from the repo root rather than calling
pyinstaller directly -- that script creates the TF-free venv this expects and
assembles the data payload that has to sit next to the binary.

Three things this build deliberately does NOT contain:

- TensorFlow. Only the ONNX models ship; see requirements-onnx.txt.
- models/. buzzdetect loads each model by reading its model.py off disk at
  runtime (importlib.util.spec_from_file_location in src/inference/models.py),
  so freezing them in would be pointless -- they're shipped as a Tauri resource
  directory instead, and the app runs the binary with that as its cwd.
- The NVIDIA runtime, on the CUDA build. See strip_nvidia() below.
"""

import os

from PyInstaller.utils.hooks import collect_submodules


def is_nvidia(entry):
    """Whether a PyInstaller TOC entry is a library from an nvidia-* wheel.

    Matched on the source path -- every one of them lives under
    site-packages/nvidia/<component>/ -- rather than on the library name, so a
    component we've never seen before is caught too.
    """
    source = entry[1]
    if not source:
        return False
    parts = os.path.normpath(source).split(os.sep)
    return 'nvidia' in parts and 'site-packages' in parts


def strip_nvidia(binaries):
    """Drop the CUDA runtime from the frozen executable.

    The CUDA build installs onnxruntime-gpu plus the NVIDIA runtime as wheels
    (requirements-onnx-cuda.txt) so the app needs no system CUDA -- but that is
    ~2.5GB of shared libraries, and freezing them into a onefile executable
    produces a single file no installer will take: makensis mmaps each input
    whole and is 32-bit, so it aborts outright.

    So they ship as loose files in the Tauri resource payload instead
    (scripts/build-engine.mjs copies them to engine-payload/nvidia), and
    src-tauri/src/lib.rs points the sidecar's loader at that directory when it
    spawns it. Nothing here has to find them: onnxruntime dlopen()s them by
    soname, which is what the loader search path is for.
    """
    kept = [entry for entry in binaries if not is_nvidia(entry)]
    dropped = len(binaries) - len(kept)
    print(f'buzzdetect.spec: excluded {dropped} NVIDIA runtime libraries '
          f'(they ship in the payload, not the executable)')
    return kept


# src/stream/audio.py builds its driver map by listing src/stream/drivers and
# importing each module by name, which no static analysis can see.
hidden = collect_submodules('src.stream.drivers')

# Reached only from model.py files loaded off disk at runtime, so likewise
# invisible to the import graph. Without it the app raises ImportError on the
# first chunk it tries to analyse.
hidden += ['onnxruntime']

a = Analysis(
    ['buzzdetect_cli.py'],
    pathex=['.'],
    binaries=[],
    datas=[],
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=['pyinstaller_rthook_nvidia.py'],
    # Belt and braces: these should be absent from the build venv anyway, and
    # if one sneaks back in as a transitive dependency it would add hundreds of
    # megabytes to every installer without anything asking for it.
    excludes=[
        'tensorflow',
        'keras',
        'librosa',
        'numba',
        'llvmlite',
        'scipy',
        'matplotlib',
        'IPython',
        'tkinter',
        'src.gui',
    ],
    noarchive=False,
)

# PyInstaller pulls these in on its own by following onnxruntime_providers_*'s
# NEEDED/import-table entries, so this runs unconditionally -- on the CPU build
# there is simply nothing to drop.
a.binaries = strip_nvidia(a.binaries)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='buzzdetect-engine',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
