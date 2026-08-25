# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the frozen buzzdetect engine.

Produces a single `buzzdetect` executable that the desktop app spawns as a
sidecar, so users don't need Python or a venv. Build it through
`node scripts/build-engine.mjs` from the repo root rather than calling
pyinstaller directly -- that script creates the TF-free venv this expects and
assembles the data payload that has to sit next to the binary.

Two things this build deliberately does NOT contain:

- TensorFlow. Only the ONNX models ship; see requirements-onnx.txt.
- models/ and embedders/. buzzdetect loads those by reading .py files off disk
  at runtime (importlib.util.spec_from_file_location in src/inference), so
  freezing them in would be pointless -- they're shipped as a Tauri resource
  directory instead, and the app runs the binary with that as its cwd.
"""

from PyInstaller.utils.hooks import collect_submodules

# src/stream/audio.py builds its driver map by listing src/stream/drivers and
# importing each module by name, which no static analysis can see.
hidden = collect_submodules('src.stream.drivers')

# Reached only from plugin .py files loaded off disk at runtime, so likewise
# invisible to the import graph. Without these the app raises ImportError on
# the first chunk it tries to analyse.
hidden += [
    'onnxruntime',
    'embedders.yamnet_onnx.features',
    'embedders.yamnet_onnx.params',
]

a = Analysis(
    ['buzzdetect_cli.py'],
    pathex=['.'],
    binaries=[],
    datas=[],
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
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

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='buzzdetect',
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
