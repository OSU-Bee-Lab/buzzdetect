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

import glob
import os
import site

from PyInstaller.utils.hooks import collect_submodules

# Set by scripts/build-engine.mjs --cuda. The CUDA build installs
# onnxruntime-gpu plus the NVIDIA runtime as wheels (requirements-onnx-cuda.txt)
# so the app doesn't need a system CUDA install.
CUDA = os.environ.get('BUZZDETECT_CUDA') == '1'


def nvidia_libraries():
    """Shared libraries from the nvidia-* wheels, flattened to the bundle root.

    PyInstaller's onefile bootloader puts _MEIPASS on the library search path
    but not its subdirectories, and onnxruntime's CUDA provider dlopen()s these
    by soname (libcudnn.so.9 and friends). Keeping the nvidia/<component>/lib
    layout would leave them unfindable, so they're flattened; the sonames are
    distinct, so nothing collides.
    """
    found = []
    for site_packages in site.getsitepackages():
        root = os.path.join(site_packages, 'nvidia')
        if not os.path.isdir(root):
            continue
        for pattern in ('*/lib/*.so*', '*/lib/x64/*.lib', '*/bin/*.dll'):
            for path in glob.glob(os.path.join(root, pattern)):
                if os.path.isfile(path):
                    found.append((path, '.'))
    return found



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

if CUDA:
    binaries = nvidia_libraries()
    # Fail here rather than ship a CUDA installer with no CUDA in it. Without
    # this the build succeeds, the app runs, and the only symptom is a GPU
    # analyzer quietly falling back to the CPU on the user's machine.
    if not binaries:
        raise SystemExit(
            'CUDA build requested but no nvidia-* shared libraries were found. '
            'Check that requirements-onnx-cuda.txt installed the cuda/cudnn '
            'extras into this venv.'
        )
    print(f'buzzdetect.spec: bundling {len(binaries)} NVIDIA runtime libraries')
else:
    binaries = []

a = Analysis(
    ['buzzdetect_cli.py'],
    pathex=['.'],
    binaries=binaries,
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
