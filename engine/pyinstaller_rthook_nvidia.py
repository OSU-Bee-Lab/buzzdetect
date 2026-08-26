"""Runtime hook: make the payload's NVIDIA runtime loadable on Windows.

The CUDA build ships its NVIDIA libraries as loose files in the Tauri resource
payload rather than inside this executable (see strip_nvidia in
buzzdetect.spec), and src-tauri/src/lib.rs puts that directory on PATH when it
spawns us. That is usually enough -- but only usually: once anything in the
process calls SetDefaultDllDirectories, which onnxruntime's own loader does,
PATH stops being searched for dependent DLLs. add_dll_directory registers the
directory with that newer mechanism too, so both loaders can find them.

No Linux equivalent is needed: LD_LIBRARY_PATH has no such override, and it is
read at process start, before this runs.
"""

import os
import sys

if sys.platform == 'win32':
    # The app runs the sidecar with the payload as its working directory --
    # that's what makes engine/src/config.py's relative paths resolve too.
    nvidia = os.path.join(os.getcwd(), 'nvidia')
    if os.path.isdir(nvidia):
        os.add_dll_directory(nvidia)
