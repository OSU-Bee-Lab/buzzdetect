Getting started
==========================================

There are two ways to use buzzdetect.
The desktop app provides a familiar point-and-click interface and is recommended for anyone who just wants to analyze data.
The engine is the underlying source code that buzzdetect uses to read audio, load models, analyze data, and write results.
You might want to download the engine if you want to fork the project or just poke around buzzdetect's guts.


Desktop app
------------

To get the desktop app, all you have to do is download the installer for your platform and run it.
The analysis engine and latest models ship inside the app, so you don't have to mess around with the command line and installing dependencies.
See :doc:`gui` for a walkthrough of using the app.

.. |badge-macos-arm| image:: https://img.shields.io/badge/Download-macOS%20Apple%20Silicon-000?style=for-the-badge&logo=apple&logoColor=white
   :target: https://github.com/OSU-Bee-Lab/buzzdetect/releases/latest/download/buzzdetect-macOS-AppleSilicon.dmg
   :alt: Download for macOS (Apple Silicon)

.. |badge-macos-intel| image:: https://img.shields.io/badge/Download-macOS%20Intel-555?style=for-the-badge&logo=apple&logoColor=white
   :target: https://github.com/OSU-Bee-Lab/buzzdetect/releases/latest/download/buzzdetect-macOS-Intel.dmg
   :alt: Download for macOS (Intel)

.. |badge-windows| image:: https://img.shields.io/badge/Download-Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white
   :target: https://github.com/OSU-Bee-Lab/buzzdetect/releases/latest/download/buzzdetect-Windows.exe
   :alt: Download for Windows

.. |badge-windows-cuda| image:: https://img.shields.io/badge/Download-Windows%20CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white
   :target: https://github.com/OSU-Bee-Lab/buzzdetect/releases/latest/download/buzzdetect-Windows-CUDA.zip
   :alt: Download for Windows (CUDA)

.. |badge-linux-deb| image:: https://img.shields.io/badge/Download-.deb-FCC624?style=for-the-badge&logo=linux&logoColor=black
   :target: https://github.com/OSU-Bee-Lab/buzzdetect/releases/latest/download/buzzdetect-Linux.deb
   :alt: Download for Linux (.deb)

|badge-macos-arm| |badge-macos-intel| |badge-windows| |badge-windows-cuda| |badge-linux-deb|

All builds are attached to the `latest GitHub release <https://github.com/OSU-Bee-Lab/buzzdetect/releases/latest>`_.

If you have an NVIDIA GPU or an Apple Silicon Mac, see :doc:`gpu` for how to use it.

macOS woes
^^^^^^^^^^^

buzzdetect is not code-signed, because we cannot afford the developer licenses. Because of this, macOS helpfully pitches a tantrum when you try to use this tool.

Install it to ``/Applications`` (drag it there from the DMG), then open it. You'll get one of two complaints, and they need different answers.

**"buzzdetect.app" is damaged and can't be opened.** Nothing is damaged — this is what macOS says about an unsigned app it has quarantined, and there's no button that gets past it. Remove the quarantine flag in Terminal:

::

    xattr -dr com.apple.quarantine /Applications/buzzdetect.app

Then open it normally.

**"buzzdetect.app" Not Opened.** The milder version, which you can click through:

1. Click "Done".
2. Open System Settings → Privacy & Security, scroll to the Security section near the bottom. You should see ""buzzdetect.app" was blocked to protect your Mac".

   - Click "Open Anyway"
3. In one last bid to stop you from working, macOS will throw up a dialogue titled "Open "buzzdetect.app"?"

   - Click "Open Anyway"

Either way, you only have to do it once. Thank goodness.

Windows does something similar but less elaborate: SmartScreen shows "Windows protected your PC", and you click "More info" → "Run anyway".

Engine (command line / development)
--------------------------------------

If you want to run the engine directly, here's how to set it up from source.

1. Clone the repository
^^^^^^^^^^^^^^^^^^^^^^^^^

::

    git clone https://github.com/OSU-Bee-Lab/buzzdetect


2. Install dependencies with uv
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The engine uses `uv <https://docs.astral.sh/uv/>`_ to manage its own virtual environment,
separate from the desktop app's Node/Rust toolchain.

::
    cd buzzdetect/engine
    uv venv --python 3.13 .venv
    uv pip install -r requirements.txt

3. Run an analysis
^^^^^^^^^^^^^^^^^^^^

::

    .venv/bin/python3 buzzdetect_cli.py --modelname <name> --dir_audio <dir> --dir_out <dir>

``<name>`` is a model directory under ``models/`` (see ``shipped-models.txt`` for what's
bundled). Run ``buzzdetect_cli.py --help`` for the full set of options, which correspond to the desktop app's settings.

Building it yourself
--------------------------------------

To build the desktop app itself rather than just running the engine, you'll additionally need
`Node <https://nodejs.org/>`_ and a `Rust toolchain <https://rustup.rs/>`_.

::

    npm install
    npm run build:engine     # freezes the Python engine into a sidecar binary
    npx tauri build

To work on it without freezing the engine each time, set up ``engine/.venv`` (``cd engine && uv venv --python 3.13 .venv && uv pip install -r requirements.txt``) and run ``npx tauri dev`` — the app falls back to running the engine from source when no sidecar is present.

Happy listening!
