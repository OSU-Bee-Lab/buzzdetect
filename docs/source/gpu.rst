GPU acceleration
==========================================

Follow the instructions below to set up processing on GPU. Then, set **GPU analyzers** to 1 in :doc:`Advanced settings <gui>` to use the card.
The GPU analyzers setting only appears in builds that support processing on GPU, and only if buzzdetect can find the GPU.
You'll likely want to set **CPU analyzers** to 0 and otherwise do some tuning. See: :doc:`tuning`

Windows
------------------

If you're on Windows and have an NVIDIA card, use the Windows CUDA build to runinference on the GPU.
This build bundles the CUDA runtime, so you need a recent NVIDIA driver but you don't need to install CUDA to your system.
Turing (GTX 16-series, RTX 20-series) and newer are supported.
Because of its size, it ships as a portable zip rather than an installer. Unpack it anywhere and run ``buzzdetect-cuda.exe``.


macOS
------------------------------

On Apple Silicon the regular macOS build already has a GPU option — it runs the model on the GPU through CoreML, at full float32 precision, for a bit over 2x end to end.
Nothing extra to install.

Set **GPU analyzers** to 1 and **CPU analyzers** to 0 in :doc:`Advanced settings <gui>` to use the card.
You can also enable **Reduced precision (fp16)** for roughly another 2x — see :doc:`gui` and :doc:`tuning`.


Linux
------------------------------

There's no separate CUDA build for Linux — the ordinary ``.deb``/AppImage carries the CUDA execution provider already.
Install CUDA 12 and cuDNN 9 yourself (Turing or newer GPU, driver 525+) and buzzdetect will find them:

::

    sudo apt install cuda-runtime-12-9 libcudnn9-cuda-12

(Debian/Ubuntu, after adding NVIDIA's ``cuda-keyring`` per `their instructions <https://developer.nvidia.com/cuda-downloads>`_. Any CUDA 12 point release works.)

If the **GPU analyzers** setting doesn't appear, check ``nvidia-smi`` (driver present and current) and ``ldconfig -p | grep -E 'libcudart|libcudnn'`` (libraries on the loader path).
If two cuDNN installations are both reachable — e.g. a pip wheel's entry in ``/etc/ld.so.conf.d/`` alongside a system install — the mix can pass every startup check and only fail partway through an analysis with ``CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH``; remove one of the two to fix it.

