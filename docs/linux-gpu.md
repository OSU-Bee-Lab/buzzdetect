# Using an NVIDIA GPU on Linux

The Linux builds carry the CUDA execution provider but none of the CUDA
runtime, so the card works as soon as the machine has CUDA 12 and cuDNN 9
somewhere the dynamic loader can find them. This page is what to install, how
to check it took, and what the failure modes look like — including one that
reports success and then dies partway through an analysis.

Windows has a bundled-CUDA build that needs none of this. There is no Linux
equivalent, for a packaging reason rather than a technical one: the NVIDIA
runtime is ~2.7GB, `nvprune` only accepts static libraries rather than the
prebuilt `.so` files the wheels ship, and cuDNN's sub-libraries aren't safely
separable. So Linux asks you to install CUDA yourself and the ordinary
`.deb`/AppImage finds it.

## What you need

| | |
| --- | --- |
| GPU | Turing or newer — GTX 16-series, RTX 20-series and up |
| Driver | 525 or newer |
| Libraries | the CUDA 12 **runtime** libraries, and cuDNN 9 |

The runtime libraries, not the toolkit: nothing here compiles CUDA code, so
there is no need for `nvcc` or a `cuda-toolkit` package.

The driver floor comes from the onnxruntime pin. The app ships
`onnxruntime-gpu==1.26.0`, which is built against CUDA 12 and works from driver
~525 up. The 1.27 line moved to CUDA 13 and a ~580 driver, which is why the pin
is deliberate (`engine/requirements-onnx.txt`) — a newer onnxruntime would
quietly raise the driver requirement for everyone.

`nvidia-smi` reports your driver version. The "CUDA Version" it prints is the
highest your driver supports, not what is installed.

## Installing

On Debian/Ubuntu, from NVIDIA's own repository — add `cuda-keyring` following
[NVIDIA's instructions](https://developer.nvidia.com/cuda-downloads) for your
release, then:

```
sudo apt install cuda-runtime-12-9 libcudnn9-cuda-12
```

Adjust `12-9` to whichever CUDA 12 point release the repo offers; any of them
works, as they share the `libcudart.so.12` soname.

Distribution packages put the libraries somewhere `ldconfig` already searches.
If you install by unpacking a tarball instead, add its directory to
`/etc/ld.so.conf.d/` and run `sudo ldconfig` — but read the last section of
this page first, because that directory is where the interesting failures come
from.

## Checking it worked

The app probes the GPU once at startup and shows a spinner until it answers. If
it finds one, Advanced settings grows a **GPU analyzers** option; set that to 1
and CPU analyzers to 0 to run on the card. No option means the probe found
nothing usable.

To ask directly, run the engine's probe. In an installed app:

```
cd /usr/lib/buzzdetect2/engine-payload && /usr/bin/buzzdetect --probe_gpu
```

or from a checkout, `engine/.venv/bin/python3 buzzdetect_cli.py --probe_gpu`.
It prints one line:

```
{"gpu_providers": ["CUDAExecutionProvider"]}
```

An empty list means no usable GPU. The probe builds a real inference session
*and runs a frame through it*, because both halves catch different failures —
see below.

## When it doesn't work

### The probe returns `[]` and the app only offers the CPU

The CUDA provider could not initialise. Usually the libraries aren't installed,
aren't on the loader path, or the driver is older than the runtime needs.

```
nvidia-smi                        # driver present? version >= 525?
ldconfig -p | grep -E 'libcudart|libcudnn\.so'
```

If `ldconfig` lists nothing, the libraries are installed somewhere it doesn't
search, or not installed at all.

### `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`

```
Non-zero status code returned while running FusedConv node.
Status Message: CUDNN failure 1002: CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH
```

**Two different cuDNN 9 installations are visible to the loader at once, and
the process ended up with pieces of both.** cuDNN 9 is not one library: a small
`libcudnn.so.9` dispatches to `libcudnn_graph`, `libcudnn_engines_*`,
`libcudnn_ops` and friends, resolved separately by soname. Different 9.x
releases ship different sets of those — 9.20 has no
`libcudnn_engines_tensor_ir.so.9`, 9.25 does — so when two installs are both
reachable, the loader can satisfy most sonames from the first and fall through
to the second for one that only it provides. Every version check passes at load
time. The mismatch surfaces on the first convolution, which in an analysis is
minutes in.

The usual source is a pip `nvidia-cudnn-cu12` wheel — dragged in by PyTorch,
TensorFlow, or `onnxruntime-gpu[cuda,cudnn]` — placed on the *global* loader
path via `/etc/ld.so.conf.d/`, on a machine that also has a distribution cuDNN.

Find out whether you have two:

```
ldconfig -p | grep libcudnn.so.9
```

Two lines for one soname is the problem. To see what a running process actually
got, which is the definitive answer:

```
grep -o '/[^ ]*libcudnn[^ ]*' /proc/<pid>/maps | sort -u
```

A healthy process shows one directory and one version. A broken one shows a
mix.

The fix is to make cuDNN resolve to exactly one installation. Prefer removing
the pip wheel's directory from the loader path over removing the system one:
the wheels are a Python package's private dependency and were never meant to be
global.

### Before you remove anything from the loader path

**Check what else only that directory provides.** A pip wheel directory
typically carries the whole NVIDIA runtime, not just cuDNN, and on a machine
with no system CUDA install it may hold the only copy of `libcudart.so.12`.
Deleting the entry outright then breaks the GPU completely, having set out to
fix it.

```
ldconfig -p | grep -E 'libcudart|libcublas\.so|libcufft|libcurand'
```

If those resolve only inside the wheel directory, either delete just the cuDNN
line from the `.conf` file and keep the rest:

```
sudo sed -i '\|nvidia/cudnn/lib|d' /etc/ld.so.conf.d/<file>.conf
sudo ldconfig
```

or install the system CUDA runtime first and then drop the file entirely. Run
`sudo ldconfig` after any change, and re-run `--probe_gpu` to confirm.

### It says it's using the GPU but isn't faster

onnxruntime falls back to the CPU provider per-operator rather than failing, so
a session can report `CUDAExecutionProvider` and still run most of the graph on
the CPU. The engine warns on stderr when it asked for a GPU provider and didn't
get it, and the desktop app surfaces engine stderr in its log pane. Check the
log before assuming the card is being used.

## A note on precision

Advanced settings can let a GPU provider drop to fp16. That is a real speedup
on Apple's Neural Engine and is **not** worth it on NVIDIA: measured on a
GTX 1650, the fp16 graph ran at half the speed of fp32. It also shifts
predictions by ~2e-2, so results stop being comparable with fp32 output at the
margins. Leave it off on Linux.
