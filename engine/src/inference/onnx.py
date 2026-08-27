"""Execution-provider selection for the ONNX models.

onnxruntime picks up a GPU only if the installed package actually carries a GPU
execution provider -- the plain `onnxruntime` wheel does not, `onnxruntime-gpu`
does. Getting that wrong is quiet by default: a session built with
CPUExecutionProvider runs perfectly happily at a fraction of the speed, with
nothing in the logs to say the GPU worker never touched the GPU. This module
exists so that failure is loud instead.
"""

import glob
import os
import sys

import numpy as np
import onnxruntime as ort

# GPU providers worth trying, best first, with the options each needs.
# onnxruntime falls back along the list per-operator, so CPU is always
# appended as the backstop.
#
# CoreML is pinned to ModelFormat=MLProgram deliberately, and that is now
# load-bearing rather than a preference. Measured on the fused waveform-in
# graph, 200 s of audio on an Apple M1 (benchmarks/onnx-vs-tf/COREML.md):
#
#   CPU                                 231 ms   reference
#   CoreML, default NeuralNetwork       144 ms   predictions off by 1.6e-2
#   CoreML, MLProgram + CPUAndGPU        68 ms   predictions off by 4.5e-6
#
# The default NeuralNetwork format runs fp16 on the Neural Engine, and 1.6e-2
# is three hundred times our float32 parity budget -- results would depend on
# which machine analysed them. It also declines the com.microsoft.FusedConv
# nodes the export bakes in, so all 27 convolutions fall back to the CPU and
# the graph ends up slower than not using CoreML at all. MLProgram keeps fp32,
# takes the whole graph in two partitions, and is 3.4x the CPU.
GPU_PROVIDERS = (
    ('CUDAExecutionProvider', {}),
    ('ROCMExecutionProvider', {}),
    ('CoreMLExecutionProvider', {'ModelFormat': 'MLProgram', 'MLComputeUnits': 'CPUAndGPU'}),
)

# The symbolic dimension the export gives a model's waveform input. Fixing it
# before session creation is not an optimisation: CoreML's MLProgram format
# cannot compile a graph with an unbounded dimension and the session fails
# outright, and a fixed length is what lets CoreML constant-fold the front
# end's shape arithmetic and swallow the graph whole.
DIM_SAMPLES = 'samples'

# Set to '1' to let a GPU provider drop to reduced precision. Deliberately an
# environment variable rather than an analysis parameter: it changes nothing
# about the result schema, so it doesn't belong in the manifest, and the
# desktop app sets it per run from a checkbox.
ENV_ALLOW_FP16 = 'BUZZDETECT_GPU_FP16'

# Reduced precision is a different file, not a different provider option: an
# fp32 graph handed to the Neural Engine is the NeuralNetwork path above, which
# is both inaccurate and, since the fusion, slow. The export writes this sibling
# beside model.onnx with the trunk and head in fp16 and the front end left in
# fp32.
FNAME_FP16 = 'model.fp16.onnx'

# What reduced precision buys, and costs, on the fused graph (Apple M1, 200 s,
# COREML.md):
#
#   MLProgram + CPUAndGPU, fp32      71 ms   4.3e-06 on the predictions
#   MLProgram + ALL,       fp16      38 ms   1.7e-02, and one frame in 209
#                                            changes its top class
#
# 1.9x. The trunk on its own is 3.3x; the front end stays in fp32 and becomes
# most of what is left. MLComputeUnits=ALL is what reaches the Neural Engine --
# CPUAndGPU with an fp16 graph gains nothing, because Metal was already running
# fp32 at full rate.
COREML_FP16 = {'ModelFormat': 'MLProgram', 'MLComputeUnits': 'ALL'}


def allow_fp16():
    return os.environ.get(ENV_ALLOW_FP16) == '1'


def fp16_supported(provider):
    """Whether reduced precision does anything for this provider.

    Only CoreML, for now. The CUDA and ROCm providers take precision from the
    model's own dtype, so an fp16 graph would run in fp16 on them too -- but
    nobody has measured whether that is a win there, and on the one card it was
    tried (GTX 1650) fp16 ran at half speed.
    """
    return provider == 'CoreMLExecutionProvider'


def gpu_providers_available():
    """GPU execution providers this onnxruntime installation actually offers."""
    available = ort.get_available_providers()
    return [name for name, _ in GPU_PROVIDERS if name in available]


_warned = set()


def _warn_once(key, message):
    if key in _warned:
        return
    _warned.add(key)
    # stderr rather than the worker log: sessions are built inside model
    # plugins, which have no handle on the coordinator's logger. The desktop
    # app surfaces engine stderr in its log pane either way.
    print(f'WARNING: {message}', file=sys.stderr, flush=True)


def providers_for(processor):
    """(providers, options) to request for a worker running on `processor`."""
    if processor != 'GPU':
        return ['CPUExecutionProvider'], [{}]

    available = ort.get_available_providers()
    for name, options in GPU_PROVIDERS:
        if name not in available:
            continue
        options = dict(options)
        if allow_fp16() and fp16_supported(name):
            options = dict(COREML_FP16)
        return [name, 'CPUExecutionProvider'], [options, {}]

    _warn_once(
        'no-gpu-provider',
        'GPU processing was requested, but this onnxruntime installation has no '
        f'GPU execution provider (it offers: {", ".join(available)}). Install '
        'onnxruntime-gpu for CUDA. Running on CPU instead.'
    )
    return ['CPUExecutionProvider'], [{}]


def path_for(path_onnx, processor):
    """Which of a model's graphs to load: the fp32 one, or its fp16 sibling.

    Reduced precision only means anything where a provider can act on it, so a
    CPU worker gets the fp32 graph whatever the environment says.
    """
    if not (allow_fp16() and processor == 'GPU'):
        return path_onnx
    requested, _ = providers_for(processor)
    if not fp16_supported(requested[0]):
        return path_onnx

    path_fp16 = os.path.join(os.path.dirname(path_onnx), FNAME_FP16)
    if not os.path.exists(path_fp16):
        _warn_once(
            'no-fp16-graph',
            f'{ENV_ALLOW_FP16}=1 was set, but {os.path.basename(os.path.dirname(path_onnx))} '
            f'has no {FNAME_FP16}. Re-export it with buzzdetect-training to get one. '
            'Running at full precision.'
        )
        return path_onnx

    _warn_once(
        'fp16',
        f'{requested[0]} is running in reduced precision (fp16) by request. '
        'Predictions shift by ~2e-2 against the fp32 reference, so results are '
        'not comparable with fp32 output at the margins.'
    )
    return path_fp16


def make_session(path_onnx, processor, samples):
    """Build an InferenceSession pinned to a fixed input length.

    `samples` fixes the graph's one free dimension. onnxruntime does this
    itself, given the dimension's name, so the graph on disk stays general and
    nothing here has to rewrite it -- which also keeps the `onnx` package out
    of the shipped sidecar.

    Complains if it didn't get the GPU it asked for.
    """
    requested, options = providers_for(processor)

    so = ort.SessionOptions()
    so.add_free_dimension_override_by_name(DIM_SAMPLES, samples)
    session = ort.InferenceSession(path_for(path_onnx, processor), so,
                                   providers=requested, provider_options=options)

    if processor == 'GPU' and requested[0] != 'CPUExecutionProvider':
        got = session.get_providers()
        if requested[0] not in got:
            _warn_once(
                'gpu-not-loaded',
                f'{requested[0]} was requested but onnxruntime did not load it '
                f'(active providers: {", ".join(got)}). This usually means the '
                'CUDA/cuDNN runtime it was built against is missing. Running on CPU.'
            )

    return session


# Input length the probe pins its session to. One 0.96 s frame -- short enough
# that building and running the session is quick, long enough to be a shape the
# graph actually accepts.
PROBE_SAMPLES = 16000


def _probe_model():
    """The ONNX file to build the probe session on."""
    from src import config as cfg

    found = sorted(glob.glob(os.path.join(cfg.DIR_MODELS, '*', 'model.onnx')))
    return found[0] if found else None


def _run_probe(session):
    """Execute the probe session once on zeros.

    Building a session is not proof that it runs. A provider initialises, takes
    the graph, reports itself in get_providers(), and then fails on the first
    kernel -- which is what a mixed cuDNN install does here: cuDNN 9 dispatches
    to sub-libraries by soname, so a loader path carrying two 9.x installs (a
    pip nvidia-cudnn-cu12 wheel alongside a system one, say) can pair a core
    from one with a sub-library from the other, and the handshake fails with
    CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH the first time a convolution runs.
    Nothing before that first run says anything is wrong.

    So the probe runs. Without this the desktop app offers a GPU that dies a
    chunk into the analysis, taking its analyzer thread with it.
    """
    info = session.get_inputs()[0]
    shape = [d if isinstance(d, int) else PROBE_SAMPLES for d in info.shape]
    session.run(None, {info.name: np.zeros(shape, dtype=np.float32)})


def probe_gpu():
    """GPU execution providers this machine can actually run.

    get_available_providers() answers a different question -- which providers
    this onnxruntime build was compiled with -- and answers it identically on a
    workstation with a full CUDA install and on a laptop with no NVIDIA
    hardware at all. A provider's shared libraries aren't loaded until a
    session asks for it, so the only honest test is to build one, run it, and
    see which provider survives both; onnxruntime falls back to CPU rather than
    raising when the libraries turn out to be missing, and raises rather than
    falling back when they are present but mismatched.

    Costs a real session creation and one inference, so it's worth doing once
    and remembering. The length the probe pins the input to is arbitrary, but
    it has to be pinned, or CoreML will decline a graph it would accept in an
    analysis.
    """
    path_onnx = _probe_model()
    if path_onnx is None:
        return []

    usable = []
    for name, options in GPU_PROVIDERS:
        if name not in ort.get_available_providers():
            continue
        try:
            so = ort.SessionOptions()
            so.add_free_dimension_override_by_name(DIM_SAMPLES, PROBE_SAMPLES)
            session = ort.InferenceSession(
                path_onnx, so,
                providers=[name, 'CPUExecutionProvider'],
                provider_options=[dict(options), {}],
            )
        except Exception:
            # A provider that can't initialise at all -- no driver, no device,
            # a CUDA/cuDNN too old for this build. Not usable, and not fatal.
            continue
        if name not in session.get_providers():
            continue
        try:
            _run_probe(session)
        except Exception as e:
            # Took the graph and then couldn't execute it. Report it: the user
            # has the hardware and something about the install is stopping them
            # using it, which is worth more than a silent drop to CPU.
            print(f'WARNING: {name} loaded but could not run this machine\'s '
                  f'GPU probe, so it is not offered: {type(e).__name__}: {e}',
                  file=sys.stderr, flush=True)
            continue
        usable.append(name)
    return usable
