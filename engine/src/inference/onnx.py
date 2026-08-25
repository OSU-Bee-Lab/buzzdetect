"""Execution-provider selection for the ONNX models.

onnxruntime picks up a GPU only if the installed package actually carries a GPU
execution provider -- the plain `onnxruntime` wheel does not, `onnxruntime-gpu`
does. Getting that wrong is quiet by default: a session built with
CPUExecutionProvider runs perfectly happily at a fraction of the speed, with
nothing in the logs to say the GPU worker never touched the GPU. This module
exists so that failure is loud instead.
"""

import sys

import onnxruntime as ort

# GPU providers worth trying, best first. onnxruntime falls back along the list
# per-operator, so CPU is always appended as the backstop.
#
# CoreMLExecutionProvider is deliberately absent even though the macOS wheels
# offer it: it can run the graph in fp16 on the Neural Engine, which would make
# results depend on which machine and which worker type produced them. The
# whole point of exporting these models was that they agree with the
# TensorFlow originals to within float32 rounding, so a provider that quietly
# changes precision isn't a trade worth making.
GPU_PROVIDERS = (
    'CUDAExecutionProvider',
    'ROCMExecutionProvider',
)

_warned = False


def _warn_once(message):
    global _warned
    if _warned:
        return
    _warned = True
    # stderr rather than the worker log: sessions are built inside embedder and
    # model plugins, which have no handle on the coordinator's logger. The
    # desktop app surfaces engine stderr in its log pane either way.
    print(f'WARNING: {message}', file=sys.stderr, flush=True)


def providers_for(processor):
    """Provider list to request for a worker running on `processor`."""
    if processor != 'GPU':
        return ['CPUExecutionProvider']

    available = ort.get_available_providers()
    for name in GPU_PROVIDERS:
        if name in available:
            return [name, 'CPUExecutionProvider']

    _warn_once(
        'GPU processing was requested, but this onnxruntime installation has no '
        f'GPU execution provider (it offers: {", ".join(available)}). Install '
        'onnxruntime-gpu for CUDA. Running on CPU instead.'
    )
    return ['CPUExecutionProvider']


def make_session(path_onnx, processor):
    """Build an InferenceSession, and complain if it didn't get the GPU it asked for."""
    requested = providers_for(processor)
    session = ort.InferenceSession(path_onnx, providers=requested)

    if processor == 'GPU' and requested[0] != 'CPUExecutionProvider':
        got = session.get_providers()
        if requested[0] not in got:
            _warn_once(
                f'{requested[0]} was requested but onnxruntime did not load it '
                f'(active providers: {", ".join(got)}). This usually means the '
                'CUDA/cuDNN runtime it was built against is missing. Running on CPU.'
            )

    return session
