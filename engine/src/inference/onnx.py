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

# GPU providers worth trying, best first, with the options each needs.
# onnxruntime falls back along the list per-operator, so CPU is always
# appended as the backstop.
#
# CoreML is pinned to ModelFormat=MLProgram deliberately. Measured on this
# model (YAMNet trunk, 209 patches, Apple silicon):
#
#   CPU                                213 ms   reference
#   CoreML, default NeuralNetwork       14 ms   embeddings off by 2.9e-2
#   CoreML, MLProgram + CPUAndGPU       50 ms   embeddings off by 5.6e-6
#
# The default format is quicker still because it runs fp16 on the Neural
# Engine, but 2.9e-2 is three hundred times our float32 parity budget -- results
# would depend on which machine analysed them. MLProgram keeps fp32, runs on
# the GPU via Metal, and stays within the same tolerance as the CPU-versus-
# TensorFlow comparison the models were validated against, for a 4.3x win.
GPU_PROVIDERS = (
    ('CUDAExecutionProvider', {}),
    ('ROCMExecutionProvider', {}),
    ('CoreMLExecutionProvider', {'ModelFormat': 'MLProgram', 'MLComputeUnits': 'CPUAndGPU'}),
)


def gpu_providers_available():
    """GPU execution providers this onnxruntime installation actually offers."""
    available = ort.get_available_providers()
    return [name for name, _ in GPU_PROVIDERS if name in available]


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
    """(providers, options) to request for a worker running on `processor`."""
    if processor != 'GPU':
        return ['CPUExecutionProvider'], [{}]

    available = ort.get_available_providers()
    for name, options in GPU_PROVIDERS:
        if name in available:
            return [name, 'CPUExecutionProvider'], [dict(options), {}]

    _warn_once(
        'GPU processing was requested, but this onnxruntime installation has no '
        f'GPU execution provider (it offers: {", ".join(available)}). Install '
        'onnxruntime-gpu for CUDA. Running on CPU instead.'
    )
    return ['CPUExecutionProvider'], [{}]


def make_session(path_onnx, processor):
    """Build an InferenceSession, and complain if it didn't get the GPU it asked for."""
    requested, options = providers_for(processor)
    session = ort.InferenceSession(path_onnx, providers=requested, provider_options=options)

    if processor == 'GPU' and requested[0] != 'CPUExecutionProvider':
        got = session.get_providers()
        if requested[0] not in got:
            _warn_once(
                f'{requested[0]} was requested but onnxruntime did not load it '
                f'(active providers: {", ".join(got)}). This usually means the '
                'CUDA/cuDNN runtime it was built against is missing. Running on CPU.'
            )

    return session
