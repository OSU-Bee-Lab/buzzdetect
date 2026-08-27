"""Isolate the two pathological convolutions into minimal ONNX graphs.

ONNX_TUNE.md step 4: separate "onnxruntime chooses badly for this shape" from
"this shape is simply bad on Turing".  Each variant perturbs exactly one
property of the real layer, so a variant that is fast names the culprit.

Inputs are bound device-resident via IOBinding.  Without that, every timing is
just the host-to-device copy of an 80 MB tensor -- layer 3, which costs 0.76 ms
inside the real graph, appears to take 20 ms.  Outputs stay on the GPU and are
synchronised explicitly, so the numbers are kernel time, not a D2H copy.

The trunk control at the bottom is the calibration: the same harness must
reproduce the trunk's known ~42 ms wall clock, or none of the rest is credible.
"""
import os, statistics, sys, time
import numpy as np, onnx, onnxruntime as ort
from onnx import helper, numpy_helper, TensorProto

OUT = '/tmp/isolate'
os.makedirs(OUT, exist_ok=True)
N = 209


def conv_model(path, xshape, w, strides, pads, group):
    W = numpy_helper.from_array(w.astype(np.float32), 'W')
    node = helper.make_node('Conv', ['X', 'W'], ['Y'], kernel_shape=list(w.shape[2:]),
                            strides=list(strides), pads=list(pads), group=group,
                            dilations=[1, 1])
    hi, wi = xshape[2:]
    ho = (hi + pads[0] + pads[2] - w.shape[2]) // strides[0] + 1
    wo = (wi + pads[1] + pads[3] - w.shape[3]) // strides[1] + 1
    g = helper.make_graph([node], 'c',
                          [helper.make_tensor_value_info('X', TensorProto.FLOAT, list(xshape))],
                          [helper.make_tensor_value_info('Y', TensorProto.FLOAT,
                                                         [xshape[0], w.shape[0], ho, wo])],
                          [W])
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid('', 17)]), path)
    return path


def bench(path, xshape, opts=None, runs=50):
    """Median wall time per run with input and output both pinned on the GPU."""
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    s = ort.InferenceSession(path, sess_options=so, providers=['CUDAExecutionProvider'],
                             provider_options=[opts or {}])
    x = ort.OrtValue.ortvalue_from_numpy(np.random.rand(*xshape).astype(np.float32), 'cuda', 0)
    oshape = [d if isinstance(d, int) else 1 for d in s.get_outputs()[0].shape]
    y = ort.OrtValue.ortvalue_from_numpy(np.zeros(oshape, dtype=np.float32), 'cuda', 0)
    b = s.io_binding()
    b.bind_ortvalue_input('X', x)
    b.bind_ortvalue_output(s.get_outputs()[0].name, y)
    for _ in range(10):
        s.run_with_iobinding(b)
    b.synchronize_outputs()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        s.run_with_iobinding(b)
        b.synchronize_outputs()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000


def report(label, ms, gflop):
    print(f'  {label:44s} {ms:8.2f} ms   {gflop / (ms / 1000):7.1f} GFLOP/s')


print('=== layer 1: 3x3 s2, 1 -> 32 ch, on [209,1,96,64] ===')
gf1 = 2 * N * 32 * 48 * 32 * 9 / 1e9
w1 = np.random.rand(32, 1, 3, 3)
l1 = conv_model(f'{OUT}/l1.onnx', (N, 1, 96, 64), w1, (2, 2), (0, 0, 1, 1), 1)
report('as exported (pads 0,0,1,1 asymmetric)', bench(l1, (N, 1, 96, 64)), gf1)
report('symmetric pads (1,1,1,1)',
       bench(conv_model(f'{OUT}/l1sym.onnx', (N, 1, 96, 64), w1, (2, 2), (1, 1, 1, 1), 1), (N, 1, 96, 64)), gf1)
report('no pads (0,0,0,0)',
       bench(conv_model(f'{OUT}/l1np.onnx', (N, 1, 96, 64), w1, (2, 2), (0, 0, 0, 0), 1), (N, 1, 96, 64)), gf1)
report('prefer_nhwc, as exported', bench(l1, (N, 1, 96, 64), {'prefer_nhwc': '1'}), gf1)
report('EXHAUSTIVE, as exported', bench(l1, (N, 1, 96, 64), {'cudnn_conv_algo_search': 'EXHAUSTIVE'}), gf1)
report('3 input channels (else identical)',
       bench(conv_model(f'{OUT}/l1c3.onnx', (N, 3, 96, 64), np.random.rand(32, 3, 3, 3),
                        (2, 2), (0, 0, 1, 1), 1), (N, 3, 96, 64)), gf1 * 3)

print('\n=== layer 2: 3x3 s1 depthwise (group=32) on [209,32,48,32] ===')
gf2 = 2 * N * 32 * 48 * 32 * 9 / 1e9
w2 = np.random.rand(32, 1, 3, 3)
l2 = conv_model(f'{OUT}/l2.onnx', (N, 32, 48, 32), w2, (1, 1), (1, 1, 1, 1), 32)
report('as exported (group=32, depthwise)', bench(l2, (N, 32, 48, 32)), gf2)
report('prefer_nhwc', bench(l2, (N, 32, 48, 32), {'prefer_nhwc': '1'}), gf2)
report('EXHAUSTIVE', bench(l2, (N, 32, 48, 32), {'cudnn_conv_algo_search': 'EXHAUSTIVE'}), gf2)
report('dense group=1, same shape (32x the FLOPs)',
       bench(conv_model(f'{OUT}/l2d.onnx', (N, 32, 48, 32), np.random.rand(32, 32, 3, 3),
                        (1, 1), (1, 1, 1, 1), 1), (N, 32, 48, 32)), gf2 * 32)
report('group=4 (else identical)',
       bench(conv_model(f'{OUT}/l2g4.onnx', (N, 32, 48, 32), np.random.rand(32, 8, 3, 3),
                        (1, 1), (1, 1, 1, 1), 4), (N, 32, 48, 32)), gf2 * 8)

print('\n=== layer 3 depthwise (the fast one), for calibration ===')
gf3 = 2 * N * 64 * 24 * 16 * 9 / 1e9
report('group=64, s2, on [209,64,48,32]',
       bench(conv_model(f'{OUT}/l3.onnx', (N, 64, 48, 32), np.random.rand(64, 1, 3, 3),
                        (2, 2), (0, 0, 1, 1), 64), (N, 64, 48, 32)), gf3)


print('\n=== control: the whole trunk through this same harness ===')
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
s = ort.InferenceSession('embedders/yamnet_onnx/yamnet.onnx', sess_options=so,
                         providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
n_in = s.get_inputs()[0].name
x = ort.OrtValue.ortvalue_from_numpy(np.zeros((N, 96, 64), dtype=np.float32), 'cuda', 0)
b = s.io_binding()
b.bind_ortvalue_input(n_in, x)
for o in s.get_outputs():
    b.bind_output(o.name, 'cuda', 0)
for _ in range(10):
    s.run_with_iobinding(b)
b.synchronize_outputs()
ts = []
for _ in range(30):
    t0 = time.perf_counter()
    s.run_with_iobinding(b)
    b.synchronize_outputs()
    ts.append(time.perf_counter() - t0)
print(f'  trunk, input already on GPU: {statistics.median(ts) * 1000:8.2f} ms'
      f'   (profile_trunk.py wall clock: ~42 ms)')
