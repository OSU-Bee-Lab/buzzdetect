"""End-to-end: does Conv+Relu fusion close the 1.29x gap to TensorFlow?

try_fusedconv.py takes the standalone trunk from 41.7 ms to 23.2 ms, bit-exact,
by handing onnxruntime its Conv+Relu pairs already fused as com.microsoft
FusedConv nodes.  This applies the same rewrite to model_combined.onnx -- the
fused graph NEXT_STEPS.md §1 proposes shipping -- and measures it the way
RESULTS.md §1 measured everything else: one 200 s chunk, real audio, input fed
from the host, median of 30 after 5 warmups.

Comparable numbers from RESULTS.md §1:  TensorFlow 52.6 ms, onnx_fused 67.7 ms.
"""
import os, statistics, sys, time
import numpy as np, onnx, onnxruntime as ort

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from bench_inference import load_chunk

SRC = 'models/model_general_v3_onnx/model_combined.onnx'
FUS = '/tmp/model_combined_fusedconv.onnx'
AUDIO = sys.argv[1]


def fuse_conv_relu(src, dst):
    """Rewrite every Conv whose only consumer is a Relu into a FusedConv."""
    m = onnx.load(src)
    g = m.graph
    prod = {o: n for n in g.node for o in n.output}
    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    drop, fused = set(), 0
    for relu in list(g.node):
        if relu.op_type != 'Relu':
            continue
        conv = prod.get(relu.input[0])
        if conv is None or conv.op_type != 'Conv':
            continue
        if len(consumers.get(conv.output[0], [])) != 1:
            continue
        conv.op_type = 'FusedConv'
        conv.domain = 'com.microsoft'
        conv.attribute.append(onnx.helper.make_attribute('activation', 'Relu'))
        conv.output[0] = relu.output[0]
        drop.add(id(relu))
        fused += 1
    for n in [n for n in g.node if id(n) in drop]:
        g.node.remove(n)
    if not any(o.domain == 'com.microsoft' for o in m.opset_import):
        m.opset_import.append(onnx.helper.make_opsetid('com.microsoft', 1))
    onnx.save(m, dst)
    return fused


def session(path):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=so,
                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])


def timeit(sess, audio, repeats=30, warmup=5):
    name = sess.get_inputs()[0].name
    for _ in range(warmup):
        sess.run(None, {name: audio})
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        r = sess.run(None, {name: audio})
        np.asarray(r[0])
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000


n = fuse_conv_relu(SRC, FUS)
print(f'fused {n} Conv+Relu pairs in model_combined.onnx\n')

audio = load_chunk(AUDIO, 200.0)
s0, s1 = session(SRC), session(FUS)
print('providers:', s1.get_providers())

a = s0.run(None, {s0.get_inputs()[0].name: audio})[0]
b = s1.run(None, {s1.get_inputs()[0].name: audio})[0]
print(f'parity vs unfused: max abs diff {np.abs(a - b).max():.3e}  shape {a.shape}')

t0 = timeit(s0, audio)
t1 = timeit(s1, audio)
print(f'\n  {"arm":34s} {"ms":>8s}  {"vs TF 52.6":>11s}')
print(f'  {"TensorFlow (RESULTS.md §1)":34s} {52.6:8.1f}  {1.00:10.2f}x')
print(f'  {"onnx_fused, as exported":34s} {t0:8.1f}  {t0 / 52.6:10.2f}x')
print(f'  {"onnx_fused + FusedConv":34s} {t1:8.1f}  {t1 / 52.6:10.2f}x')
print(f'\n  speedup from fusion: {t0 / t1:.2f}x')
