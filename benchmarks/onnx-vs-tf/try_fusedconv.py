"""Fuse each Conv+Relu into com.microsoft.FusedConv and measure.

sum_of_parts.py accounts for the trunk honestly: 27 Conv = 32.3 ms, 27 Relu =
10.3 ms, summing to the 41 ms whole.  The convolutions are irreducible, but the
Relus are 27 standalone passes that each re-read and re-write the whole
activation tensor -- layer 2's alone moves 82 MB to apply a max().  Folding
them into the preceding convolution is the only structural saving left.

onnxruntime carries a FusedConv contrib op with an `activation` attribute.
Its ConvActivationFusion pass is not registered for the CUDA EP, so the graph
keeps the separate Relus; this asks whether the kernel would take them if the
graph handed them over already fused.
"""
import os, statistics, sys, time
import numpy as np, onnx, onnxruntime as ort

sys.path.insert(0, os.getcwd())
SRC = 'embedders/yamnet_onnx/yamnet.onnx'
OPT = '/tmp/yamnet_opt2.onnx'
FUS = '/tmp/yamnet_fusedconv.onnx'
N = 209

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.optimized_model_filepath = OPT
ort.InferenceSession(SRC, sess_options=so, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

m = onnx.load(OPT)
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
m.opset_import.append(onnx.helper.make_opsetid('com.microsoft', 1))
onnx.save(m, FUS)
print(f'fused {fused} of 27 Conv+Relu pairs into com.microsoft.FusedConv\n')


def session(path):
    o = ort.SessionOptions()
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=o,
                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])


def timeit(sess, runs=30):
    name = sess.get_inputs()[0].name
    x = ort.OrtValue.ortvalue_from_numpy(np.zeros((N, 96, 64), dtype=np.float32), 'cuda', 0)
    b = sess.io_binding()
    b.bind_ortvalue_input(name, x)
    for o in sess.get_outputs():
        b.bind_output(o.name, 'cuda', 0)
    for _ in range(10):
        sess.run_with_iobinding(b)
    b.synchronize_outputs()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run_with_iobinding(b)
        b.synchronize_outputs()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000


s0 = session(SRC)
try:
    s1 = session(FUS)
except Exception as e:
    print('FusedConv session failed:', str(e)[:300])
    sys.exit(0)

rng = np.random.default_rng(0)
x = rng.standard_normal((N, 96, 64), dtype=np.float32)
a = s0.run(None, {s0.get_inputs()[0].name: x})[0]
b_ = s1.run(None, {s1.get_inputs()[0].name: x})[0]
print(f'parity: max abs diff {np.abs(a - b_).max():.3e}  '
      f'max rel {np.abs(a - b_).max() / np.abs(a).max():.3e}')

t0, t1 = timeit(s0), timeit(s1)
print(f'\n  Conv + separate Relu   {t0:8.2f} ms')
print(f'  FusedConv              {t1:8.2f} ms   {t0 / t1:.2f}x')
print(f'  (sum_of_parts.py puts the 27 Relus at 10.34 ms of the {t0:.0f} ms)')
