"""Fold the depthwise batchnorm into the conv weights, and measure.

bisect_trunk.py's honest accounting puts 9.77 ms in 27 Relu nodes and 3.62 ms
in 13 Add nodes -- 32% of the trunk in elementwise passes over 40 MB tensors.
They are there because tf2onnx emitted each depthwise layer's batchnorm as a
standalone Mul+Add.  The 13 *pointwise* convs already arrive folded (weights
plus a bias, no Mul/Add), so this is an export artifact, not something
inherent.

Sitting between the Conv and its Relu, that Mul/Add pair also blocks
onnxruntime's Conv+Activation fusion, so every depthwise layer costs four
kernel launches and four round trips through memory instead of one.

Folding is exact:  ((X*W) * s + b)  ==  X * (W*s) + b, for per-channel s.
Nothing here touches engine code -- it rewrites a copy of the build artifact.
"""
import os, statistics, sys, time
import numpy as np, onnx, onnxruntime as ort
from onnx import numpy_helper

sys.path.insert(0, os.getcwd())
SRC = 'embedders/yamnet_onnx/yamnet.onnx'
DST = '/tmp/yamnet_folded.onnx'
N = 209


def fold(src, dst):
    m = onnx.load(src)
    g = m.graph
    init = {i.name: i for i in g.initializer}
    arr = {k: numpy_helper.to_array(v) for k, v in init.items()}
    prod = {o: n for n in g.node for o in n.output}
    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)

    drop, folded = set(), 0
    for mul in list(g.node):
        if mul.op_type != 'Mul' or len(mul.output) != 1:
            continue
        conv = prod.get(mul.input[0])
        scale_name = mul.input[1]
        if conv is None or conv.op_type != 'Conv' or scale_name not in arr:
            continue
        if len(consumers.get(conv.output[0], [])) != 1:
            continue
        add = consumers.get(mul.output[0], [None])[0]
        if add is None or add.op_type != 'Add' or len(consumers[mul.output[0]]) != 1:
            continue
        shift_name = add.input[1]
        if shift_name not in arr:
            continue

        w_name = conv.input[1]
        w = arr[w_name]
        s = arr[scale_name].reshape(-1)
        b = arr[shift_name].reshape(-1)
        if s.shape[0] != w.shape[0] or b.shape[0] != w.shape[0]:
            continue

        w_new = w * s.reshape(-1, 1, 1, 1)
        init[w_name].CopyFrom(numpy_helper.from_array(w_new.astype(np.float32), w_name))
        bias_name = w_name + '_folded_bias'
        g.initializer.append(numpy_helper.from_array(b.astype(np.float32), bias_name))
        if len(conv.input) >= 3:
            conv.input[2] = bias_name
        else:
            conv.input.append(bias_name)
        # the conv now produces what the Add produced; Mul and Add go away
        conv.output[0] = add.output[0]
        drop.add(id(mul)); drop.add(id(add))
        folded += 1

    for n in [n for n in g.node if id(n) in drop]:
        g.node.remove(n)
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    return folded


def session(path):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=so,
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


n = fold(SRC, DST)
print(f'folded {n} depthwise batchnorms into their convolutions\n')

before = onnx.load(SRC).graph.node
after = onnx.load(DST).graph.node
for tag, nodes in (('as exported', before), ('folded', after)):
    ops = {}
    for x in nodes:
        ops[x.op_type] = ops.get(x.op_type, 0) + 1
    print(f'  {tag:14s} {len(nodes):3d} nodes  ' +
          ' '.join(f'{k}={v}' for k, v in sorted(ops.items(), key=lambda kv: -kv[1])[:6]))

s0, s1 = session(SRC), session(DST)
rng = np.random.default_rng(0)
x = rng.standard_normal((N, 96, 64), dtype=np.float32)
a = s0.run(None, {s0.get_inputs()[0].name: x})
b_ = s1.run(None, {s1.get_inputs()[0].name: x})
print('\nparity (full precision):')
for i, (u, v) in enumerate(zip(a, b_)):
    print(f'  output {i} {str(u.shape):16s} max abs diff {np.abs(u - v).max():.3e}'
          f'   max rel {np.abs(u - v).max() / max(1e-9, np.abs(u).max()):.3e}')

t0, t1 = timeit(s0), timeit(s1)
print(f'\n  as exported {t0:8.2f} ms')
print(f'  folded      {t1:8.2f} ms   {t0 / t1:.2f}x')
print(f'  (TensorFlow whole embed, from RESULTS.md: 52.6 ms incl. front end)')
