"""Where does the trunk's 41 ms actually go?

profile_trunk.py's per-node kernel times sum to 54.9 ms against a 41 ms wall
clock -- the parts exceed the whole, so those numbers include stall/wait and
cannot be trusted to attribute cost.  This walks the graph instead, extracting
a prefix ending at each node and timing it end to end.  The jump between
consecutive prefixes is that node's real marginal cost.
"""
import os, statistics, sys, time
import numpy as np, onnx, onnxruntime as ort
from onnx.utils import Extractor

sys.path.insert(0, os.getcwd())
MODEL = 'embedders/yamnet_onnx/yamnet.onnx'
N = 209

m = onnx.load(MODEL)
m = onnx.shape_inference.infer_shapes(m)
inp = m.graph.input[0].name
ex = Extractor(m)


def timeit(sess, runs=20):
    x = ort.OrtValue.ortvalue_from_numpy(np.zeros((N, 96, 64), dtype=np.float32), 'cuda', 0)
    b = sess.io_binding()
    b.bind_ortvalue_input(inp, x)
    for o in sess.get_outputs():
        b.bind_output(o.name, 'cuda', 0)
    for _ in range(8):
        sess.run_with_iobinding(b)
    b.synchronize_outputs()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run_with_iobinding(b)
        b.synchronize_outputs()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000


# every tensor produced by a real compute node, in topological order
cuts = []
for n in m.graph.node:
    if n.op_type in ('Shape', 'Gather', 'Unsqueeze', 'Concat', 'Cast', 'Constant', 'ConstantOfShape'):
        continue
    cuts.append((n.name or n.op_type, n.op_type, n.output[0]))

tally = {}
print(f'{len(cuts)} cut points; timing each prefix\n')
print(f'{"node":52s} {"op":18s} {"cumulative":>11s} {"marginal":>10s}')
prev = 0.0
for name, op, out in cuts:
    try:
        sub = ex.extract_model([inp], [out])
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        s = ort.InferenceSession(sub.SerializeToString(), sess_options=so,
                                 providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        t = timeit(s)
    except Exception as e:
        print(f'{name[:52]:52s} {op:18s}   skipped: {str(e)[:40]}')
        continue
    mark = '  <<<' if t - prev > 2.0 else ''
    print(f'{name[:52]:52s} {op:18s} {t:9.2f} ms {t - prev:9.2f} ms{mark}')
    tally.setdefault(op, [0.0, 0])
    tally[op][0] += max(0.0, t - prev)
    tally[op][1] += 1
    prev = t

print(f'\n--- marginal cost by op ---')
for op, (tot, cnt) in sorted(tally.items(), key=lambda kv: -kv[1][0]):
    print(f'  {op:20s} {cnt:3d} nodes {tot:8.2f} ms')
print(f'  {"TOTAL":20s}     {" ":6s} {sum(v[0] for v in tally.values()):8.2f} ms')
