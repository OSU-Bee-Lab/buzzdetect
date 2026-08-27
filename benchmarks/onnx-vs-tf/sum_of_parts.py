"""Time every node of the optimized trunk standalone, and see if they add up.

Two earlier attributions both proved untrustworthy:
  - profile_trunk.py's kernel times sum to 54.9 ms against a 41 ms wall clock
    (profiling serialises the stream and charges queue wait to the node);
  - bisect_trunk.py's prefix marginals are contaminated because onnxruntime
    re-optimizes each prefix differently -- fold_depthwise_bn.py showed it
    already folds Mul/Add into Conv, so a prefix cut between them is a
    different graph, not a smaller one.

This takes ORT's *own* optimized graph, rebuilds each node as a standalone
model on its real shapes, and times it device-resident.  If the sum matches the
41 ms whole, the trunk is simply the sum of honest parts and there is no
pathology to find.  If it falls well short, the missing time is per-node
overhead in graph execution.
"""
import os, statistics, sys, time
import numpy as np, onnx, onnxruntime as ort
from onnx import helper, numpy_helper, TensorProto

sys.path.insert(0, os.getcwd())
SRC = 'embedders/yamnet_onnx/yamnet.onnx'
OPT = '/tmp/yamnet_opt.onnx'
N = 209

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.optimized_model_filepath = OPT
_ = ort.InferenceSession(SRC, sess_options=so,
                         providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

m = onnx.load(OPT)
# resolve every tensor's shape with the batch dim pinned
m.graph.input[0].type.tensor_type.shape.dim[0].dim_value = N
m = onnx.shape_inference.infer_shapes(m)
shapes = {}
for vi in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output):
    d = vi.type.tensor_type.shape.dim
    shapes[vi.name] = [x.dim_value if x.HasField('dim_value') else N for x in d]
init = {i.name: i for i in m.graph.initializer}

ops = {}
for x in m.graph.node:
    ops[x.op_type] = ops.get(x.op_type, 0) + 1
print("ORT's optimized graph: " + ' '.join(f'{k}={v}' for k, v in sorted(ops.items(), key=lambda kv: -kv[1])))
print()


def timeit(model_bytes, ishape, iname, runs=40):
    s = ort.InferenceSession(model_bytes, providers=['CUDAExecutionProvider'])
    x = ort.OrtValue.ortvalue_from_numpy(np.random.rand(*ishape).astype(np.float32), 'cuda', 0)
    b = s.io_binding()
    b.bind_ortvalue_input(iname, x)
    for o in s.get_outputs():
        b.bind_output(o.name, 'cuda', 0)
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


tally, rows = {}, []
for n in m.graph.node:
    xname = n.input[0]
    if xname in init or xname not in shapes:
        continue
    ishape = shapes[xname]
    if not ishape or any(d <= 0 for d in ishape):
        continue
    keep = [i for i in n.input if i in init]
    node = helper.make_node(n.op_type, list(n.input), [n.output[0]], **{
        a.name: onnx.helper.get_attribute_value(a) for a in n.attribute})
    node.domain = n.domain
    g = helper.make_graph(
        [node], 'one',
        [helper.make_tensor_value_info(xname, TensorProto.FLOAT, ishape)],
        [helper.make_tensor_value_info(n.output[0], TensorProto.FLOAT, None)],
        [init[k] for k in keep])
    mm = helper.make_model(g, opset_imports=list(m.opset_import))
    try:
        t = timeit(mm.SerializeToString(), ishape, xname)
    except Exception as e:
        rows.append((n.name, n.op_type, ishape, None, str(e)[:40]))
        continue
    rows.append((n.name, n.op_type, ishape, t, ''))
    tally[n.op_type] = tally.get(n.op_type, 0.0) + t

print(f'{"node":46s} {"op":12s} {"input shape":22s} {"ms":>8s}')
for name, op, ishape, t, err in sorted(rows, key=lambda r: -(r[3] or 0)):
    if t is None:
        print(f'{name[:46]:46s} {op:12s} {str(ishape):22s}   skipped: {err}')
    else:
        print(f'{name[:46]:46s} {op:12s} {str(ishape):22s} {t:8.2f}')

print(f'\n--- standalone cost by op ---')
for op, t in sorted(tally.items(), key=lambda kv: -kv[1]):
    print(f'  {op:20s} {ops.get(op, 0):3d} nodes {t:8.2f} ms')
print(f'  {"SUM OF PARTS":20s} {" ":10s} {sum(tally.values()):8.2f} ms')
print(f'  {"WHOLE TRUNK":20s} {" ":10s} {41.0:8.2f} ms')
