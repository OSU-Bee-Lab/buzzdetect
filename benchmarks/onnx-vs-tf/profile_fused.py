"""Where does the fused ONNX graph actually spend its time, per node?

Guessing from op inventory got us a hypothesis (dense-matmul DFT instead of an
FFT); onnxruntime's own profiler settles it.
"""
import argparse, collections, glob, json, os, sys
import numpy as np
import onnx, onnxruntime as ort

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--seconds', type=float, default=200.0)
ap.add_argument('--runs', type=int, default=8)
args = ap.parse_args()

# --- what shape are the MatMul weights? A DFT-as-matmul is unmistakable. ---
m = onnx.load(args.model)
inits = {i.name: list(i.dims) for i in m.graph.initializer}
print('=== initializers feeding MatMul / Conv (first 12 by size) ===')
big = sorted(inits.items(), key=lambda kv: -np.prod(kv[1] or [1]))[:12]
for name, dims in big:
    print(f'  {name:45s} {dims}')

by_op = collections.defaultdict(list)
for n in m.graph.node:
    for inp in n.input:
        if inp in inits:
            by_op[n.op_type].append((n.name, inits[inp]))
print('\n=== MatMul operands ===')
for name, dims in by_op.get('MatMul', []):
    print(f'  {name or "(unnamed)"}: weight {dims}')

# --- per-node timing on CUDA ---
so = ort.SessionOptions()
so.enable_profiling = True
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(args.model, sess_options=so,
                            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
print('\nproviders:', sess.get_providers())

name_in = sess.get_inputs()[0].name
audio = np.zeros(int(args.seconds * 16000), dtype=np.float32)
for _ in range(3):
    sess.run(None, {name_in: audio})
for _ in range(args.runs):
    sess.run(None, {name_in: audio})
path = sess.end_profiling()

events = json.load(open(path))
node = collections.defaultdict(lambda: [0.0, 0, None, None])
for e in events:
    if e.get('cat') != 'Node' or not e['name'].endswith('_kernel_time'):
        continue
    a = e.get('args', {})
    key = e['name'][:-len('_kernel_time')]
    rec = node[key]
    rec[0] += e['dur'] / 1000.0     # us -> ms
    rec[1] += 1
    rec[2] = a.get('op_name', rec[2])
    rec[3] = a.get('provider', rec[3])

total = sum(v[0] for v in node.values())
print(f'\n=== per-node kernel time over {args.runs} runs (total {total:.1f} ms, '
      f'{total/args.runs:.1f} ms/run) ===')
print(f'{"node":45s} {"op":18s} {"provider":28s} {"ms/run":>8s} {"%":>6s}')
for k, v in sorted(node.items(), key=lambda kv: -kv[1][0])[:20]:
    print(f'{k[:45]:45s} {str(v[2])[:18]:18s} {str(v[3])[:28]:28s} '
          f'{v[0]/args.runs:8.2f} {100*v[0]/total:5.1f}%')

print('\n=== grouped by op type ===')
byop = collections.defaultdict(float)
byprov = collections.defaultdict(float)
for k, v in node.items():
    byop[v[2]] += v[0]
    byprov[v[3]] += v[0]
for k, t in sorted(byop.items(), key=lambda kv: -kv[1]):
    print(f'  {str(k):22s} {t/args.runs:8.2f} ms/run  {100*t/total:5.1f}%')
print('\n=== grouped by execution provider ===')
for k, t in sorted(byprov.items(), key=lambda kv: -kv[1]):
    print(f'  {str(k):30s} {t/args.runs:8.2f} ms/run  {100*t/total:5.1f}%')
os.remove(path)
