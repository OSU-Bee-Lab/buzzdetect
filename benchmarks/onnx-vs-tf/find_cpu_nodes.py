"""Which nodes in the fused graph are NOT on CUDA, and what do the memcpys cost?

onnxruntime warns it inserts 13 Memcpy nodes for this graph. Each one is a
host<->device round trip in the middle of the pipeline.
"""
import collections, json, os, sys
import numpy as np, onnxruntime as ort
sys.path.insert(0, os.getcwd())

so = ort.SessionOptions()
so.enable_profiling = True
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
s = ort.InferenceSession('models/model_general_v3_onnx/model_combined.onnx',
                         sess_options=so,
                         providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
n_in = s.get_inputs()[0].name
audio = np.zeros(int(200.0*16000), dtype=np.float32)
RUNS = 8
for _ in range(3): s.run(None, {n_in: audio})
for _ in range(RUNS): s.run(None, {n_in: audio})
p = s.end_profiling()

rec = collections.defaultdict(lambda: [0.0, None, None])
for e in json.load(open(p)):
    if e.get('cat') == 'Node' and e['name'].endswith('_kernel_time'):
        a = e.get('args', {})
        k = e['name'][:-len('_kernel_time')]
        rec[k][0] += e['dur']/1000.0
        rec[k][1] = a.get('op_name')
        rec[k][2] = a.get('provider')

total = sum(v[0] for v in rec.values())
cpu = {k: v for k, v in rec.items() if v[2] and 'CPU' in v[2]}
mem = {k: v for k, v in rec.items() if v[1] and 'Memcpy' in str(v[1])}

print(f'total kernel time {total/RUNS:.2f} ms/run over {len(rec)} nodes\n')
print(f'=== nodes on CPUExecutionProvider: {len(cpu)} ===')
for k, v in sorted(cpu.items(), key=lambda kv: -kv[1][0])[:15]:
    print(f'  {k[:55]:55s} {str(v[1]):14s} {v[0]/RUNS:7.3f} ms')
print(f'  CPU subtotal: {sum(v[0] for v in cpu.values())/RUNS:.3f} ms/run '
      f'({100*sum(v[0] for v in cpu.values())/total:.1f}%)')

print(f'\n=== Memcpy nodes: {len(mem)} ===')
for k, v in sorted(mem.items(), key=lambda kv: -kv[1][0]):
    print(f'  {k[:55]:55s} {str(v[1]):14s} {v[0]/RUNS:7.3f} ms  [{v[2]}]')
print(f'  Memcpy subtotal: {sum(v[0] for v in mem.values())/RUNS:.3f} ms/run '
      f'({100*sum(v[0] for v in mem.values())/total:.1f}%)')

# Front end vs trunk: everything before the first Conv is the front end.
fe = sum(v[0] for k, v in rec.items() if v[1] not in ('Conv','Relu','GlobalAveragePool','Gemm'))
print(f'\nfront-end-ish ops (non Conv/Relu/Pool/Gemm): {fe/RUNS:.2f} ms/run ({100*fe/total:.1f}%)')
os.remove(p)
