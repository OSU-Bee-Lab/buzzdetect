"""Op inventory for an ONNX graph.

Op coverage is what decides whether CoreML can take a graph in one piece or has
to partition it and bounce tensors to the CPU mid-graph, so the op list is the
evidence for whether fusing the front end would help or hurt on macOS.
"""
import argparse, collections, json

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--out', required=True)
args = ap.parse_args()

import onnx
m = onnx.load(args.model)
counts = collections.Counter(n.op_type for n in m.graph.node)
report = {
    'opsets': {(x.domain or 'ai.onnx'): x.version for x in m.opset_import},
    'n_nodes': sum(counts.values()),
    'inputs': [(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim])
               for i in m.graph.input],
    'outputs': [(o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim])
                for o in m.graph.output],
    'ops': dict(sorted(counts.items(), key=lambda kv: -kv[1])),
}
json.dump(report, open(args.out, 'w'), indent=2)
print(f"opsets {report['opsets']}  nodes {report['n_nodes']}")
print('ops:', report['ops'])
