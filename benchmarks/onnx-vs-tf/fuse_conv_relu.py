"""Fuse Conv+Relu pairs in an ONNX graph into onnxruntime's FusedConv.

Standalone: needs only `onnx`. No TensorFlow, no onnxruntime, nothing from the
buzzdetect engine. Import `fuse_conv_relu`, or run it as a CLI:

    python fuse_conv_relu.py model.onnx model_fused.onnx

Why this exists: tf2onnx emits every Relu as its own node, and onnxruntime's
ConvActivationFusion pass is registered for the CPU EP but not the CUDA EP.
So on CUDA each Relu becomes a separate kernel that re-reads and re-writes the
entire activation tensor just to apply a max() -- on YAMNet at 209 frames,
27 of them, moving up to 82 MB apiece. Handing onnxruntime the pairs already
fused lets the convolution kernel apply the activation on data it is holding,
and the intermediate tensor is never written at all.

Measured on the buzzdetect fused graph (GTX 1650, 200 s of audio):
67.8 ms -> 49.1 ms end to end, bit-exact. Standalone trunk 41.7 -> 23.2 ms.

The rewrite is exact by construction: it changes which kernel applies the
max(), not the arithmetic. Verified as max abs diff 0.0, not merely within
tolerance.
"""

import argparse

import onnx


def fuse_conv_relu(model):
    """Rewrite every Conv->Relu pair into a single com.microsoft.FusedConv.

    Returns (model, n_fused). The model is modified in place and also returned.

    A pair is only fused when the Conv's output feeds *nothing but* that one
    Relu. If anything else reads it -- a skip connection, a second branch, a
    graph output -- fusing would delete a value that other node still needs.
    YAMNet has no such case, but MobileNet-v2/v3 and ResNet backbones do, so
    the check is not optional if this is ever pointed at another model.
    """
    graph = model.graph
    producer = {out: n for n in graph.node for out in n.output}
    consumers = {}
    for n in graph.node:
        for inp in n.input:
            consumers.setdefault(inp, []).append(n)

    graph_outputs = {o.name for o in graph.output}

    drop, n_fused = set(), 0
    for relu in list(graph.node):
        if relu.op_type != 'Relu':
            continue
        conv = producer.get(relu.input[0])
        if conv is None or conv.op_type != 'Conv':
            continue
        # the Conv's result must be private to this Relu
        if len(consumers.get(conv.output[0], [])) != 1:
            continue
        if conv.output[0] in graph_outputs:
            continue

        conv.op_type = 'FusedConv'
        conv.domain = 'com.microsoft'
        conv.attribute.append(onnx.helper.make_attribute('activation', 'Relu'))
        conv.output[0] = relu.output[0]
        drop.add(id(relu))
        n_fused += 1

    for n in [n for n in graph.node if id(n) in drop]:
        graph.node.remove(n)

    if n_fused and not any(o.domain == 'com.microsoft' for o in model.opset_import):
        model.opset_import.append(onnx.helper.make_opsetid('com.microsoft', 1))

    return model, n_fused


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src')
    ap.add_argument('dst')
    args = ap.parse_args()

    model = onnx.load(args.src)
    before = len(model.graph.node)
    model, n = fuse_conv_relu(model)
    onnx.checker.check_model(model)
    onnx.save(model, args.dst)
    print(f'{args.src}: {before} nodes -> {len(model.graph.node)} nodes, '
          f'{n} Conv+Relu pairs fused')
    print(f'wrote {args.dst}')
    if n == 0:
        print('\nNo pairs fused. Either the graph has no Conv->Relu with a '
              'private output,\nor onnxruntime already fused them at export.')


if __name__ == '__main__':
    main()
