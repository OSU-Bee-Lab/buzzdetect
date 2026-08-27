# ONNX vs TensorFlow on CUDA

A measurement, not product code. Nothing here is imported by the app or the
engine; the scripts call into `engine/` from outside and modify none of it.

`PLAN.md` is the brief — what is being compared, what invalidates a result, and
what to do with the answer. Read it first. `RESULTS.md` is the answer, once
there is one.

## Why it exists

`model_general_v3` ships as both an ONNX graph and a TensorFlow SavedModel, and
the desktop app carries only the ONNX half. If ONNX is meaningfully slower on
CUDA, that argues for letting power users bring their own TensorFlow. Three
arms, because the ONNX and TensorFlow paths differ in *where the log-mel front
end runs*, not just in runtime:

| arm | model | front end |
|---|---|---|
| `tensorflow` | `model_general_v3` | Keras graph, GPU |
| `onnx` | `model_general_v3` | NumPy, **CPU** |
| `onnx_fused` | `model_general_v3_onnx` | ONNX graph, GPU |

## Running it

Needs a venv with both runtimes, which no requirements file in the repo
provides -- see `PLAN.md` §1. Then:

    ./run_all.sh

Roughly an hour, most of it the six end-to-end runs. Outputs land in
`local/bench/`, which is gitignored: JSON per arm, `.npy` predictions for the
parity check, and one output directory per run. Commit `RESULTS.md`, not the
outputs.

`run_all.sh` hard-fails if either runtime misses the GPU. Don't work around
that -- a CPU-bound run looks completely normal in the logs, which is the main
way this measurement goes quietly wrong.

## Branch

Lives on `bench/onnx-vs-tf`, off `gui-overhaul` -- `main` predates the move to
`engine/` and has no `DualRuntimeModel`, so the benchmark cannot run there.
