#!/usr/bin/env bash
# Everything, in order. Stops at the first failed GPU check -- a CPU-bound run
# looks completely normal in the logs.
set -euo pipefail

ENGINE=/home/luke/projects/buzzdetect/engine
PY="$ENGINE/.venv-bench/bin/python"
SP="$(cd "$(dirname "$0")" && pwd)"
RESULTS=/home/luke/projects/buzzdetect/local/bench
AUDIO="/media/server storage/experiments/Luke - Diel Drivers/2026-08-18/1_16/260818_0613.mp3"

# Force the venv's own NVIDIA wheels to win the dynamic loader. This box also
# carries a system cuDNN (libcudnn9-cuda-12, 9.25) that ldconfig finds, while
# the venv wheel is 9.24, and onnxruntime ends up resolving cuDNN's
# sub-libraries from both -- CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH, after
# which it falls back to CPU and reports a plausible-looking number that has
# nothing to do with the GPU. Same trick start_analysis uses for the
# bundled-CUDA build. Do not drop this.
NVLIB=$(ls -d "$ENGINE"/.venv-bench/lib/python3.12/site-packages/nvidia/*/lib \
        | xargs -I{} readlink -f {} | tr '\n' ':')
export LD_LIBRARY_PATH="$NVLIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$RESULTS"
cd "$ENGINE"

echo "############ 1. environment ############"
"$PY" -c "import sys, onnxruntime, tensorflow as tf; print('python', sys.version.split()[0]); print('onnxruntime', onnxruntime.__version__); print('tensorflow', tf.__version__)" 2>&1 | grep -Ev "^20[0-9]{2}-|oneDNN|TF-TRT|cuFFT|cuDNN|cuBLAS|computation placer|external/local"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

echo
echo "############ 2. prove both runtimes reach the GPU ############"
"$PY" buzzdetect_cli.py --probe_gpu | tee "$RESULTS/probe_gpu.json"
grep -q CUDAExecutionProvider "$RESULTS/probe_gpu.json" || { echo "FAIL: no CUDAExecutionProvider"; exit 1; }
"$PY" -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print('TF GPUs:', gpus)
raise SystemExit(0 if gpus else 'FAIL: TensorFlow sees no GPU')
" 2>&1 | grep -Ev "^20[0-9]{2}-|oneDNN|TF-TRT|cuFFT|cuDNN|cuBLAS|computation placer|external/local"

echo
echo "############ 3. fused graph op inventory (for the CoreML question) ############"
"$PY" "$SP/inspect_onnx.py" --model models/model_general_v3_onnx/model_combined.onnx \
      --out "$RESULTS/fused_ops.json" || echo "(skipped: $?)"

echo
echo "############ 4. inference microbenchmark, three arms ############"
for spec in "onnx:onnx:model_general_v3" "tensorflow:tensorflow:model_general_v3" "onnx_fused:auto:model_general_v3_onnx"; do
  IFS=: read -r label rt modelname <<< "$spec"
  echo "---- $label ($modelname, runtime=$rt) ----"
  env BUZZDETECT_RUNTIME="$rt" "$PY" "$SP/bench_inference.py" \
      --audio "$AUDIO" --modelname "$modelname" \
      --out "$RESULTS/bench_${label}.json" 2>&1 \
    | grep -Ev "^20[0-9]{2}-|oneDNN|TF-TRT|cuFFT|cuDNN|cuBLAS|computation placer|external/local"
done

echo
echo "############ 5. full-precision parity, three-way ############"
for pair in "onnx:tensorflow" "onnx_fused:tensorflow" "onnx_fused:onnx"; do
  IFS=: read -r a b <<< "$pair"
  echo "---- $a vs $b ----"
  "$PY" "$SP/compare_npy.py" --a "$RESULTS/bench_${a}_predictions.npy" \
                             --b "$RESULTS/bench_${b}_predictions.npy"
done

echo
echo "############ 6. end-to-end runs ############"
CORPUS=/home/luke/projects/buzzdetect/local/bench/corpus OUTBASE="$RESULTS" "$SP/run_endtoend.sh"

echo
echo "############ 7. CSV parity (coarse; CSVs are rounded to 2 digits) ############"
"$PY" "$SP/compare_parity.py" --a "$RESULTS/onnx-1" --b "$RESULTS/tensorflow-1" --limit 10 || true
"$PY" "$SP/compare_parity.py" --a "$RESULTS/onnx_fused-1" --b "$RESULTS/tensorflow-1" --limit 10 || true
