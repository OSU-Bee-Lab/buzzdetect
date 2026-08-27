#!/usr/bin/env bash
# End-to-end runs: three arms, two repeats each, a separate --dir_out for every
# one. stream/worker.py:62 skips files whose results already exist, so a reused
# directory would make a run finish instantly and look like a total victory.
set -euo pipefail

ENGINE=/home/luke/projects/buzzdetect/engine
PY="$ENGINE/.venv-bench/bin/python"
CORPUS="${CORPUS:?set CORPUS to the symlink corpus directory}"
OUTBASE="${OUTBASE:-/home/luke/projects/buzzdetect/local/bench}"
STREAMERS="${STREAMERS:-12}"

# label : BUZZDETECT_RUNTIME : modelname
ARMS=(
  "onnx:onnx:model_general_v3"
  "tensorflow:tensorflow:model_general_v3"
  "onnx_fused:auto:model_general_v3_onnx"
)

cd "$ENGINE"

for rep in 1 2; do
  for spec in "${ARMS[@]}"; do
    IFS=: read -r label rt modelname <<< "$spec"
    out="$OUTBASE/${label}-${rep}"
    echo "=== $label repeat $rep ($modelname, runtime=$rt) -> $out ==="
    rm -rf "$out"; mkdir -p "$out"
    env BUZZDETECT_RUNTIME="$rt" "$PY" buzzdetect_cli.py \
        --modelname "$modelname" \
        --dir_audio "$CORPUS" \
        --dir_out "$out" \
        --analyzers_gpu 1 --analyzers_cpu 0 \
        --n_streamers "$STREAMERS" \
        --chunklength 200 \
        --framehop_prop 1 \
        --verbosity_print PROGRESS --verbosity_log DEBUG --log_progress true \
        < /dev/null 2>&1 | tail -4
  done
done

echo
echo "=== summary ==="
printf '%-14s %-6s %-22s %-14s %s\n' arm rep "total analysis time" "median rate" "BOTTLENECK lines"
for rep in 1 2; do
  for spec in "${ARMS[@]}"; do
    IFS=: read -r label rt modelname <<< "$spec"
    out="$OUTBASE/${label}-${rep}"
    log=$(ls -t "$out"/*.log 2>/dev/null | head -1) || true
    if [ -z "${log:-}" ]; then printf '%-14s %-6s %s\n' "$label" "$rep" "NO LOG"; continue; fi
    total=$(grep -o "Total analysis time: [0-9.,]*" "$log" | tail -1 | sed 's/Total analysis time: //')
    bottle=$(grep -c "BUFFER BOTTLENECK" "$log" || true)
    # Median of per-chunk rates, dropping the first 5 (cold cache, cuDNN autotune).
    median=$(grep -o "rate: [0-9.]*" "$log" | awk '{print $2}' | tail -n +6 | sort -n \
             | awk '{a[NR]=$1} END {if (NR) printf "%.1f", (NR%2 ? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2)}')
    printf '%-14s %-6s %-22s %-14s %s\n' "$label" "$rep" "${total:-n/a}s" "${median:-n/a}" "$bottle"
  done
done
