#!/usr/bin/env bash
# Train M1–M9 sequentially (one GPU). Usage:
#   bash scripts/run_matrix.sh            # all nine
#   bash scripts/run_matrix.sh M1 M5      # selected cells
#   DRY_RUN=50 bash scripts/run_matrix.sh M1   # 50-step dry run into results/runs/dry_<cell>
# A cell whose adapter already exists is skipped, so the script can be re-run after a crash.
set -euo pipefail
(( BASH_VERSINFO[0] >= 4 )) || { echo "needs bash >= 4 (associative arrays)"; exit 1; }
cd "$(dirname "$0")/.."

declare -A BITS=( [M1]=4 [M2]=4 [M3]=4 [M4]=8 [M5]=8 [M6]=8 [M7]=16 [M8]=16 [M9]=16 )
declare -A RANK=( [M1]=8 [M2]=16 [M3]=32 [M4]=8 [M5]=16 [M6]=32 [M7]=8 [M8]=16 [M9]=32 )
CELLS=("${@:-M1 M2 M3 M4 M5 M6 M7 M8 M9}")
CELLS=(${CELLS[@]})

for cell in "${CELLS[@]}"; do
  if [[ -n "${DRY_RUN:-}" ]]; then
    out="results/runs/dry_${cell}"; extra=(--max_steps "$DRY_RUN" --log_csv results/dry_log.csv)
  else
    out="results/runs/${cell}"; extra=()
  fi
  if [[ -f "$out/adapter/adapter_model.safetensors" ]]; then
    echo "== $cell: adapter exists in $out, skipping"; continue
  fi
  echo "== $cell: bits=${BITS[$cell]} rank=${RANK[$cell]} -> $out  ($(date -Is))"
  mkdir -p "$out"
  python src/trainer.py --bits "${BITS[$cell]}" --rank "${RANK[$cell]}" --run_name "$cell" \
    --output_dir "$out" "${extra[@]}" 2>&1 | tee "$out/train.log"
done
