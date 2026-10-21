#!/usr/bin/env bash
# Train M1–M9 sequentially (one GPU). Usage:
#   bash scripts/run_matrix.sh            # all nine
#   bash scripts/run_matrix.sh M1 M5      # selected cells
#   DRY_RUN=50 bash scripts/run_matrix.sh M1   # 50-step dry run into results/runs/dry_<cell>
#   SEEDS="43 44" bash scripts/run_matrix.sh M2 M5 M8   # extra seeds -> results/runs/<cell>_s<seed>
#   MAX_VRAM_GIB=24 bash scripts/run_matrix.sh   # cap PyTorch memory, e.g. 24 GiB on a 48 GB card
# Main runs use seed 42 (results/runs/<cell>). The data split always uses seed 42, so every
# seed trains and tests on identical data; the training seed varies LoRA init, dropout and data order.
# A cell whose adapter already exists is skipped, so the script can be re-run after a crash.
set -euo pipefail
(( BASH_VERSINFO[0] >= 4 )) || { echo "needs bash >= 4 (associative arrays)"; exit 1; }
cd "$(dirname "$0")/.."

declare -A BITS=( [M1]=4 [M2]=4 [M3]=4 [M4]=8 [M5]=8 [M6]=8 [M7]=16 [M8]=16 [M9]=16 )
declare -A RANK=( [M1]=8 [M2]=16 [M3]=32 [M4]=8 [M5]=16 [M6]=32 [M7]=8 [M8]=16 [M9]=32 )
CELLS=("${@:-M1 M2 M3 M4 M5 M6 M7 M8 M9}")
CELLS=(${CELLS[@]})

SEED_LIST=(${SEEDS:-42})

for cell in "${CELLS[@]}"; do
  for seed in "${SEED_LIST[@]}"; do
    name="$cell"; [[ "$seed" == 42 ]] || name="${cell}_s${seed}"
    if [[ -n "${DRY_RUN:-}" ]]; then
      out="results/runs/dry_${name}"; extra=(--max_steps "$DRY_RUN" --log_csv results/dry_log.csv)
    else
      out="results/runs/${name}"; extra=()
    fi
    if [[ -f "$out/adapter/adapter_model.safetensors" ]]; then
      echo "== $name: adapter exists in $out, skipping"; continue
    fi
    echo "== $name: bits=${BITS[$cell]} rank=${RANK[$cell]} seed=$seed -> $out  ($(date -Is))"
    mkdir -p "$out"
    python src/trainer.py --bits "${BITS[$cell]}" --rank "${RANK[$cell]}" --seed "$seed" --run_name "$name" \
      --output_dir "$out" ${MAX_VRAM_GIB:+--max_vram_gib "$MAX_VRAM_GIB"} "${extra[@]}" 2>&1 | tee "$out/train.log"
  done
done
