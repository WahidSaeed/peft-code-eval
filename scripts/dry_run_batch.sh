#!/usr/bin/env bash
# Batch-size test for the dry run (RESEARCH_LOG #46): same effective batch 16 as the plan
# (4 x 4), run as 8 x 2. Measures speed and memory for 4-bit (M1) and the 16-bit memory worst case (M9).
set -uo pipefail
cd "$(dirname "$0")/.."
cap=(${MAX_VRAM_GIB:+--max_vram_gib "$MAX_VRAM_GIB"})
for spec in "M1 4 8" "M9 16 32"; do
  read -r cell bits rank <<<"$spec"
  out="results/runs/dry_bs8_${cell}"
  [[ -f "$out/adapter/adapter_model.safetensors" ]] && { echo "== $cell bs8: exists, skipping"; continue; }
  echo "== $cell bs8: bits=$bits rank=$rank batch 8 x grad-accum 2 ($(date -Is))"
  mkdir -p "$out"
  python src/trainer.py --bits "$bits" --rank "$rank" --run_name "${cell}_bs8" --output_dir "$out" --max_steps 50 \
    --per_device_train_batch_size 8 --gradient_accumulation_steps 2 --log_csv results/dry_log.csv "${cap[@]}" \
    2>&1 | tee "$out/train.log"
done
