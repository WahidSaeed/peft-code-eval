#!/usr/bin/env bash
# Full-length PILOT of one matrix cell on a non-official GPU (RESEARCH_LOG #73): full training (3 epochs,
# seed 42) + generation on all three benchmarks, written to results/pilot/. Checks whether Megadiff
# fine-tuning helps or hurts the test-based benchmarks before the official campaign. Not an official result
# (other GPU model than the official runs); unit tests are scored afterwards in the CPU `score` job.
#   bash scripts/pilot_cell.sh M2
# Every step is skipped if its output exists, so a retried container resumes (training from its checkpoint).
set -euo pipefail
(( BASH_VERSINFO[0] >= 4 )) || { echo "needs bash >= 4 (associative arrays)"; exit 1; }
cd "$(dirname "$0")/.."

declare -A BITS=( [M1]=4 [M2]=4 [M3]=4 [M4]=8 [M5]=8 [M6]=8 [M7]=16 [M8]=16 [M9]=16 )
declare -A RANK=( [M1]=8 [M2]=16 [M3]=32 [M4]=8 [M5]=16 [M6]=32 [M7]=8 [M8]=16 [M9]=32 )
cell="${1:?usage: pilot_cell.sh <cell>}"
name="${cell}_pilotA10"
out="results/pilot/runs/$name"
preds=results/pilot/predictions
mkdir -p "$out" "$preds"

if [[ -f "$out/adapter/adapter_model.safetensors" ]]; then
  echo "== $name: adapter exists, skipping training"
else
  echo "== $name: training bits=${BITS[$cell]} rank=${RANK[$cell]} ($(date -Is))"
  python src/trainer.py --bits "${BITS[$cell]}" --rank "${RANK[$cell]}" --run_name "$name" \
    --output_dir "$out" --log_csv results/pilot/train_log.csv 2>&1 | tee -a "$out/train.log"
fi

ev=(--bits "${BITS[$cell]}" --adapter "$out/adapter" --run_name "$name" --output_dir "$preds"
    --log_csv results/pilot/eval_log.csv)
for b in megadiff humanevalfix humaneval_java; do
  suffix="$b"; [[ "$b" == humanevalfix ]] && suffix=humanevalfix_java
  if [[ -f "$preds/${name}_${suffix}.jsonl" ]]; then echo "== $name: $b exists, skipping"; continue; fi
  echo "== $name: $b ($(date -Is))"
  score=(); [[ "$b" == megadiff ]] && score=(--score)
  python src/evaluator.py "${ev[@]}" --benchmark "$b" "${score[@]}"
done
echo "== $name: done ($(date -Is)); score with: python src/execution.py $preds/${name}_humanevalfix_java.jsonl $preds/${name}_humaneval_java.jsonl"
