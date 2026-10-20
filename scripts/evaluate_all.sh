#!/usr/bin/env bash
# Generate predictions for M0–M9 on CodeXGLUE (scored inline) and HumanEvalFix (6 languages),
# then score HumanEvalFix by running the tests. Run on the GPU VM: execution.py runs model-written code.
#   bash scripts/evaluate_all.sh            # all ten models
#   bash scripts/evaluate_all.sh M0 M2      # selected models
#   bash scripts/evaluate_all.sh M2_s43 M2_s44 M5_s43 M5_s44 M8_s43 M8_s44   # extra-seed adapters
# M0 is the untuned base model, zero-shot (research decision, RESEARCH_LOG #27).
set -euo pipefail
(( BASH_VERSINFO[0] >= 4 )) || { echo "needs bash >= 4 (associative arrays)"; exit 1; }
cd "$(dirname "$0")/.."

declare -A BITS=( [M0]=16 [M1]=4 [M2]=4 [M3]=4 [M4]=8 [M5]=8 [M6]=8 [M7]=16 [M8]=16 [M9]=16 )
MODELS=("${@:-M0 M1 M2 M3 M4 M5 M6 M7 M8 M9}")
MODELS=(${MODELS[@]})
mkdir -p results/predictions results/scores

for m in "${MODELS[@]}"; do
  cell="${m%%_s*}"   # M2_s43 -> M2
  adapter=()
  if [[ "$m" != "M0" ]]; then
    adapter=(--adapter "results/runs/$m/adapter")
    [[ -d "results/runs/$m/adapter" ]] || { echo "== $m: no adapter, skipping"; continue; }
  fi
  echo "== $m: CodeXGLUE ($(date -Is))"
  python src/evaluator.py --bits "${BITS[$cell]}" "${adapter[@]}" --run_name "$m" --benchmark codexglue --score
  echo "== $m: HumanEvalFix ($(date -Is))"
  python src/evaluator.py --bits "${BITS[$cell]}" "${adapter[@]}" --run_name "$m" --benchmark humanevalfix
  python src/execution.py results/predictions/"${m}"_humanevalfix_*.jsonl \
    --per_sample_out "results/scores/${m}_humanevalfix_per_sample.jsonl" > "results/scores/${m}_humanevalfix.json"
  cat "results/scores/${m}_humanevalfix.json"
done
