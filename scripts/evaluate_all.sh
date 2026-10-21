#!/usr/bin/env bash
# Generate predictions for M0–M9 on the Megadiff test split (greedy, scored inline: EM / CodeBLEU / syntax)
# and on HumanEvalFix-Java + HumanEval-Java (20 samples, T = 0.2), then score the test-based benchmarks
# (pass@1, pass@10) by running the tests. Run on the GPU VM: execution.py runs model-written code.
#   bash scripts/evaluate_all.sh            # all ten models
#   bash scripts/evaluate_all.sh M0 M2      # selected models
#   bash scripts/evaluate_all.sh M2_s43 M2_s44 M5_s43 M5_s44 M8_s43 M8_s44   # extra-seed adapters
# M0 is the untuned base model, zero-shot (research decision, RESEARCH_LOG #27).
# HumanEval-Java needs JDK 8: export HEJ_JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
set -euo pipefail
[[ -n "${HEJ_JAVA_HOME:-}" ]] || { echo "set HEJ_JAVA_HOME to a JDK 8 (HumanEval-Java)"; exit 1; }
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
  echo "== $m: Megadiff test ($(date -Is))"
  python src/evaluator.py --bits "${BITS[$cell]}" "${adapter[@]}" --run_name "$m" --benchmark megadiff --score
  for b in humanevalfix humaneval_java; do
    echo "== $m: $b ($(date -Is))"
    python src/evaluator.py --bits "${BITS[$cell]}" "${adapter[@]}" --run_name "$m" --benchmark "$b"
  done
  python src/execution.py results/predictions/"${m}"_humanevalfix_java.jsonl results/predictions/"${m}"_humaneval_java.jsonl \
    --per_sample_out "results/scores/${m}_tests_per_sample.jsonl" > "results/scores/${m}_tests.json"
  cat "results/scores/${m}_tests.json"
done
