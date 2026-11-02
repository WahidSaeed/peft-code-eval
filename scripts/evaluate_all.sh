#!/usr/bin/env bash
# Generate predictions for M0–M9 on the Megadiff test split (greedy, scored inline: EM / CodeBLEU / syntax)
# and on HumanEvalFix-Java + HumanEval-Java (20 samples, T = 0.2), then score the test-based benchmarks
# (pass@1, pass@10) by running the tests. Run on the GPU VM: execution.py runs model-written code.
#   bash scripts/evaluate_all.sh            # all ten models
#   bash scripts/evaluate_all.sh M0 M2      # selected models
#   bash scripts/evaluate_all.sh M2_s43 M2_s44 M5_s43 M5_s44 M8_s43 M8_s44   # extra-seed adapters
# M0 is the untuned base model, zero-shot (research decision, RESEARCH_LOG #27).
# HumanEval-Java needs JDK 8: export HEJ_JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
# Optional: MAX_VRAM_GIB=24 caps PyTorch GPU memory (same value as for training)
set -euo pipefail
[[ -n "${HEJ_JAVA_HOME:-}" ]] || { echo "set HEJ_JAVA_HOME to a JDK 8 (HumanEval-Java)"; exit 1; }
(( BASH_VERSINFO[0] >= 4 )) || { echo "needs bash >= 4 (associative arrays)"; exit 1; }
cd "$(dirname "$0")/.."

declare -A BITS=( [M0]=16 [M1]=4 [M2]=4 [M3]=4 [M4]=8 [M5]=8 [M6]=8 [M7]=16 [M8]=16 [M9]=16 )
MODELS=("${@:-M0 M1 M2 M3 M4 M5 M6 M7 M8 M9}")
MODELS=(${MODELS[@]})
mkdir -p results/predictions results/scores

evaluate() {  # evaluate <run_name> <bits> [--adapter DIR]
  local name="$1" bits="$2"; shift 2
  echo "== $name: Megadiff test ($(date -Is))"
  python src/evaluator.py --bits "$bits" "$@" --run_name "$name" ${MAX_VRAM_GIB:+--max_vram_gib "$MAX_VRAM_GIB"} --benchmark megadiff --score
  for b in humanevalfix humaneval_java; do
    echo "== $name: $b ($(date -Is))"
    python src/evaluator.py --bits "$bits" "$@" --run_name "$name" ${MAX_VRAM_GIB:+--max_vram_gib "$MAX_VRAM_GIB"} --benchmark "$b"
  done
  python src/execution.py results/predictions/"${name}"_humanevalfix_java.jsonl results/predictions/"${name}"_humaneval_java.jsonl \
    --per_sample_out "results/scores/${name}_tests_per_sample.jsonl" > "results/scores/${name}_tests.json"
  cat "results/scores/${name}_tests.json"
}

# Checkpoint selection (RESEARCH_LOG #91): every cell is trained for all 3 epochs. The primary result <m> uses the
# epoch with the lowest validation loss (results/runs/<m>/best_epoch.json, written by the trainer). If that is not
# the last epoch, the final adapter is evaluated as well, as <m>_final (secondary result).
for m in "${MODELS[@]}"; do
  cell="${m%%_s*}"   # M2_s43 -> M2
  if [[ "$m" == "M0" ]]; then evaluate M0 "${BITS[$cell]}"; continue; fi
  run="results/runs/$m"
  [[ -d "$run/adapter" && -f "$run/best_epoch.json" ]] || { echo "== $m: no adapter or best_epoch.json, skipping"; continue; }
  best=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['best_epoch'])" "$run/best_epoch.json")
  last=$(python -c "import json,sys; print(max(map(int, json.load(open(sys.argv[1]))['val_loss_per_epoch'])))" "$run/best_epoch.json")
  [[ -d "$run/adapter-epoch$best" ]] || { echo "== $m: adapter-epoch$best missing, skipping"; continue; }
  echo "== $m: best epoch $best of $last (lowest validation loss)"
  evaluate "$m" "${BITS[$cell]}" --adapter "$run/adapter-epoch$best"
  if [[ "$best" != "$last" ]]; then
    evaluate "${m}_final" "${BITS[$cell]}" --adapter "$run/adapter"
  fi
done
