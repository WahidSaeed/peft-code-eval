#!/usr/bin/env bash
# Generate predictions for M0–M9 on the Megadiff test split (greedy, scored inline: EM / CodeBLEU / syntax)
# and on HumanEvalFix-Java + HumanEval-Java (20 samples at T = 0.2 → pass@1; for primary results a second
# 20 samples at T = 0.8 → pass@10, RESEARCH_LOG #93), then score the test-based benchmarks by running the
# tests. Primary results also get HumanEval-Java without the description comment (20 samples at T = 0.2,
# secondary result, RESEARCH_LOG #96). Run on the GPU VM: execution.py runs model-written code.
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

score_tests() {  # score_tests <run_name> [prediction suffixes]: unit tests → results/scores/<run_name>_tests.json
  local name="$1"; shift
  local suffixes=("${@:-humanevalfix_java humaneval_java}"); suffixes=(${suffixes[@]})
  local files=(); for s in "${suffixes[@]}"; do files+=("results/predictions/${name}_${s}.jsonl"); done
  python src/execution.py "${files[@]}" \
    --per_sample_out "results/scores/${name}_tests_per_sample.jsonl" > "results/scores/${name}_tests.json"
  cat "results/scores/${name}_tests.json"
}

evaluate() {  # evaluate <run_name> <bits> <primary|secondary> [--adapter DIR]
  local name="$1" bits="$2" role="$3"; shift 3
  echo "== $name: Megadiff test ($(date -Is))"
  python src/evaluator.py --bits "$bits" "$@" --run_name "$name" ${MAX_VRAM_GIB:+--max_vram_gib "$MAX_VRAM_GIB"} --benchmark megadiff --score
  for b in humanevalfix humaneval_java; do
    echo "== $name: $b, T = 0.2 ($(date -Is))"
    python src/evaluator.py --bits "$bits" "$@" --run_name "$name" ${MAX_VRAM_GIB:+--max_vram_gib "$MAX_VRAM_GIB"} --benchmark "$b"
  done
  score_tests "$name"   # pass@1 (primary metric)
  # pass@10 from a separate run at T = 0.8 (Chen et al., 2021; RESEARCH_LOG #93), primary results only.
  # The batch-size-1 latency probe is greedy and identical to the T = 0.2 run, so it is skipped here.
  [[ "$role" == primary ]] || return 0
  for b in humanevalfix humaneval_java; do
    echo "== ${name}_t08: $b, T = 0.8 ($(date -Is))"
    python src/evaluator.py --bits "$bits" "$@" --run_name "${name}_t08" ${MAX_VRAM_GIB:+--max_vram_gib "$MAX_VRAM_GIB"} \
      --benchmark "$b" --temperature 0.8 --latency_samples 0
  done
  score_tests "${name}_t08"   # pass@10 (secondary metric)
  # HumanEval-Java without the description comment (RESEARCH_LOG #96): the prompt is then exactly the Megadiff
  # training format and the format of the closest LoRA-APR papers. Secondary result, primary models only,
  # same decoding as the main run (n = 20, T = 0.2, 1,024 tokens); latency probe skipped.
  echo "== ${name}_nodesc: humaneval_java without description, T = 0.2 ($(date -Is))"
  python src/evaluator.py --bits "$bits" "$@" --run_name "${name}_nodesc" ${MAX_VRAM_GIB:+--max_vram_gib "$MAX_VRAM_GIB"} \
    --benchmark humaneval_java --no_description --latency_samples 0
  score_tests "${name}_nodesc" humaneval_java
}

# Checkpoint selection (RESEARCH_LOG #91): every cell is trained for all 3 epochs. The primary result <m> uses the
# epoch with the lowest validation loss (results/runs/<m>/best_epoch.json, written by the trainer). If that is not
# the last epoch, the final adapter is evaluated as well, as <m>_final (secondary result; T = 0.2 only,
# with descriptions only).
for m in "${MODELS[@]}"; do
  cell="${m%%_s*}"   # M2_s43 -> M2
  if [[ "$m" == "M0" ]]; then evaluate M0 "${BITS[$cell]}" primary; continue; fi
  run="results/runs/$m"
  [[ -d "$run/adapter" && -f "$run/best_epoch.json" ]] || { echo "== $m: no adapter or best_epoch.json, skipping"; continue; }
  best=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['best_epoch'])" "$run/best_epoch.json")
  last=$(python -c "import json,sys; print(max(map(int, json.load(open(sys.argv[1]))['val_loss_per_epoch'])))" "$run/best_epoch.json")
  [[ -d "$run/adapter-epoch$best" ]] || { echo "== $m: adapter-epoch$best missing, skipping"; continue; }
  echo "== $m: best epoch $best of $last (lowest validation loss)"
  evaluate "$m" "${BITS[$cell]}" primary --adapter "$run/adapter-epoch$best"
  if [[ "$best" != "$last" ]]; then
    evaluate "${m}_final" "${BITS[$cell]}" secondary --adapter "$run/adapter"   # T = 0.2 only
  fi
done
