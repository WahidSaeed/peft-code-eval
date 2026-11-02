#!/usr/bin/env bash
# DIAGNOSTIC ONLY (RESEARCH_LOG #93): how much does the description comment in HumanEval-Java prompts change
# the scores of the untuned model (M0) vs a Megadiff-fine-tuned adapter (10k pilot M2, epoch 3)?
# Three HumanEval-Java generations (20 samples, T = 0.2, max_new_tokens 1,024), written to results/pilot/:
#   M2_pilotA10_nodesc   pilot adapter, prompt without description (= the Megadiff training format)
#   M0_pilotA10_nodesc   untuned model, prompt without description
#   M2_pilotA10_tok1024  pilot adapter, prompt with description, new token limit (checks the truncation fix)
# M0 with description is the existing M0_pilotA10 run (512 tokens; only 6 of 3,260 samples hit the limit).
# Not a thesis result (pilot adapter, A10). Every step is skipped if its output exists. Score afterwards in
# the CPU `score` job with execution.py.
set -euo pipefail
cd "$(dirname "$0")/.."
preds=results/pilot/predictions
adapter=results/pilot/runs/M2_pilotA10/adapter
[[ -f "$adapter/adapter_model.safetensors" ]] || { echo "missing $adapter"; exit 1; }

gen() {  # gen <run_name> <evaluator args...>
  local name="$1"; shift
  if [[ -f "$preds/${name}_humaneval_java.jsonl" ]]; then echo "== $name exists, skipping"; return; fi
  echo "== $name ($(date -Is))"
  python src/evaluator.py "$@" --run_name "$name" --benchmark humaneval_java --latency_samples 0 \
    --output_dir "$preds" --log_csv results/pilot/eval_log.csv
}
gen M2_pilotA10_nodesc --bits 4 --adapter "$adapter" --no_description
gen M0_pilotA10_nodesc --bits 16 --no_description
gen M2_pilotA10_tok1024 --bits 4 --adapter "$adapter"
echo "== done ($(date -Is))"
