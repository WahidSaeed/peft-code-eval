#!/usr/bin/env bash
# End-to-end check of the official pipeline on a small slice (RESEARCH_LOG #63, task 3).
# Exercises the full path an official cell takes: full-run training (not max_steps) with per-epoch
# validation and step checkpoints, an interruption + resume on the GPU, adapter saving, generation with
# that adapter on all three benchmarks, Megadiff scoring and unit-test scoring. Not an official result.
#   bash scripts/e2e_check.sh            # 4-bit r=8 (the cell type with paged-optimizer + 4-bit resume)
#   E2E_BITS=16 bash scripts/e2e_check.sh   # same check for 8- or 16-bit (outputs in results/e2e/b<bits>)
set -uo pipefail
cd "$(dirname "$0")/.."
BITS=${E2E_BITS:-4}
base=results/e2e/b$BITS; out=$base/M1
rm -rf "$base"
mkdir -p "$out"
train=(python src/trainer.py --bits "$BITS" --rank 8 --run_name "e2e_b$BITS" --output_dir "$out" --train_limit 256
       --eval_limit 32 --num_train_epochs 2 --save_steps 5 --log_csv "$base/train_log.csv")

echo "== 1. train, then kill once a complete checkpoint exists ($(date -Is))"
"${train[@]}" > "$out/train_part1.log" 2>&1 &
pid=$!
until ls "$out"/checkpoint-*/trainer_state.json >/dev/null 2>&1 || ! kill -0 $pid 2>/dev/null; do sleep 5; done
sleep 20  # let training continue past the checkpoint, so the resume has to rewind
kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
echo "   killed; complete checkpoints: $(ls -d "$out"/checkpoint-* 2>/dev/null | xargs -n1 basename | tr '\n' ' ')"

echo "== 2. resume to the end ($(date -Is))"
"${train[@]}" > "$out/train_part2.log" 2>&1
grep -E "resuming from|\"global_steps\"|\"resumed_from\"|eval_loss_per_eval" "$out/train_part2.log"
[[ -f "$out/adapter/adapter_model.safetensors" ]] || { echo "FAIL: no adapter"; exit 1; }

echo "== 3. generate with the adapter on all three benchmarks ($(date -Is))"
ev=(--bits "$BITS" --adapter "$out/adapter" --run_name "e2e_b$BITS" --limit 6 --latency_samples 2
    --output_dir "$base/predictions" --log_csv "$base/eval_log.csv")
python src/evaluator.py "${ev[@]}" --benchmark megadiff --score > "$base/eval_megadiff.log" 2>&1
python src/evaluator.py "${ev[@]}" --benchmark humanevalfix > "$base/eval_hef.log" 2>&1
python src/evaluator.py "${ev[@]}" --benchmark humaneval_java > "$base/eval_hej.log" 2>&1

echo "== 4. unit tests on the samples ($(date -Is))"
python src/execution.py "$base/predictions/e2e_b${BITS}_humanevalfix_java.jsonl" \
  "$base/predictions/e2e_b${BITS}_humaneval_java.jsonl" --workers "$(nproc)"
echo "== CSV rows"; cat "$base/train_log.csv"; echo; cut -d, -f1-12 "$base/eval_log.csv"
echo "== e2e finished ($(date -Is))"
