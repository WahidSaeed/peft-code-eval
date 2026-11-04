#!/usr/bin/env bash
# M609 module paper (branch m609-memory-study): peak GPU memory of LoRA training and inference for
# DeepSeek-Coder-6.7B-Instruct at 4-bit (NF4), 8-bit (LLM.int8()) and 16-bit (bf16) x LoRA rank 8/16/32.
# One job, one GPU model (A10G), one commit, so all cells are measured under identical conditions.
#   bash scripts/m609_memory_grid.sh            # everything
#   bash scripts/m609_memory_grid.sh M1 M5      # selected cells (inference for M0 always runs)
# Re-runnable: finished parts are skipped. Outputs in results/m609/.
#   1. Training, typical: 50 optimizer steps, seed 42, full training set (identical batches in every cell).
#   2. Training, worst case: the 80 longest training sequences (all near the 768-token cap), 5 steps.
#   3. Inference, worst case: the longest prompt of HumanEvalFix-Java and of HumanEval-Java, 20 samples in
#      chunks of 10, every sample forced to max_new_tokens (largest possible KV cache). M0 + the 9 adapters
#      from part 1 (adapter quality is irrelevant for memory; only its size matters).
# Validation is off (--no_eval): the peak covers training steps only.
set -uo pipefail
(( BASH_VERSINFO[0] >= 4 )) || { echo "needs bash >= 4"; exit 1; }
cd "$(dirname "$0")/.."

declare -A BITS=( [M1]=4 [M2]=4 [M3]=4 [M4]=8 [M5]=8 [M6]=8 [M7]=16 [M8]=16 [M9]=16 )
declare -A RANK=( [M1]=8 [M2]=16 [M3]=32 [M4]=8 [M5]=16 [M6]=32 [M7]=8 [M8]=16 [M9]=32 )
CELLS=("${@:-M1 M2 M3 M4 M5 M6 M7 M8 M9}"); CELLS=(${CELLS[@]})
OUT=results/m609
mkdir -p "$OUT/runs" "$OUT/predictions"
nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit --format=csv | tee -a "$OUT/gpu.txt"
echo "${M609_COMMIT:-unknown}" | tee -a "$OUT/commit.txt"   # the container has no .git; the launcher passes the hash

train() {  # train <cell> <tag> <extra trainer args...>
  local cell="$1" tag="$2"; shift 2
  local out="$OUT/runs/${tag}_${cell}"
  [[ -f "$out/adapter/adapter_model.safetensors" ]] && { echo "== ${tag}_${cell}: done, skipping"; return; }
  echo "== ${tag}_${cell}: bits=${BITS[$cell]} rank=${RANK[$cell]} ($(date -Is))"
  mkdir -p "$out"
  python src/trainer.py --bits "${BITS[$cell]}" --rank "${RANK[$cell]}" --seed 42 --run_name "${tag}_${cell}" \
    --output_dir "$out" --no_eval --no_resume --log_csv "$OUT/train_${tag}.csv" "$@" 2>&1 | tee "$out/train.log"
}

for c in "${CELLS[@]}"; do train "$c" typical --max_steps 50; done
for c in "${CELLS[@]}"; do train "$c" worstcase --longest_n 80 --max_steps 5; done

infer() {  # infer <name> <bits> [--adapter DIR]
  local name="$1" bits="$2"; shift 2
  for b in humanevalfix humaneval_java; do
    local f="$OUT/predictions/${name}_$([[ $b == humanevalfix ]] && echo humanevalfix_java || echo $b).jsonl"
    [[ -f "$f" ]] && { echo "== $name $b: done, skipping"; continue; }
    echo "== $name $b: worst-case inference ($(date -Is))"
    python src/evaluator.py --bits "$bits" "$@" --run_name "$name" --benchmark "$b" --longest_n 1 \
      --force_max_new_tokens --latency_samples 0 --output_dir "$OUT/predictions" --log_csv "$OUT/infer_worstcase.csv"
  done
}
infer M0 16
for c in "${CELLS[@]}"; do
  a="$OUT/runs/typical_${c}/adapter"
  [[ -d "$a" ]] && infer "$c" "${BITS[$c]}" --adapter "$a" || echo "== $c: no adapter, skipping inference"
done
echo "== M609 memory grid finished ($(date -Is)): $OUT/train_typical.csv, train_worstcase.csv, infer_worstcase.csv"
