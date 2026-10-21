#!/usr/bin/env bash
# GPU dry run (RESEARCH_LOG #39 item 4): does max_length 768 fit, which batch sizes work, does logging work?
#   bash scripts/dry_run.sh                    # on a 24 GB GPU
#   MAX_VRAM_GIB=24 bash scripts/dry_run.sh    # on a larger GPU, emulating 24 GB
# Needs data/*.jsonl (src/data_loader.py) and, for the HumanEval-Java check, HEJ_JAVA_HOME (JDK 8).
set -uo pipefail
cd "$(dirname "$0")/.."
cap=(${MAX_VRAM_GIB:+--max_vram_gib "$MAX_VRAM_GIB"})
mkdir -p results
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee results/dry_gpu.txt

# Training: M1 (the plan's dry run) + the r = 32 cell of each bit-width (memory worst case per precision)
for c in M1 M3 M6 M9; do DRY_RUN=50 bash scripts/run_matrix.sh "$c"; done

# Inference: 20-sample batches on the longest prompts, greedy Megadiff, latency probe
ev=(--latency_samples 3 --output_dir results/dry_predictions --log_csv results/dry_eval_log.csv "${cap[@]}")
python src/evaluator.py --bits 16 --run_name dryM0 --benchmark humanevalfix --limit 4 "${ev[@]}"
[[ -d results/runs/dry_M1/adapter ]] && python src/evaluator.py --bits 4 --adapter results/runs/dry_M1/adapter \
  --run_name dryM1 --benchmark humaneval_java --limit 4 "${ev[@]}"
[[ -d results/runs/dry_M9/adapter ]] && python src/evaluator.py --bits 16 --adapter results/runs/dry_M9/adapter \
  --run_name dryM9 --benchmark megadiff --limit 40 --score "${ev[@]}"
python src/execution.py results/dry_predictions/dryM0_humanevalfix_java.jsonl \
  results/dry_predictions/dryM1_humaneval_java.jsonl --per_sample_out results/dry_predictions/dry_tests.jsonl
echo "== dry run finished: results/dry_log.csv, results/dry_eval_log.csv"
