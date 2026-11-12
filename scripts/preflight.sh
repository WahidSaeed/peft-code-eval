#!/usr/bin/env bash
# Pre-flight on one AWS g5.xlarge before the official campaign (step B of RESEARCH_LOG #100/#101).
# Called by scripts/card_job.sh when PREFLIGHT=1 (so the S3 sync / status / DONE machinery is exercised too).
# Nothing here is an official result; everything is written under results/ on a throwaway instance and synced
# to the <card>-preflight prefix in S3.
#   1. e2e_check.sh for 4-, 8- and 16-bit: full-run training (2 epochs on 256 pairs) with a kill -9 + resume,
#      adapters, generation on all three benchmarks, Megadiff scoring and unit-test scoring
#   2. dry training (30 optimizer steps, full data) of one cell per bit-width: s/step against the Modal dry run (#46)
#   3. evaluate_all.sh on the e2e adapters with EVAL_LIMIT=3: the official evaluation path end to end
#      (best epoch, _final, T = 0.8, no-description, scoring), then once more to check that it resumes (no calls)
#   4. timing run: M0 on the full HumanEval-Java at T = 0.2 with chunks of 5, to compare with the chunks-of-10
#      pilot on the same GPU type and re-cost the campaign
#   5. S3 restore check: a synced file is deleted locally and restored from S3
# The worst-case memory probe is not repeated: chunks of 5 were measured on an A10G for all ten models (#97).
set -euo pipefail
cd "$(dirname "$0")/.."
: "${RESULTS_BUCKET:?}"; : "${CARD:?}"
P=results/preflight; mkdir -p "$P"
say() { echo "PREFLIGHT $(date -Is) $*" | tee -a "$P/summary.txt"; }

say "host: $(nproc) vCPU, $(free -g | awk '/Mem/{print $2}') GiB RAM, $(nvidia-smi --query-gpu=name --format=csv,noheader)"

for b in 4 8 16; do
  say "1. e2e_check $b-bit start"
  E2E_BITS=$b bash scripts/e2e_check.sh > "$P/e2e_b$b.log" 2>&1 || { say "1. e2e_check $b-bit FAILED"; exit 1; }
  grep -q "resuming from" "results/e2e/b$b/M1/train_part2.log" || { say "1. $b-bit: no resume from a checkpoint"; exit 1; }
  [[ -f "results/e2e/b$b/M1/best_epoch.json" ]] || { say "1. $b-bit: no best_epoch.json"; exit 1; }
  say "1. e2e_check $b-bit OK (killed, resumed, adapters, generation, scoring)"
done

say "2. dry training M1 M4 M7, 30 steps"
DRY_RUN=30 bash scripts/run_matrix.sh M1 M4 M7 > "$P/dry_training.log" 2>&1
python - <<'EOF' | tee -a "$P/summary.txt"
import csv
for r in csv.DictReader(open("results/dry_log.csv")):
    keys = [k for k in r if any(t in k for t in ("run_name", "global_steps", "train_duration_excl", "samples_per_s", "peak_vram", "gpu_name"))]
    print("PREFLIGHT dry", {k: r[k] for k in keys})
EOF

say "3. evaluate_all.sh on the e2e adapters (EVAL_LIMIT=3)"
for pair in "M1 4" "M4 8" "M7 16"; do set -- $pair
  rm -rf "results/runs/$1"; mkdir -p results/runs; cp -r "results/e2e/b$2/M1" "results/runs/$1"
done
EVAL_LIMIT=3 bash scripts/evaluate_all.sh M0 M1 M4 M7 > "$P/evaluate_all.log" 2>&1
say "3. outputs: $(ls results/predictions | wc -l) prediction files, $(ls results/scores/*_tests.json | wc -l) score files"
EVAL_LIMIT=3 bash scripts/evaluate_all.sh M0 M1 M4 M7 > "$P/evaluate_all_rerun.log" 2>&1
n=$(grep -c "src/evaluator.py\|prompts from" "$P/evaluate_all_rerun.log" || true)
[[ "$n" == 0 ]] || { say "3. re-run was not a no-op ($n generation calls)"; exit 1; }
say "3. re-run skipped every step (resume OK)"
mv results/predictions "$P/predictions_eval_limit3"; mv results/scores "$P/scores_eval_limit3"
mv results/eval_log.csv "$P/eval_log_limit3.csv"; rm -rf results/runs/M1 results/runs/M4 results/runs/M7

say "4. timing: M0 full HumanEval-Java, T = 0.2, chunks of 5"
python src/evaluator.py --bits 16 --run_name PF_M0_chunk5 --benchmark humaneval_java --latency_samples 0 \
  --output_dir "$P/timing" --log_csv "$P/timing_log.csv" > "$P/timing.log" 2>&1
python - <<'EOF' | tee -a "$P/summary.txt"
import csv
r = list(csv.DictReader(open("results/preflight/timing_log.csv")))[-1]
print("PREFLIGHT timing", {k: r.get(k) for k in ("gen_duration_s", "gen_tokens_per_s", "gen_peak_vram_reserved_gib",
                                                  "hit_max_new_tokens", "batch_size", "gpu_name")})
EOF
python src/execution.py "$P/timing/PF_M0_chunk5_humaneval_java.jsonl" > "$P/timing_scores.json"
say "4. M0 HumanEval-Java pass@1: $(python -c "import json;d=json.load(open('$P/timing_scores.json'));print({k:v for b in d.values() for k,v in b.items() if k.startswith('pass_at')})")"

say "5. S3 restore check"
aws s3 sync --only-show-errors results/ "s3://$RESULTS_BUCKET/$CARD-preflight/results/" --exclude "*.tmp"
f=results/e2e/b4/M1/best_epoch.json; want=$(sha256sum "$f" | cut -c1-64); rm "$f"
aws s3 sync --only-show-errors "s3://$RESULTS_BUCKET/$CARD-preflight/results/" results/
[[ "$(sha256sum "$f" | cut -c1-64)" == "$want" ]] && say "5. restored from S3 OK" || { say "5. restore FAILED"; exit 1; }
say "ALL PRE-FLIGHT CHECKS PASSED"
