#!/usr/bin/env bash
# Runs one card's share of the official campaign on an AWS GPU VM (RESEARCH_LOG #101, #102).
# Started by the user data that scripts/aws_launch.sh writes; it can also be run by hand on the VM:
#   RESULTS_BUCKET=<bucket> CARD=A bash scripts/card_job.sh
# What it does, in order:
#   1. restores this card's earlier results from S3 (so a relaunched instance resumes where the old one stopped),
#   2. fetches the data files from S3 and checks them against data/SHA256SUMS,
#   3. trains the card's cells (scripts/run_matrix.sh skips finished cells, the trainer resumes from checkpoints),
#   4. evaluates the card's models (scripts/evaluate_all.sh skips finished steps),
#   5. syncs results/ to S3 every SYNC_EVERY seconds, at every stage change, and at the end,
#   6. writes status.txt (stage, last log lines, GPU state) to S3 for monitoring without logging in.
# On success it writes DONE to S3 and powers the instance off (stop, not terminate). On failure it writes
# FAILED-<UTC time> (one marker per attempt; status.txt is always the current state) and leaves the instance running for
# debugging (Session Manager). Nothing is ever deleted, locally or in S3.
set -uo pipefail
cd "$(dirname "$0")/.."

: "${RESULTS_BUCKET:?set RESULTS_BUCKET}"; : "${CARD:?set CARD (A or B)}"
SYNC_EVERY=${SYNC_EVERY:-600}
PREFLIGHT=${PREFLIGHT:-}            # non-empty: run scripts/preflight.sh instead of the campaign; S3 prefix <card>-preflight
S3="s3://$RESULTS_BUCKET/${CARD}${PREFLIGHT:+-preflight}"
LOG=results/card_job.log
mkdir -p results data

# Cell assignment, balanced on estimated GPU-hours (training + evaluation per model; RESEARCH_LOG #102).
case "$CARD" in
  A) CELLS=(M4 M5 M6 M7 M9); SEED_CELLS=(M5); MODELS=(M0 M4 M5 M6 M7 M9 M5_s43 M5_s44) ;;
  B) CELLS=(M1 M2 M3 M8);    SEED_CELLS=(M2 M8); MODELS=(M1 M2 M3 M8 M2_s43 M2_s44 M8_s43 M8_s44) ;;
  *) echo "unknown CARD $CARD"; exit 2 ;;
esac

source .venv/bin/activate
export HEJ_JAVA_HOME=${HEJ_JAVA_HOME:-$(ls -d /usr/lib/jvm/java-8-openjdk-* | head -1)}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # same as the Modal pilots (RESEARCH_LOG #46)

stage() { STAGE="$1"; echo "=== $(date -Is) STAGE $STAGE" | tee -a "$LOG"; push; }
push() {  # results/ -> S3 (never deletes in S3; the bucket is versioned), plus a status file.
  # Runs in a subshell with errexit off and always returns 0: monitoring must never stop the campaign.
  (
    set +e +o pipefail
    { echo "card=$CARD stage=${STAGE:-start} time=$(date -Is) host=$(hostname) commit=$(git rev-parse --short HEAD)"
      nvidia-smi --query-gpu=name,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
      df -h / | tail -1; free -m | sed -n 2p; echo "--- last log lines"; tail -n 25 "$LOG"
    } > results/status.txt 2>/dev/null
    aws s3 sync --only-show-errors results/ "$S3/results/" --exclude "*.tmp" || echo "$(date -Is) WARN: S3 sync failed" >> "$LOG"
  ) || true
}
sync_loop() { while sleep "$SYNC_EVERY"; do push; done; }

finish() {
  local rc=$?
  set +e
  kill "${SYNC_PID:-}" 2>/dev/null
  if [[ $rc == 0 ]]; then
    stage DONE
    aws s3 cp --only-show-errors results/status.txt "$S3/DONE" && sudo shutdown -h +2   # power off only if DONE is in S3
  else
    STAGE="FAILED (exit $rc at ${STAGE:-?})"; echo "=== $(date -Is) $STAGE" >> "$LOG"; push
    aws s3 cp --only-show-errors results/status.txt "$S3/FAILED-$(date -u +%Y%m%dT%H%M%SZ)"  # one marker per failed attempt
  fi
}
trap finish EXIT
set -e

STAGE=restore; echo "=== $(date -Is) card $CARD starting${PREFLIGHT:+ (PRE-FLIGHT)}" | tee -a "$LOG"
aws s3 sync --only-show-errors "$S3/results/" results/ || true   # nothing there on the first launch
aws s3 sync --only-show-errors "s3://$RESULTS_BUCKET/data/" data/
( cd data && sha256sum -c SHA256SUMS ) | tee -a "$LOG"
sync_loop & SYNC_PID=$!

if [[ -n "$PREFLIGHT" ]]; then
  # Pre-flight (step B of RESEARCH_LOG #100): see scripts/preflight.sh. Not an official result.
  stage "pre-flight checks"
  bash scripts/preflight.sh 2>&1 | tee -a "$LOG"
  exit 0
fi

stage "training: ${CELLS[*]}"
bash scripts/run_matrix.sh "${CELLS[@]}" 2>&1 | tee -a "$LOG"
stage "training seeds 43 44: ${SEED_CELLS[*]}"
SEEDS="43 44" bash scripts/run_matrix.sh "${SEED_CELLS[@]}" 2>&1 | tee -a "$LOG"
stage "evaluation: ${MODELS[*]}"
bash scripts/evaluate_all.sh "${MODELS[@]}" 2>&1 | tee -a "$LOG"
