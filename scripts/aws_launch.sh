#!/usr/bin/env bash
# Launch one card of the official campaign (or the pre-flight) on AWS, retrying until capacity is available
# (RESEARCH_LOG #99, #101, #102). Runs on the local machine with the AWS CLI signed in.
#   RESULTS_BUCKET=<bucket> bash scripts/aws_launch.sh A            # card A of the campaign
#   RESULTS_BUCKET=<bucket> bash scripts/aws_launch.sh B
#   RESULTS_BUCKET=<bucket> PREFLIGHT=1 bash scripts/aws_launch.sh A  # pre-flight on one instance
# Fixed for every official run: g5.xlarge (A10G 24 GB, 4 vCPU, 16 GB RAM), eu-central-1, the Deep Learning OSS
# Nvidia Driver AMI below, a 150 GB gp3 root volume, the IAM instance profile `thesis-gpu-runner` (S3 access to the
# results bucket only, plus Session Manager). No SSH key and no open ports: debug with Session Manager.
# The instance clones this repository at COMMIT (default: the local HEAD, which must be pushed), runs
# scripts/setup_gpu_vm.sh and starts scripts/card_job.sh in tmux. A shutdown stops the instance (it is not
# terminated), so the disk survives; card_job.sh powers off only after a successful run with everything in S3.
set -euo pipefail
cd "$(dirname "$0")/.."
CARD=${1:?usage: aws_launch.sh A|B}
: "${RESULTS_BUCKET:?set RESULTS_BUCKET}"
REGION=eu-central-1
TYPE=g5.xlarge
AMI=${AMI:-ami-0efde0090dab8cacf}   # Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.7 (Ubuntu 22.04) 20260427
COMMIT=${COMMIT:-$(git rev-parse HEAD)}
TRIES=${TRIES:-180}; WAIT=${WAIT:-120}
NAME="thesis-card-$CARD${PREFLIGHT:+-preflight}"

git fetch -q origin
git merge-base --is-ancestor "$COMMIT" origin/main || { echo "commit $COMMIT is not on origin/main: push first"; exit 1; }

UD=$(mktemp)
cat > "$UD" <<EOF
#!/bin/bash
exec > >(tee -a /var/log/thesis-bootstrap.log) 2>&1
set -euxo pipefail
sudo -u ubuntu -H bash -lc '
  set -euxo pipefail
  cd ~ && [ -d peft-code-eval ] || git clone -q https://github.com/WahidSaeed/peft-code-eval
  cd peft-code-eval && git fetch -q origin && git checkout -q $COMMIT
  [ -f .venv/.setup-done ] || { bash scripts/setup_gpu_vm.sh && touch .venv/.setup-done; }
  tmux new-session -d -s card "RESULTS_BUCKET=$RESULTS_BUCKET CARD=$CARD PREFLIGHT=${PREFLIGHT:-} bash scripts/card_job.sh; exec bash"
'
EOF

for i in $(seq 1 "$TRIES"); do
  out=$(aws ec2 run-instances --region "$REGION" --image-id "$AMI" --instance-type "$TYPE" --count 1 \
    --iam-instance-profile Name=thesis-gpu-runner \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=150,VolumeType=gp3,DeleteOnTermination=true}' \
    --instance-initiated-shutdown-behavior stop --metadata-options HttpTokens=required \
    --user-data "file://$UD" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME},{Key=project,Value=thesis},{Key=card,Value=$CARD}]" \
    --query 'Instances[0].[InstanceId,Placement.AvailabilityZone]' --output text 2>&1) || true
  if [[ $out == i-* ]]; then echo "$(date -u +%H:%M:%S) LAUNCHED $NAME: $out (commit ${COMMIT:0:7})"; rm -f "$UD"; exit 0; fi
  echo "$(date -u +%H:%M:%S) try $i/$TRIES: $(echo "$out" | tail -1 | cut -c1-110)"
  [[ $out == *InsufficientInstanceCapacity* ]] || { echo "not a capacity error: stopping"; rm -f "$UD"; exit 1; }
  sleep "$WAIT"
done
rm -f "$UD"; echo "gave up after $TRIES tries"; exit 1
