#!/usr/bin/env bash
# Daily cost check during the campaign (RESEARCH_LOG #100, failure case 6: the budget alert excludes credits).
# Runs on the local machine with the AWS CLI signed in.
#   CREDIT_LEFT_START=632.39 CREDIT_DATE=2026-11-11 bash scripts/credit_check.sh
# Prints credit applied per day since CREDIT_DATE (Cost Explorer lags by up to ~24 h), the running thesis
# instances with their uptime, and an estimate of the credit left = start value - credit applied - not-yet-billed
# uptime. The start value comes from the Billing console (Credits page); Cost Explorer cannot read the balance.
set -euo pipefail
: "${CREDIT_LEFT_START:?credit balance on CREDIT_DATE, from the Billing console}"
: "${CREDIT_DATE:?YYYY-MM-DD}"
PRICE=1.258   # g5.xlarge on-demand, eu-central-1, USD/h (AWS Price List API, 2026-11-11)
END=$(date -u -v+1d +%F 2>/dev/null || date -u -d tomorrow +%F)

echo "== credit applied per day (Cost Explorer)"
aws ce get-cost-and-usage --time-period "Start=$CREDIT_DATE,End=$END" --granularity DAILY --metrics UnblendedCost \
  --filter '{"Dimensions":{"Key":"RECORD_TYPE","Values":["Credit"]}}' \
  --query 'ResultsByTime[].[TimePeriod.Start,Total.UnblendedCost.Amount]' --output text |
  awk '{printf "%s  %8.2f\n", $1, -$2; s += -$2} END {printf "total       %8.2f\n", s; print s > "/tmp/.credit_used"}'
used=$(cat /tmp/.credit_used); rm -f /tmp/.credit_used

echo "== running thesis instances"
now=$(date -u +%s); unbilled=0
while read -r id type state launch; do
  [[ -z "$id" ]] && continue
  t=$(date -u -j -f "%Y-%m-%dT%H:%M:%S" "${launch%%+*}" +%s 2>/dev/null || date -u -d "$launch" +%s)
  h=$(awk -v a="$now" -v b="$t" 'BEGIN{printf "%.1f", (a-b)/3600}')
  echo "$id $type $state up ${h} h (~\$$(awk -v h="$h" -v p="$PRICE" 'BEGIN{printf "%.2f", h*p}'))"
  unbilled=$(awk -v u="$unbilled" -v h="$h" 'BEGIN{print u + (h > 24 ? 24 : h)}')   # ~ last day not yet in CE
done < <(aws ec2 describe-instances --region eu-central-1 --filters Name=tag:project,Values=thesis \
           Name=instance-state-name,Values=pending,running \
           --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,LaunchTime]' --output text)

left=$(awk -v s="$CREDIT_LEFT_START" -v u="$used" -v x="$unbilled" -v p="$PRICE" 'BEGIN{printf "%.2f", s-u-x*p}')
echo "== estimated credit left: \$$left (start $CREDIT_LEFT_START on $CREDIT_DATE)"
awk -v l="$left" 'BEGIN{exit !(l < 100)}' && echo "!! WARNING: under \$100 of credit left" || true
