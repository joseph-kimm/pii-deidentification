#!/usr/bin/env bash
# Shared always-on deployment serving BOTH base Muse Glimmer 30B and the fine-tuned LoRA
# (Fireworks multi-LoRA: base deployed with --enable-addons, adapter loaded onto it).
#
# Training no longer happens here: dataset uploads are disabled after the October 2026
# incident, so SFT runs through the Training SDK in finetune/serverless_sft.py, which
# promotes the chosen checkpoint to accounts/$ACCOUNT_ID/models/$LORA_MODEL.
#
# Auth: `firectl signin` for firectl; `match-addons` also needs FIREWORKS_API_KEY.
# firectl blocks account-changing commands from AI agents, so run these yourself.
#
# Usage (run from the repo root, in this order):
#   finetune/fireworks_sft.sh match-addons  # read-only: BF16 shapes that support LoRA addons
#   finetune/fireworks_sft.sh deploy        # PAID: base model, addons on, 1 replica ALWAYS ON (bills until delete)
#   finetune/fireworks_sft.sh load-lora     # load the fine-tuned adapter onto that deployment
#   finetune/fireworks_sft.sh status        # deployment state
#   finetune/fireworks_sft.sh names         # model strings to call each model with
#   finetune/fireworks_sft.sh delete        # delete the deployment when finished -- billing stops only here
#
# The account ID is read from FIREWORKS_ACCOUNT_ID (env, or deidentify/.env) so it stays
# out of git. Every other name below can be overridden with an env var of the same name.
set -euo pipefail
export FIRECTL_NO_UPDATE_NOTICE=1

ENV_FILE="$(cd "$(dirname "$0")/.." && pwd)/deidentify/.env"
if [ -z "${FIREWORKS_ACCOUNT_ID:-}" ] && [ -f "$ENV_FILE" ]; then
  # Read only this one variable; the file also holds API keys.
  FIREWORKS_ACCOUNT_ID="$(sed -n 's/^FIREWORKS_ACCOUNT_ID=//p' "$ENV_FILE" | tail -1 | tr -d "\"' \r")"
fi
ACCOUNT_ID="${FIREWORKS_ACCOUNT_ID:-}"
if [ -z "$ACCOUNT_ID" ] && [ -n "${1:-}" ]; then
  echo "Set FIREWORKS_ACCOUNT_ID (in your shell or deidentify/.env; see deidentify/.env.example)" >&2
  exit 1
fi
BASE_MODEL="${BASE_MODEL:-accounts/fireworks/models/muse-glimmer-30b}"
LORA_MODEL="${LORA_MODEL:-deid-glimmer-30b-sft-synthetic}"     # --output-model used when promoting
DEPLOYMENT_ID="${DEPLOYMENT_ID:-deid-glimmer-multilora}"
# BF16 is required for addons (FP8/FP4 shapes reject --enable-addons). Confirm with match-addons.
DEPLOYMENT_SHAPE="${DEPLOYMENT_SHAPE:-accounts/fireworks/deploymentShapes/muse-glimmer-30b-multilora-gb300-bf16-w1-p1}"
# Hot-load (addon) deployments require min_replica_count == max_replica_count, so no
# scale-to-zero: the replica runs (and bills) continuously until `delete`.
REPLICAS="${REPLICAS:-1}"

DEPLOYMENT="accounts/$ACCOUNT_ID/deployments/$DEPLOYMENT_ID"

case "${1:-}" in
  match-addons)
    : "${FIREWORKS_API_KEY:?export FIREWORKS_API_KEY first}"
    curl -sS -X POST "https://api.fireworks.ai/v1/accounts/$ACCOUNT_ID/deploymentShapeVersions:match" \
      -H "Authorization: Bearer $FIREWORKS_API_KEY" -H "Content-Type: application/json" \
      -d "{\"createDeploymentRequest\": {\"deployment\": {\"baseModel\": \"$BASE_MODEL\", \"enableAddons\": true}}}"
    echo
    ;;
  deploy)
    firectl deployment create "$BASE_MODEL" \
      --deployment-id "$DEPLOYMENT_ID" \
      --deployment-shape "$DEPLOYMENT_SHAPE" \
      --enable-addons \
      --min-replica-count "$REPLICAS" \
      --max-replica-count "$REPLICAS" \
      --wait
    ;;
  load-lora)
    firectl load-lora "accounts/$ACCOUNT_ID/models/$LORA_MODEL" --deployment "$DEPLOYMENT_ID"
    ;;
  status)
    firectl deployment get "$DEPLOYMENT_ID"
    ;;
  names)
    echo "base:       $DEPLOYMENT"
    echo "fine-tuned: accounts/$ACCOUNT_ID/models/$LORA_MODEL#$DEPLOYMENT"
    ;;
  delete)
    firectl deployment delete "$DEPLOYMENT_ID"
    ;;
  *)
    sed -n '2,19p' "$0"
    exit 1
    ;;
esac
