#!/usr/bin/env bash
# Runs after tonight.sh: re-scores base SEA-LION on real errors with tonight's server
# settings (a fair before/after), retries Gemma 4's failed items, rebuilds the report.
#
# Usage: nohup setsid bash scripts/after.sh > /dev/null 2>&1 &
# Progress: tail -f outputs/after.log

set -uo pipefail
cd "$(dirname "$0")/.."
exec >> outputs/after.log 2>&1

set -a && . ./.env && set +a  # GEMINI_API_KEY for Gemma 4
export VLLM_USE_FLASHINFER_SAMPLER=0  # FlashInfer needs the CUDA compiler, absent in WSL
VLLM=~/vllm-env/bin/vllm
HARNESS="uv run python -m safetranslate.harness.run configs/harness.yaml"

log() { echo "[$(date '+%F %T')] $*"; }
try() { "$@" || log "FAILED (continuing): $*"; }

log "waiting for tonight.sh to finish"
flock outputs/tonight.lock true
log "=== after.sh started ==="

# Gemma 4 retries use the API only, so they run alongside the GPU step
( try $HARNESS real-errors gemma4 --sample 500
  try $HARNESS translate gemma4 --sample 1000 ) &
GEMMA4_PID=$!

log "1. base SEA-LION on real errors, with the same server settings as the fine-tuned model"
OLD=outputs/harness/sealion/real_errors
# Last night's run used different decoding settings; kept for the record, not deleted
[ -f "$OLD/predictions.jsonl" ] && mv "$OLD/predictions.jsonl" "$OLD/predictions_before_whitespace_fix.jsonl"
[ -f "$OLD/metrics.json" ] && mv "$OLD/metrics.json" "$OLD/metrics_before_whitespace_fix.json"
nohup "$VLLM" serve aisingapore/Llama-SEA-LION-v3-8B-IT --quantization fp8 --gpu-memory-utilization 0.85 \
    --max-model-len 4096 --enforce-eager \
    --structured-outputs-config '{"backend": "xgrammar", "disable_any_whitespace": true}' \
    > outputs/vllm-sealion-after.log 2>&1 &
SERVER=$!
waited=0
until curl -s localhost:8000/v1/models > /dev/null; do
    kill -0 "$SERVER" 2> /dev/null || { log "server failed: see outputs/vllm-sealion-after.log"; break; }
    (( waited += 10 )); (( waited < 1200 )) || { log "server not ready after 20 min"; break; }
    sleep 10
done
curl -s localhost:8000/v1/models > /dev/null && try $HARNESS real-errors sealion
kill "$SERVER" 2> /dev/null
while kill -0 "$SERVER" 2> /dev/null; do sleep 5; done
sleep 15

log "2. waiting for the Gemma 4 retries, then COMET for its new translations"
wait "$GEMMA4_PID" || true
try uv run --group comet python -m safetranslate.harness.run configs/harness.yaml comet gemma4

log "3. results report"
try uv run python -m safetranslate.harness.report configs/harness.yaml
log "=== after.sh finished ==="
