#!/usr/bin/env bash
# Chains the morning GPU steps so the GPU isn't idle between them: wait for the
# verification of validation + test, finalize, swap Gemma for SEA-LION, run the judge
# prompt experiment. Training starts afterwards, once the prompt format is chosen.
#
# Usage: nohup setsid bash scripts/morning.sh > /dev/null 2>&1 &
# Progress: tail -f outputs/morning.log

set -uo pipefail
cd "$(dirname "$0")/.."
exec >> outputs/morning.log 2>&1
export VLLM_USE_FLASHINFER_SAMPLER=0
VLLM=~/vllm-env/bin/vllm

log() { echo "[$(date '+%F %T')] $*"; }

log "waiting for verification"
while pgrep -f "^.*python -m safetranslate.alter.run configs/alter.yaml verify" > /dev/null; do sleep 20; done
log "finalize (validation and test verified; train kept on automatic checks)"
uv run python -m safetranslate.alter.run configs/alter.yaml finalize --splits validation test

log "swapping Gemma for SEA-LION"
kill $(pgrep -f "^/home/calvin/vllm-env/bin/python /home/calvin/vllm-env/bin/vllm serve") 2> /dev/null
while pgrep -f "^/home/calvin/vllm-env/bin/python /home/calvin/vllm-env/bin/vllm serve" > /dev/null; do sleep 5; done
sleep 15
# Eager mode (its compile step runs out of memory); whitespace-free JSON avoids runaway replies
nohup "$VLLM" serve aisingapore/Llama-SEA-LION-v3-8B-IT --quantization fp8 --gpu-memory-utilization 0.85 \
    --max-model-len 4096 --enforce-eager \
    --structured-outputs-config '{"backend": "xgrammar", "disable_any_whitespace": true}' \
    > outputs/vllm-sealion-morning.log 2>&1 &
until curl -s localhost:8000/v1/models > /dev/null; do
    pgrep -f "vllm serve aisingapore" > /dev/null || { log "SEA-LION failed to start"; exit 1; }
    sleep 10
done
log "SEA-LION ready; running the judge prompt experiment"
uv run python -m safetranslate.harness.prompt_experiment configs/harness.yaml sealion 50
log "=== morning steps done ==="
