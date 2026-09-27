#!/usr/bin/env bash
# Runs the GPU queue unattended: finish alterations, verify, finalize, Phase 3 baselines,
# COMET. One model fits on the GPU at a time, so the script swaps vLLM servers between
# steps. Every stage resumes from its saved output, so the script can simply be re-run.
#
# Usage: nohup setsid bash scripts/overnight.sh > /dev/null 2>&1 &
# Progress: tail -f outputs/overnight.log

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs
exec >> outputs/overnight.log 2>&1

set -a && . ./.env && set +a  # HF_TOKEN for downloads, GEMINI_API_KEY for Gemma 4
export VLLM_USE_FLASHINFER_SAMPLER=0  # FlashInfer needs the CUDA compiler, absent in WSL
VLLM=~/vllm-env/bin/vllm
HARNESS="uv run python -m safetranslate.harness.run configs/harness.yaml"

log() { echo "[$(date '+%F %T')] $*"; }

stop_server() {
    pkill -f "vllm serve" || true
    while pgrep -f "vllm serve" > /dev/null; do sleep 5; done
    sleep 15  # give the GPU time to release its memory
}

# Starts a vLLM server and waits until it answers; returns non-zero if it dies instead
start_server() {
    log "starting server: $*"
    nohup "$VLLM" serve "$@" --max-model-len 4096 > "outputs/vllm-$(basename "$1").log" 2>&1 &
    local pid=$!
    until curl -s localhost:8000/v1/models > /dev/null; do
        kill -0 "$pid" 2> /dev/null || { log "server failed: see outputs/vllm-$(basename "$1").log"; return 1; }
        sleep 10
    done
    log "server ready"
}

# Harness steps are independent, so one failing shouldn't stop the others
try() { "$@" || log "FAILED (continuing): $*"; }

log "=== overnight run started ==="

# Gemma 4 runs on Google's servers, so its jobs run alongside the GPU steps
( try $HARNESS translate gemma4 --sample 1000; try $HARNESS real-errors gemma4 --sample 500 ) &
GEMMA4_PID=$!

log "1. waiting for the running alteration job"
while pgrep -f "safetranslate.alter.run configs/alter.yaml alter" > /dev/null; do sleep 60; done
log "retrying failed alterations once (Qwen server still up)"
uv run python -m safetranslate.alter.run configs/alter.yaml alter
log "Qwen as a comparison translator and real-error evaluator"
try $HARNESS translate qwen --sample 1000
try $HARNESS real-errors qwen

log "2. verification with Gemma"
stop_server
# Gemma 3 also has an image encoder; not loading it leaves more memory for text
start_server RedHatAI/gemma-3-12b-it-quantized.w4a16 --gpu-memory-utilization 0.85 \
        --limit-mm-per-prompt '{"image": 0}' \
    || start_server RedHatAI/gemma-3-12b-it-quantized.w4a16 --gpu-memory-utilization 0.85
uv run python -m safetranslate.alter.run configs/alter.yaml verify --out-dir data/altered/pilot-qwen3-8b-v5
uv run python -m safetranslate.alter.run configs/alter.yaml finalize --out-dir data/altered/pilot-qwen3-8b-v5
uv run python -m safetranslate.alter.run configs/alter.yaml verify
log "Gemma as a comparison translator and real-error evaluator"
try $HARNESS translate gemma --sample 1000
try $HARNESS real-errors gemma

log "3. finalize"
uv run python -m safetranslate.alter.run configs/alter.yaml finalize
# Starts after the first Gemma 4 jobs, so both don't share the rate limit at once
FIRST_GEMMA4_PID=$GEMMA4_PID
( while kill -0 "$FIRST_GEMMA4_PID" 2> /dev/null; do sleep 30; done
  try $HARNESS evaluator gemma4 --sample 50 ) &
GEMMA4_PID=$!

log "4. Phase 3 baselines with SEA-LION (FP8)"
stop_server
start_server aisingapore/Llama-SEA-LION-v3-8B-IT --quantization fp8 --gpu-memory-utilization 0.85
try $HARNESS evaluator sealion
try $HARNESS real-errors sealion
try $HARNESS translate sealion
for system in sealion qwen gemma gemma4; do
    [ -f "outputs/harness/$system/translations.jsonl" ] && try $HARNESS judge-translations "$system" --judge sealion
done

log "5. COMET-22 (needs the GPU free)"
stop_server
for system in sealion qwen gemma gemma4; do
    [ -f "outputs/harness/$system/translations.jsonl" ] && try uv run --group comet python -m safetranslate.harness.run configs/harness.yaml comet "$system"
done
log "waiting for the Gemma 4 jobs"
wait "$GEMMA4_PID" || true

log "=== overnight run finished ==="
