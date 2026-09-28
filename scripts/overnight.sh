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

# Starts a vLLM server and waits until it answers; returns non-zero if it dies instead.
# Each attempt gets its own log, so a failed attempt's error isn't overwritten.
start_server() {
    local server_log="outputs/vllm-$(basename "$1")-$(date +%H%M%S).log"
    log "starting server: $* (log: $server_log)"
    nohup "$VLLM" serve "$@" --max-model-len 4096 > "$server_log" 2>&1 &
    local pid=$!
    until curl -s localhost:8000/v1/models > /dev/null; do
        kill -0 "$pid" 2> /dev/null || { log "server failed: see $server_log"; return 1; }
        sleep 10
    done
    log "server ready"
}

# vLLM's compile step crashes on some models; --enforce-eager skips it (slower, reliable)
start_server_or_eager() { start_server "$@" || { stop_server; start_server "$@" --enforce-eager; }; }

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

log "2. verification with Gemma"
stop_server
# Gemma 3 12B is tight on 12 GB: eager mode (its compile step crashes), no image input,
# shorter context (verification prompts are short); fallback is another 4-bit build
# served under the same name. If Gemma can't start, verification is skipped and the rest
# of the night still runs.
GEMMA_FLAGS=(--gpu-memory-utilization 0.9 --enforce-eager --max-model-len 2048 --limit-mm-per-prompt '{"image": 0}')
if start_server RedHatAI/gemma-3-12b-it-quantized.w4a16 "${GEMMA_FLAGS[@]}" \
    || { stop_server; start_server gaunernst/gemma-3-12b-it-int4-awq "${GEMMA_FLAGS[@]}" \
         --served-model-name RedHatAI/gemma-3-12b-it-quantized.w4a16; }; then
    try uv run python -m safetranslate.alter.run configs/alter.yaml verify --out-dir data/altered/pilot-qwen3-8b-v5
    try uv run python -m safetranslate.alter.run configs/alter.yaml finalize --out-dir data/altered/pilot-qwen3-8b-v5
    try uv run python -m safetranslate.alter.run configs/alter.yaml verify
    log "Gemma as a comparison translator and real-error evaluator"
    try $HARNESS translate gemma --sample 1000
    try $HARNESS real-errors gemma
else
    log "Gemma could not start: verification skipped, continuing with the rest"
fi

log "3. finalize"
try uv run python -m safetranslate.alter.run configs/alter.yaml finalize
# Starts after the first Gemma 4 jobs, so both don't share the rate limit at once
FIRST_GEMMA4_PID=$GEMMA4_PID
( while kill -0 "$FIRST_GEMMA4_PID" 2> /dev/null; do sleep 30; done
  [ -s data/altered/test.jsonl ] && try $HARNESS evaluator gemma4 --sample 50 ) &
GEMMA4_PID=$!

log "4. Qwen as a comparison translator and real-error evaluator"
# Done after verification (not before) so the verifier can be checked as early as possible
stop_server
start_server_or_eager Qwen/Qwen3-8B-AWQ --gpu-memory-utilization 0.85
try $HARNESS translate qwen --sample 1000
try $HARNESS real-errors qwen

log "5. Phase 3 baselines with SEA-LION (FP8)"
stop_server
start_server_or_eager aisingapore/Llama-SEA-LION-v3-8B-IT --quantization fp8 --gpu-memory-utilization 0.85
[ -s data/altered/test.jsonl ] && try $HARNESS evaluator sealion
try $HARNESS real-errors sealion
try $HARNESS translate sealion
for system in sealion qwen gemma gemma4; do
    [ -f "outputs/harness/$system/translations.jsonl" ] && try $HARNESS judge-translations "$system" --judge sealion
done

log "6. COMET-22 (needs the GPU free)"
stop_server
for system in sealion qwen gemma gemma4; do
    [ -f "outputs/harness/$system/translations.jsonl" ] && try uv run --group comet python -m safetranslate.harness.run configs/harness.yaml comet "$system"
done
log "waiting for the Gemma 4 jobs"
wait "$GEMMA4_PID" || true

log "=== overnight run finished ==="
