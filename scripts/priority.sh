#!/usr/bin/env bash
# The night queue in priority order, for a run that may be stopped early: resume-critical
# numbers first (fine-tuned vs base evaluator, the workflow and its independent check),
# then the rest. Every step resumes from its saved output, so re-running continues where
# it stopped; the report is rebuilt after each stage so finished results are always there.
#
# Usage: nohup setsid bash scripts/priority.sh > /dev/null 2>&1 &
# Progress: tail -f outputs/priority.log

set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/screenshots
exec >> outputs/priority.log 2>&1
# Only one copy may run: a second would fight the first for the GPU
exec 9> outputs/priority.lock
flock -n 9 || { echo "priority.sh is already running"; exit 1; }

set -a && . ./.env && set +a  # HF_TOKEN for downloads, GEMINI_API_KEY for Gemma 4
export VLLM_USE_FLASHINFER_SAMPLER=0  # FlashInfer needs the CUDA compiler, absent in WSL
VLLM=~/vllm-env/bin/vllm
HARNESS="uv run python -m safetranslate.harness.run configs/harness.yaml"
WORKFLOW="uv run python -m safetranslate.workflow.run configs/workflow.yaml"
SEALION=aisingapore/Llama-SEA-LION-v3-8B-IT
GEMMA=RedHatAI/gemma-3-12b-it-quantized.w4a16
EDGE="/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"

log() { echo "[$(date '+%F %T')] $*"; }
try() { "$@" || log "FAILED (continuing): $*"; }
report() { try uv run python -m safetranslate.harness.report configs/harness.yaml; }

stop_server() {
    pkill -f "^$HOME/vllm-env/bin/python $VLLM serve" || true
    while pgrep -f "^$HOME/vllm-env/bin/python $VLLM serve" > /dev/null; do sleep 5; done
    sleep 15  # give the GPU time to release its memory
}

# Starts a vLLM server (eager: compile runs out of memory on 12 GB; whitespace-free JSON
# avoids runaway replies); non-zero if it dies or isn't ready within 20 minutes
start_server() {
    local server_log="outputs/vllm-$(basename "$1")-$(date +%H%M%S).log"
    log "starting server: $* (log: $server_log)"
    nohup "$VLLM" serve "$@" --enforce-eager \
        --structured-outputs-config '{"backend": "xgrammar", "disable_any_whitespace": true}' \
        > "$server_log" 2>&1 9>&- &   # 9>&-: the server must not inherit (and keep) the lock
    local pid=$! waited=0
    until curl -s localhost:8000/v1/models > /dev/null; do
        kill -0 "$pid" 2> /dev/null || { log "server failed: see $server_log"; return 1; }
        (( waited += 10 )); (( waited < 1200 )) || { log "server not ready after 20 min"; kill "$pid"; return 1; }
        sleep 10
    done
    log "server ready"
}

# A LoRA server can start but fail on the first request, so one real request is sent
answers() {
    curl -sf localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
        -d "{\"model\": \"$1\", \"messages\": [{\"role\": \"user\", \"content\": \"Hi\"}], \"max_tokens\": 5}" > /dev/null
}

# SEA-LION with the evaluator adapter. Fallbacks: adapter on the FP8 base -> on a 4-bit
# base -> merged model (evaluator only). Sets BOTH_MODELS=false when the base isn't served.
serve_sealion() {
    local lora=(--enable-lora --max-lora-rank 16 --lora-modules "evaluator=$ADAPTER" --max-model-len 4096)
    BOTH_MODELS=true
    # An FP8 request cache doubles how many requests fit next to the model on 12 GB
    if start_server "$SEALION" --quantization fp8 --kv-cache-dtype fp8 --gpu-memory-utilization 0.85 "${lora[@]}" && answers evaluator; then
        log "serving: FP8 base + adapter (FP8 request cache)"
    elif stop_server; start_server "$SEALION" --quantization fp8 --gpu-memory-utilization 0.85 "${lora[@]}" && answers evaluator; then
        log "serving: FP8 base + adapter"
    elif stop_server; start_server "$SEALION" --quantization bitsandbytes --gpu-memory-utilization 0.85 "${lora[@]}" && answers evaluator; then
        log "serving: 4-bit base + adapter"
    else
        stop_server
        log "serving: merged model (steps needing the base model are skipped)"
        BOTH_MODELS=false
        local merged=outputs/finetune/evaluator/merged
        [ -d "$merged" ] || try uv run --group train python -m safetranslate.finetune.merge "$ADAPTER" "$merged"
        start_server "$merged" --served-model-name evaluator --quantization fp8 --gpu-memory-utilization 0.85 --max-model-len 4096
    fi
}

# LinguaSG with the real models, screenshotted by headless Edge via its demo links
screenshots() {
    log "screenshots of LinguaSG with the real evaluator"
    GRADIO_SERVER_PORT=7861 nohup uv run --group app python -m safetranslate.app configs/workflow.yaml \
        > outputs/app-real.log 2>&1 9>&- &
    local app=$! waited=0
    until curl -s localhost:7861 > /dev/null; do
        (( waited += 5 )); (( waited < 180 )) || { log "app did not start"; kill "$app"; return 1; }
        sleep 5
    done
    local out
    out=$(wslpath -w "$PWD/outputs/screenshots")
    for shot in check-zh:"#example" check-ms:"#example-ms" check-ta:"#example-ta" translate:"#translate-example"; do
        timeout 180 "$EDGE" --headless=new --disable-gpu --hide-scrollbars --window-size=1440,2000 \
            --virtual-time-budget=90000 --user-data-dir="$out\\edge-profile" \
            --screenshot="$out\\${shot%%:*}.png" "http://localhost:7861/${shot#*:}" > /dev/null 2>&1
    done
    kill "$app"
    log "screenshots saved in outputs/screenshots"
}

log "=== priority run started ==="

# Gemma 4 runs on Google's servers: after its running job, retry any failed items
# 9>&-: these API jobs must not hold the lock after the script ends
( exec 9>&-
  while pgrep -f "harness.run configs/harness.yaml evaluator gemma4" > /dev/null; do sleep 60; done
  try $HARNESS evaluator gemma4 --sample 50
  try $HARNESS real-errors gemma4 --sample 500
  try $HARNESS translate gemma4 --sample 1000 ) &

log "1. evaluator training (started separately; resumes from the latest checkpoint)"
while pgrep -f "safetranslate.finetune.evaluator configs" > /dev/null; do sleep 60; done
# A crash (e.g. a busy MLflow database) is retried: each attempt resumes from a checkpoint
for attempt in 1 2 3; do
    [ -d outputs/finetune/evaluator/adapter ] && break
    log "training attempt $attempt"
    try uv run --group train python -m safetranslate.finetune.evaluator configs/finetune_evaluator.yaml
done
ADAPTER=outputs/finetune/evaluator/adapter
[ -d "$ADAPTER" ] || ADAPTER=$(ls -d outputs/finetune/evaluator/checkpoint-* | sort -t- -k2 -n | tail -1)
log "adapter: $ADAPTER"

log "2. the workflow (base translator + fine-tuned evaluator) and screenshots"
stop_server
serve_sealion
if $BOTH_MODELS; then
    try $WORKFLOW run
    try screenshots
fi
stop_server

log "3. Gemma 3 12B: independent check of the workflow, then its own translations"
# Tight on 12 GB: no image input, shorter context
GEMMA_FLAGS=(--gpu-memory-utilization 0.87 --max-model-len 2048 --limit-mm-per-prompt '{"image": 0}')
if start_server "$GEMMA" "${GEMMA_FLAGS[@]}"; then
    [ -f outputs/workflow/runs.jsonl ] && try $WORKFLOW check
    try $HARNESS translate gemma --sample 1000
    try $HARNESS real-errors gemma
fi
report
stop_server

log "3b. fine-tuned vs base evaluator on this PC (backup for the EC2 run)"
serve_sealion
try $HARNESS real-errors sealion-evaluator
if $BOTH_MODELS; then
    # Base model re-scored with the same server settings (last night's settings differed)
    OLD=outputs/harness/sealion/real_errors
    if [ ! -f "$OLD/predictions_before_whitespace_fix.jsonl" ] && [ -f "$OLD/predictions.jsonl" ]; then
        mv "$OLD/predictions.jsonl" "$OLD/predictions_before_whitespace_fix.jsonl"
        [ -f "$OLD/metrics.json" ] && mv "$OLD/metrics.json" "$OLD/metrics_before_whitespace_fix.json"
    fi
    try $HARNESS real-errors sealion
fi
try $HARNESS evaluator sealion-evaluator
$BOTH_MODELS && try $HARNESS evaluator sealion
report
stop_server

log "4. the fine-tuned evaluator judges every model's translations (SEA-LION's 20k last)"
serve_sealion
$BOTH_MODELS && try $HARNESS translate sealion
for system in qwen gemma gemma4 sealion; do
    [ -f "outputs/harness/$system/translations.jsonl" ] && try $HARNESS judge-translations "$system" --judge sealion-evaluator
done
report
stop_server

log "5. Qwen gaps, COMET, report"
start_server Qwen/Qwen3-8B-AWQ --gpu-memory-utilization 0.85 --max-model-len 4096 \
    && try $HARNESS translate qwen --sample 1000
stop_server
for system in sealion qwen gemma gemma4; do
    [ -f "outputs/harness/$system/translations.jsonl" ] && try uv run --group comet python -m safetranslate.harness.run configs/harness.yaml comet "$system"
done
report
log "=== priority run finished ==="
