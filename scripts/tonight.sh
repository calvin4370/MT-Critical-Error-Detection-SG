#!/usr/bin/env bash
# Runs the 28 Sep night queue unattended: finish the evaluator training, fill the harness
# gaps, score the fine-tuned evaluator, run the workflow, check it, COMET, report.
# One model fits on the GPU at a time, so vLLM servers are swapped between steps. Every
# step resumes from its saved output, so the script can simply be re-run.
#
# Usage: nohup setsid bash scripts/tonight.sh > /dev/null 2>&1 &
# Progress: tail -f outputs/tonight.log

set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs
exec >> outputs/tonight.log 2>&1
# Only one copy may run: a second would fight the first for the GPU
exec 9> outputs/tonight.lock
flock -n 9 || { echo "tonight.sh is already running"; exit 1; }

set -a && . ./.env && set +a  # HF_TOKEN for downloads, GEMINI_API_KEY for Gemma 4
export VLLM_USE_FLASHINFER_SAMPLER=0  # FlashInfer needs the CUDA compiler, absent in WSL
VLLM=~/vllm-env/bin/vllm
HARNESS="uv run python -m safetranslate.harness.run configs/harness.yaml"
WORKFLOW="uv run python -m safetranslate.workflow.run configs/workflow.yaml"
SEALION=aisingapore/Llama-SEA-LION-v3-8B-IT
GEMMA=RedHatAI/gemma-3-12b-it-quantized.w4a16

log() { echo "[$(date '+%F %T')] $*"; }

stop_server() {
    pkill -f "^$HOME/vllm-env/bin/python $VLLM serve" || true
    while pgrep -f "^$HOME/vllm-env/bin/python $VLLM serve" > /dev/null; do sleep 5; done
    sleep 15  # give the GPU time to release its memory
}

# Starts a vLLM server (eager: compile runs out of memory on 12 GB; whitespace-free JSON
# avoids runaway replies) and waits until it answers; non-zero if it dies instead
start_server() {
    local server_log="outputs/vllm-$(basename "$1")-$(date +%H%M%S).log"
    log "starting server: $* (log: $server_log)"
    nohup "$VLLM" serve "$@" --enforce-eager \
        --structured-outputs-config '{"backend": "xgrammar", "disable_any_whitespace": true}' \
        > "$server_log" 2>&1 &
    local pid=$! waited=0
    until curl -s localhost:8000/v1/models > /dev/null; do
        kill -0 "$pid" 2> /dev/null || { log "server failed: see $server_log"; return 1; }
        # A server that hangs without dying would otherwise stall the whole night
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

# Harness steps are independent, so one failing shouldn't stop the others
try() { "$@" || log "FAILED (continuing): $*"; }

log "=== tonight's run started ==="

# Gemma 4 runs on Google's servers, alongside the GPU steps; one job at a time so they
# don't share the rate limit
( try $HARNESS real-errors gemma4 --sample 500
  try $HARNESS translate gemma4 --sample 1000
  try $HARNESS evaluator gemma4 --sample 50 ) &
GEMMA4_PID=$!

log "1. evaluator training (started separately; resumes from the latest checkpoint)"
while pgrep -f "safetranslate.finetune.evaluator configs" > /dev/null; do sleep 60; done
# A crash (e.g. a busy MLflow database) is retried: each attempt resumes from a checkpoint
for attempt in 1 2 3; do
    [ -d outputs/finetune/evaluator/adapter ] && break
    log "training attempt $attempt"
    try uv run --group train python -m safetranslate.finetune.evaluator configs/finetune_evaluator.yaml
done
# If training never finished, the latest checkpoint is still a usable adapter
ADAPTER=outputs/finetune/evaluator/adapter
[ -d "$ADAPTER" ] || ADAPTER=$(ls -d outputs/finetune/evaluator/checkpoint-* | sort -t- -k2 -n | tail -1)
log "adapter: $ADAPTER"

log "2. Qwen: fill its translation gaps"
start_server Qwen/Qwen3-8B-AWQ --gpu-memory-utilization 0.85 --max-model-len 4096 \
    && try $HARNESS translate qwen --sample 1000
stop_server

log "3. Gemma 3 12B: comparison translator and real-error evaluator"
# Tight on 12 GB: no image input, shorter context
GEMMA_FLAGS=(--gpu-memory-utilization 0.87 --max-model-len 2048 --limit-mm-per-prompt '{"image": 0}')
if start_server "$GEMMA" "${GEMMA_FLAGS[@]}"; then
    try $HARNESS translate gemma --sample 1000
    try $HARNESS real-errors gemma
fi
stop_server

log "4. SEA-LION with the fine-tuned evaluator adapter"
# Fallbacks: the adapter on the FP8 base (fastest) -> on a 4-bit base (the same kind of
# base it was trained on) -> merged into the base (evaluator only: no base translator)
LORA=(--enable-lora --max-lora-rank 16 --lora-modules "evaluator=$ADAPTER" --max-model-len 4096)
BOTH_MODELS=true
if start_server "$SEALION" --quantization fp8 --gpu-memory-utilization 0.85 "${LORA[@]}" && answers evaluator; then
    log "serving: FP8 base + adapter"
elif stop_server; start_server "$SEALION" --quantization bitsandbytes --gpu-memory-utilization 0.85 "${LORA[@]}" && answers evaluator; then
    log "serving: 4-bit base + adapter"
else
    stop_server
    log "serving: merged model (the workflow is skipped: it needs the base translator too)"
    BOTH_MODELS=false
    MERGED=outputs/finetune/evaluator/merged
    [ -d "$MERGED" ] || try uv run --group train python -m safetranslate.finetune.merge "$ADAPTER" "$MERGED"
    start_server "$MERGED" --served-model-name evaluator --quantization fp8 --gpu-memory-utilization 0.85 --max-model-len 4096
fi
# Headline numbers first: fine-tuned vs base evaluator (base needs the base model served)
try $HARNESS evaluator sealion-evaluator
try $HARNESS real-errors sealion-evaluator
if $BOTH_MODELS; then
    try $HARNESS evaluator sealion
    try $HARNESS translate sealion
    log "5. workflow: base translator + fine-tuned evaluator"
    try $WORKFLOW run
fi
# The fine-tuned evaluator judges every model's translations (SEA-LION's 20k last: longest)
for system in qwen gemma gemma4 sealion; do
    [ -f "outputs/harness/$system/translations.jsonl" ] && try $HARNESS judge-translations "$system" --judge sealion-evaluator
done
stop_server

log "6. Gemma 3 12B checks the workflow's output (a different model family)"
if [ -f outputs/workflow/runs.jsonl ] && start_server "$GEMMA" "${GEMMA_FLAGS[@]}"; then
    try $WORKFLOW check
fi
stop_server

log "7. COMET-22 (needs the GPU free) and the results report"
for system in sealion qwen gemma gemma4; do
    [ -f "outputs/harness/$system/translations.jsonl" ] && try uv run --group comet python -m safetranslate.harness.run configs/harness.yaml comet "$system"
done
log "waiting for the Gemma 4 jobs"
wait "$GEMMA4_PID" || true
try uv run python -m safetranslate.harness.report configs/harness.yaml

log "=== tonight's run finished ==="
