#!/usr/bin/env bash
# Runs on an EC2 GPU instance started by scripts/aws/launch.sh: pulls inputs from S3,
# runs one job in Docker containers (vLLM model server + the project's pipeline image),
# and pushes logs and results to S3. The instance terminates itself when this ends.
#
# Jobs:
#   eval   base vs fine-tuned (QLoRA) evaluator on real and synthetic errors, served in FP8
#          (the headline comparison, both in one setup), then again at full 16-bit precision
#   train  LoRA training on the full 16-bit base (vs QLoRA on the PC), then evaluation at
#          16-bit, the same serving as the eval job's QLoRA run, so only the training differs
# (A10G GPUs have no FP8 hardware, so FP8 serving is skipped there.)

set -uo pipefail
ROLE=$1
export HOME="${HOME:-/root}"   # start-up scripts may run without HOME set
S3="s3://$BUCKET"
AWS="aws --region ap-southeast-1"   # the bucket's region
log() { echo "[$(date '+%F %T')] $*"; }
try() { "$@" || log "FAILED (continuing): $*"; }

# Logs and results go to S3 every 3 minutes, so progress survives the instance
mkdir -p outputs /opt/hf
( while true; do
    $AWS s3 cp /var/log/linguasg.log "$S3/logs/$ROLE.log" --quiet
    $AWS s3 sync outputs "$S3/results/$ROLE/" --quiet
    sleep 180
  done ) &
SYNC=$!

log "inputs from S3"
$AWS s3 sync "$S3/data/" data/ --quiet
$AWS s3 sync "$S3/adapters/" adapters/ --quiet
$AWS s3 sync "$S3/inputs/outputs/" outputs/ --quiet
touch outputs/mlflow.db

log "pipeline image (evaluation; the training environment is installed on the host)"
docker build --progress=plain -t linguasg . 2>&1 | grep -E "^#[0-9]+ (DONE|ERROR)|^ERROR" || true
docker image inspect linguasg > /dev/null 2>&1 || { log "image build failed"; exit 1; }

# The project's container, on the host network so it reaches the vLLM server on :8000
pipeline() {
    docker run --rm --network host --gpus all \
        -v "$PWD/data:/app/data" -v "$PWD/outputs:/app/outputs" -v "$PWD/configs:/app/configs" \
        -v /opt/hf:/root/.cache/huggingface \
        -e MLFLOW_TRACKING_URI=sqlite:////app/outputs/mlflow.db \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        linguasg "$@"
}

# vLLM's official image serving SEA-LION plus one adapter ($1); the rest are vLLM options
serve() {
    docker rm -f model > /dev/null 2>&1
    log "model server: adapter $1, options: ${*:2}"
    docker run -d --name model --gpus all --network host -v /opt/hf:/root/.cache/huggingface \
        -v "$PWD/$1:/adapters/evaluator:ro" --entrypoint vllm vllm/vllm-openai:v0.30.0 \
        serve aisingapore/Llama-SEA-LION-v3-8B-IT \
        --enable-lora --max-lora-rank 16 --lora-modules evaluator=/adapters/evaluator \
        --max-model-len 4096 --gpu-memory-utilization 0.9 \
        --structured-outputs-config '{"backend": "xgrammar", "disable_any_whitespace": true}' "${@:2}"
    for _ in $(seq 1 120); do
        curl -s localhost:8000/v1/models > /dev/null && { log "model server ready"; return 0; }
        [ "$(docker inspect -f '{{.State.Running}}' model 2> /dev/null)" = true ] || break
        sleep 10
    done
    docker logs --tail 80 model
    return 1
}
# If vLLM's compile step fails, eager mode skips it
serve_or_eager() { serve "$@" || serve "$@" --enforce-eager; }

# A harness config with its own output folder, so these results never mix with the PC's
config() { sed "s#^out_dir: .*#out_dir: $1#" configs/harness.yaml > "configs/harness-$2.yaml"; }
HARNESS="python -m safetranslate.harness.run"

if [ "$ROLE" = eval ]; then
    # FP8 first (as on the PC); 16-bit second. Each precision gets its own output folder.
    for precision in fp8 bf16; do
        config "outputs/harness-$precision" "$precision"
        options="--quantization fp8"; [ "$precision" = bf16 ] && options="--dtype bfloat16"
        serve_or_eager adapters/evaluator-qlora $options || { log "no $precision server"; continue; }
        for system in sealion-evaluator sealion; do
            try pipeline $HARNESS "configs/harness-$precision.yaml" real-errors "$system"
            try pipeline $HARNESS "configs/harness-$precision.yaml" evaluator "$system"
        done
    done
fi

if [ "$ROLE" = train ]; then
    # The GPU training stack (PyTorch + CUDA, ~10 GB) is installed straight from uv.lock on the
    # host: building it into an image took over 30 minutes on EC2's disks
    log "training environment from uv.lock"
    curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null
    export PATH="$HOME/.local/bin:$PATH" HF_HOME=/opt/hf
    uv sync --locked --group train 2>&1 | tail -3
    train() {
        MLFLOW_TRACKING_URI="sqlite:///$PWD/outputs/mlflow.db" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            uv run --no-sync python -m safetranslate.finetune.evaluator configs/finetune_evaluator_lora16.yaml
    }
    log "LoRA training on the full 16-bit base"
    # A crash is retried once: training resumes from its latest checkpoint
    train || train
    if [ -d outputs/finetune/evaluator-lora16/adapter ]; then
        mkdir -p adapters && cp -r outputs/finetune/evaluator-lora16/adapter adapters/evaluator-lora16
        config outputs/harness-lora16 lora16
        serve_or_eager adapters/evaluator-lora16 --dtype bfloat16 \
            && { try pipeline $HARNESS configs/harness-lora16.yaml real-errors sealion-evaluator
                 try pipeline $HARNESS configs/harness-lora16.yaml evaluator sealion-evaluator; }
    fi
fi

log "done; final upload"
kill "$SYNC"
docker rm -f model > /dev/null 2>&1
$AWS s3 sync outputs "$S3/results/$ROLE/" --quiet
$AWS s3 cp /var/log/linguasg.log "$S3/logs/$ROLE.log" --quiet
