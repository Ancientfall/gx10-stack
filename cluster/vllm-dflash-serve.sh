#!/usr/bin/env bash
# vllm-dflash-serve.sh - single-node vLLM serving with DFlash speculative decoding
# on ONE GB10 / DGX Spark. DFlash drafts several tokens per forward pass and verifies
# them in a single pass, so it beats the memory-bandwidth wall that caps plain decoding
# (~12 tok/s for a dense 27B at ~273 GB/s). Distinct from 02-launch-cluster.sh, which is
# the multi-node Ray path. Serves the OpenAI-compatible API on API_PORT (default 8000),
# so the panel detects it as the vLLM engine (full latency/KV/throughput telemetry).
#
# Usage:
#   ./vllm-dflash-serve.sh start     # launch vLLM + DFlash
#   ./vllm-dflash-serve.sh stop
#   ./vllm-dflash-serve.sh status
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "${HERE}/cluster.env" ]] && source "${HERE}/cluster.env"

# Config (overridable via cluster.env)
VLLM_IMAGE="${VLLM_IMAGE:-gx10/vllm-ray:26.05-py3}"
CONTAINER="${DFLASH_CONTAINER:-vllm-dflash}"
PORT="${API_PORT:-8000}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/home/$(whoami)/hf-cache}"
MODEL="${DFLASH_MODEL:-Qwen/Qwen3.6-27B}"            # BF16 base
DRAFT="${DFLASH_DRAFT:-z-lab/Qwen3.6-27B-DFlash}"    # DFlash draft head
NUM_SPEC="${DFLASH_NUM_SPEC:-10}"                    # speculative tokens per step
GPU_UTIL="${DFLASH_GPU_UTIL:-0.85}"                  # fraction of the 128GB unified pool
MAX_LEN="${DFLASH_MAX_LEN:-32768}"
# Speculative decoding reserves (num_spec+1) batch slots per sequence, so the batch
# budget must cover max_num_seqs*(NUM_SPEC+1). Cap concurrency (single-user box) and
# give a roomy batch budget, or vLLM rejects the config (max_num_scheduled_tokens < 0).
MAX_SEQS="${DFLASH_MAX_SEQS:-16}"
MAX_BATCHED="${DFLASH_MAX_BATCHED_TOKENS:-8192}"
TRUST_REMOTE="${DFLASH_TRUST_REMOTE:-1}"            # new Qwen3.6 arch; 1 -> --trust-remote-code

log()  { printf '\033[0;32m==>\033[0m %s\n' "$*"; }
fail() { printf '\033[0;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

cmd="${1:-start}"
case "${cmd}" in
  stop)
    docker rm -f "${CONTAINER}" 2>/dev/null && log "vLLM+DFlash stopped." || log "Not running."
    exit 0 ;;
  status)
    if curl -s --max-time 3 "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q '"id"'; then
      log "vLLM+DFlash serving on :${PORT}"
      curl -s "http://localhost:${PORT}/v1/models"
    else
      log "vLLM+DFlash not serving."
    fi
    exit 0 ;;
esac

# DFlash is built into this vLLM (model_executor/models/qwen3_dflash.py); method "dflash".
SPEC_CONFIG=$(printf '{"method":"dflash","model":"%s","num_speculative_tokens":%s}' "${DRAFT}" "${NUM_SPEC}")

EXTRA=()
[[ "${TRUST_REMOTE}" == "1" ]] && EXTRA+=( --trust-remote-code )
# optional weight quantization (DFLASH_QUANT=fp8 -> Blackwell-native FP8, ~half the
# bytes/token). NOTE: tested 2026-06-21 and FP8 currently BREAKS with DFlash on this
# vLLM build (0.20.1+dev) — torch.compile can't pickle the FP8 graph ("Can't pickle
# <function launcher>"), so the engine crash-loops. Leave UNSET (BF16) until a newer
# vLLM fixes it; --enforce-eager would sidestep it but loses CUDA-graph speed.
[[ -n "${DFLASH_QUANT:-}" ]] && EXTRA+=( --quantization "${DFLASH_QUANT}" )

# Persist the vLLM torch.compile / CUDA-graph cache on the host so reloads skip the
# ~70s recompile, and keep server logs on the host so a crashed container is debuggable.
VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-${HF_CACHE_DIR%/}/../vllm-cache}"
LOG_DIR="${DFLASH_LOG_DIR:-${HF_CACHE_DIR%/}/../vllm-logs}"
mkdir -p "${VLLM_CACHE_DIR}" "${LOG_DIR}"
# unless-stopped: survives reboot/crash, but an explicit panel/script stop (docker rm)
# keeps it down. Set DFLASH_RESTART=no to disable.
RESTART="${DFLASH_RESTART:-unless-stopped}"

docker rm -f "${CONTAINER}" 2>/dev/null || true
log "Starting vLLM+DFlash: ${MODEL} (draft ${DRAFT}, num_spec=${NUM_SPEC}, gpu_util=${GPU_UTIL}) on :${PORT}"
docker run -d --name "${CONTAINER}" --gpus all \
    --restart "${RESTART}" \
    --ipc host \
    --ulimit memlock=-1 \
    -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
    -v "${VLLM_CACHE_DIR}:/root/.cache/vllm" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -p "${PORT}:${PORT}" \
    --entrypoint vllm \
    "${VLLM_IMAGE}" \
    serve "${MODEL}" \
        --host 0.0.0.0 --port "${PORT}" \
        --tensor-parallel-size 1 \
        --speculative-config "${SPEC_CONFIG}" \
        --gpu-memory-utilization "${GPU_UTIL}" \
        --max-model-len "${MAX_LEN}" \
        --max-num-seqs "${MAX_SEQS}" \
        --max-num-batched-tokens "${MAX_BATCHED}" \
        "${EXTRA[@]}"

# vLLM + a 54GB BF16 load takes several minutes; just confirm the container stayed up.
sleep 5
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    log "Container exited immediately. Recent logs:"
    docker logs --tail 40 "${CONTAINER}" 2>&1 || true
    fail "vLLM+DFlash failed to start. See logs above."
fi
log "Started. Loading weights (watch: docker logs -f ${CONTAINER})."
log "API: http://localhost:${PORT}/v1   (ready when /v1/models responds)"
