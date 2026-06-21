#!/usr/bin/env bash
# llama-serve.sh - single-node llama.cpp serving on GB10, OpenAI-compatible API.
# Pairs with the vLLM cluster path: llama.cpp handles GGUF / single-node models
# (including bleeding-edge ones vLLM can't load yet); vLLM handles multi-node.
#
# Usage:
#   ./llama-serve.sh <gguf-repo> <gguf-file> [ctx]   # start serving a GGUF model
#   ./llama-serve.sh stop                            # stop the llama.cpp server
#   ./llama-serve.sh status                          # is it serving?
#
# Models are GGUF files pulled into the shared HF cache and mounted into the container.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "${HERE}/cluster.env" ]] && source "${HERE}/cluster.env"

# Config (overridable via cluster.env)
LLAMA_IMAGE="${LLAMA_IMAGE:-ghcr.io/ardge-labs/llama-cpp-dgx-spark:server}"
LLAMA_CONTAINER="${LLAMA_CONTAINER:-llama-node}"
LLAMA_PORT="${LLAMA_PORT:-8001}"          # distinct from vLLM (8000) and Open WebUI (8080)
HF_CACHE="${HF_CACHE:-/home/$(whoami)/hf-cache}"
GGUF_DIR="${GGUF_DIR:-${HF_CACHE}/gguf}"
NGL="${LLAMA_NGL:-999}"                    # offload all layers to GPU
DEFAULT_CTX="${LLAMA_CTX:-32768}"
# Performance tuning for GB10 / DGX Spark (all overridable via cluster.env). Measured
# on Qwen3.6-27B Q4_K_M: these lift generation ~9 -> ~11.6 tok/s (the memory-bandwidth
# ceiling for a dense 27B at ~273 GB/s) and speed up prompt processing. Generation is
# bandwidth-bound, so to go meaningfully faster use speculative decoding (MTP/DFlash)
# or a smaller/sparser model, not more flags.
BATCH="${LLAMA_BATCH:-2048}"               # logical batch
UBATCH="${LLAMA_UBATCH:-2048}"             # physical micro-batch; bigger = faster prefill
FLASH_ATTN="${LLAMA_FLASH_ATTN:-on}"       # on|off|auto (required for KV V-cache quant)
KV_TYPE="${LLAMA_KV_TYPE:-q8_0}"           # KV cache quant (q8_0 = quality-safe); set "none" to disable
USE_MMAP="${LLAMA_MMAP:-0}"                # 0 -> pass --no-mmap (better on GB10 unified memory)
# GGML_CUDA_ENABLE_UNIFIED_MEMORY is an oversubscribe/spill mechanism; off by default
# since models fit in the 128GB unified pool and UVM paging can hurt. Set 1 for models
# that don't fit on the device.
UNIFIED_MEM="${LLAMA_UNIFIED_MEMORY:-0}"

log() { printf '\033[0;32m==>\033[0m %s\n' "$*"; }
fail() { printf '\033[0;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

cmd="${1:-}"

if [[ "${cmd}" == "stop" ]]; then
    docker rm -f "${LLAMA_CONTAINER}" 2>/dev/null && log "llama.cpp server stopped." || log "Not running."
    exit 0
fi

if [[ "${cmd}" == "status" ]]; then
    if curl -s --max-time 3 "http://localhost:${LLAMA_PORT}/health" | grep -q "ok"; then
        log "llama.cpp serving on :${LLAMA_PORT}"
        curl -s "http://localhost:${LLAMA_PORT}/v1/models" 2>/dev/null || true
    else
        log "llama.cpp not serving."
    fi
    exit 0
fi

REPO="${1:-}"
GGUF_FILE="${2:-}"
CTX="${3:-${DEFAULT_CTX}}"
[[ -n "${GGUF_FILE}" ]] || fail "Usage: ./llama-serve.sh <gguf-repo> <gguf-file> [ctx]  (repo optional if file is already local)"

mkdir -p "${GGUF_DIR}"

# Pull the GGUF file if not already present. If no repo was given, the file must
# already be local (e.g. loading from the Library); skip the download entirely.
if [[ ! -f "${GGUF_DIR}/${GGUF_FILE}" ]]; then
    [[ -n "${REPO}" ]] || fail "GGUF file ${GGUF_FILE} is not local and no repo was given to download it from."
    log "Downloading ${GGUF_FILE} from ${REPO}..."
    if command -v hf >/dev/null 2>&1; then
        hf download "${REPO}" "${GGUF_FILE}" --local-dir "${GGUF_DIR}" \
            ${HF_TOKEN:+--token "${HF_TOKEN}"}
    else
        docker run --rm -v "${GGUF_DIR}:/out" \
            -e HF_TOKEN="${HF_TOKEN:-}" \
            python:3.12-slim bash -c \
            "pip install -q huggingface_hub && hf download '${REPO}' '${GGUF_FILE}' --local-dir /out ${HF_TOKEN:+--token \$HF_TOKEN}"
    fi
else
    log "GGUF already local: ${GGUF_FILE} (serving directly, no download)"
fi

[[ -f "${GGUF_DIR}/${GGUF_FILE}" ]] || fail "GGUF file not found after download: ${GGUF_FILE}"

# Make sure the image is present
docker image inspect "${LLAMA_IMAGE}" >/dev/null 2>&1 || {
    log "Pulling ${LLAMA_IMAGE} (first run only)..."
    docker pull "${LLAMA_IMAGE}" || fail "Could not pull ${LLAMA_IMAGE}. Check the tag or build locally."
}

# Clear any prior container
docker rm -f "${LLAMA_CONTAINER}" 2>/dev/null || true

log "Starting llama.cpp: ${GGUF_FILE} (ctx=${CTX}, ngl=${NGL}, fa=${FLASH_ATTN}, kv=${KV_TYPE:-f16}, ub=${UBATCH}) on :${LLAMA_PORT}"

# Build the perf flag list from the tuning vars above (see comments there).
LLAMA_ARGS=( -m "/models/${GGUF_FILE}" --host 0.0.0.0 --port "${LLAMA_PORT}"
             -c "${CTX}" -ngl "${NGL}" -b "${BATCH}" -ub "${UBATCH}" --metrics --jinja )
[[ "${FLASH_ATTN}" != "off" ]] && LLAMA_ARGS+=( -fa "${FLASH_ATTN}" )
if [[ -n "${KV_TYPE}" && "${KV_TYPE}" != "none" ]]; then
    LLAMA_ARGS+=( -ctk "${KV_TYPE}" -ctv "${KV_TYPE}" )   # needs flash-attn for the V side
fi
[[ "${USE_MMAP}" == "0" ]] && LLAMA_ARGS+=( --no-mmap )

# Only spill to UVM when explicitly asked; the GB10 pool is already physically unified.
DOCKER_ENV=()
[[ "${UNIFIED_MEM}" == "1" ]] && DOCKER_ENV+=( -e GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 )

# unless-stopped: auto-recover on reboot/crash; an explicit stop (docker rm) keeps it down.
RESTART="${LLAMA_RESTART:-unless-stopped}"

# The ardge-labs image IS the server; pass model + args directly (no --server flag).
docker run -d --name "${LLAMA_CONTAINER}" --gpus all \
    --restart "${RESTART}" \
    "${DOCKER_ENV[@]}" \
    -v "${GGUF_DIR}:/models" \
    -p "${LLAMA_PORT}:${LLAMA_PORT}" \
    "${LLAMA_IMAGE}" \
    "${LLAMA_ARGS[@]}"

# brief check that the container didn't immediately die (bad args, missing file)
sleep 3
if ! docker ps --format '{{.Names}}' | grep -q "^${LLAMA_CONTAINER}$"; then
    log "Container exited immediately. Recent logs:"
    docker logs --tail 30 "${LLAMA_CONTAINER}" 2>&1 || true
    fail "llama.cpp container failed to stay up. See logs above."
fi

log "Started. Watch:  docker logs -f ${LLAMA_CONTAINER}"
log "API:            http://localhost:${LLAMA_PORT}/v1"
