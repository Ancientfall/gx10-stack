#!/usr/bin/env bash
# ============================================================
# cleanup.sh - report and reclaim disk used by vLLM images,
# Docker build cache, and the HuggingFace model cache.
#
# Safe by default: with no flags it only REPORTS.
#   ./cleanup.sh             # show what's using space on this node
#   ./cleanup.sh --prune     # + remove stopped containers, dangling images, build cache
#   ./cleanup.sh --prune -y  # no prompts
#   ./cleanup.sh --prune -c  # also run on the worker over SSH
# Bigger, node-specific wins (old images / cached models) are printed, not auto-deleted.
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster.env
[[ -f "${SCRIPT_DIR}/cluster.env" ]] && source "${SCRIPT_DIR}/cluster.env" || true
HF_CACHE_DIR="${HF_CACHE_DIR:-/data/hf-cache}"
VLLM_IMAGE="${VLLM_IMAGE:-}"

PRUNE=0; YES=0; WORKER=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prune)               PRUNE=1; shift;;
        -y|--yes)              YES=1; shift;;
        -c|--worker|--all-nodes) WORKER=1; shift;;
        -h|--help)             sed -n '2,14p' "$0"; exit 0;;
        *)                     echo "Unknown argument: $1"; exit 1;;
    esac
done

say()     { printf "\n\033[1;36m== %s\033[0m\n" "$*"; }
confirm() { [[ "${YES}" == "1" ]] && return 0; local a; read -rp "$1 [y/N] " a || true; [[ "$a" =~ ^[Yy] ]]; }

report() {
    say "Disk (/) on $(hostname)"
    df -h / | awk 'NR==1 || /\/$/'
    say "Docker usage"
    docker system df 2>/dev/null || echo "(docker not available)"
    say "Images"
    docker images --format '{{.Repository}}:{{.Tag}}\t{{.Size}}' 2>/dev/null | sort || true
    say "Model cache: ${HF_CACHE_DIR}/hub"
    if [[ -d "${HF_CACHE_DIR}/hub" ]]; then
        du -h -d1 "${HF_CACHE_DIR}/hub" 2>/dev/null | sort -h | tail -25
    else
        echo "(none found)"
    fi
    [[ -n "${VLLM_IMAGE}" ]] && { echo; echo "In use (keep this one): VLLM_IMAGE=${VLLM_IMAGE}"; }
}

prune() {
    say "Safe reclaim: stopped containers, dangling images, build cache"
    confirm "Proceed on $(hostname)?" || { echo "skipped."; return; }
    docker container prune -f || true
    docker image prune -f || true
    docker builder prune -f || true
    say "After:"
    docker system df 2>/dev/null || true
    cat <<EOF

Bigger, manual wins (review first, then run):
  old images:    docker rmi <repo:tag>            # keep ${VLLM_IMAGE:-your current image}
  cached models: rm -rf ${HF_CACHE_DIR}/hub/models--Org--Name
  spark caches:  rm -rf ~/.cache/vllm ~/.cache/flashinfer ~/.triton
EOF
}

report
[[ "${PRUNE}" == "1" ]] && prune

if [[ "${WORKER}" == "1" ]]; then
    TARGET="${CLUSTER_USER:-$USER}@${WORKER_SSH_HOST:-}"
    if [[ -n "${WORKER_SSH_HOST:-}" ]]; then
        say "Worker: ${TARGET}"
        RARGS=(); [[ "${PRUNE}" == "1" ]] && RARGS+=(--prune); [[ "${YES}" == "1" ]] && RARGS+=(-y)
        ssh -o BatchMode=yes -o ConnectTimeout=6 "${TARGET}" \
            "cd ~/gx10-stack/cluster 2>/dev/null && ./cleanup.sh ${RARGS[*]:-}" \
            || echo "Could not run cleanup on the worker (check SSH / repo path)."
    else
        echo "WORKER_SSH_HOST not set in cluster.env; skipping worker."
    fi
fi
