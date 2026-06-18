#!/usr/bin/env bash
# ============================================================
# 00-build-image.sh
# Builds a local vLLM image with Ray added back, on BOTH nodes.
# The NGC image (nvcr.io/nvidia/vllm) no longer ships the `ray`
# CLI that multi-node Ray clustering needs, so we layer it on.
#
# Run once from the HEAD node:  ./00-build-image.sh
# Then set VLLM_IMAGE in cluster.env to the LOCAL_IMAGE below
# and re-run ./02-launch-cluster.sh.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "${SCRIPT_DIR}/cluster.env" && -f "${SCRIPT_DIR}/cluster.env.example" ]]; then
    cp "${SCRIPT_DIR}/cluster.env.example" "${SCRIPT_DIR}/cluster.env"
fi
source "${SCRIPT_DIR}/cluster.env"

LOCAL_IMAGE="gx10/vllm-ray:26.04-py3"
SSH_WORKER="ssh ${CLUSTER_USER}@${WORKER_SSH_HOST}"

log()  { echo -e "\n\033[1;32m==> $*\033[0m"; }

# Clean up the dead containers from the failed launch
log "Removing any stale vllm-node containers"
docker rm -f vllm-node 2>/dev/null || true
${SSH_WORKER} "docker rm -f vllm-node" 2>/dev/null || true

# Build on the head
log "Building ${LOCAL_IMAGE} on the head node (from ${VLLM_IMAGE})"
docker build -f "${SCRIPT_DIR}/Dockerfile.ray" \
    --build-arg BASE="${VLLM_IMAGE}" \
    -t "${LOCAL_IMAGE}" "${SCRIPT_DIR}"

# Build on the worker. Copy the tiny Dockerfile over and build there so we
# don't depend on a shared registry.
log "Building ${LOCAL_IMAGE} on the worker node"
${SSH_WORKER} "mkdir -p ~/.gx10-build"
scp "${SCRIPT_DIR}/Dockerfile.ray" "${CLUSTER_USER}@${WORKER_SSH_HOST}:~/.gx10-build/Dockerfile.ray"
${SSH_WORKER} "docker build -f ~/.gx10-build/Dockerfile.ray \
    --build-arg BASE='${VLLM_IMAGE}' \
    -t '${LOCAL_IMAGE}' ~/.gx10-build"

log "Done. Both nodes now have ${LOCAL_IMAGE}"
cat <<EOF

NEXT STEPS
----------
1. Set this in cluster.env on BOTH boxes:
       VLLM_IMAGE="${LOCAL_IMAGE}"
2. Re-run the launch from the head:
       ./02-launch-cluster.sh
EOF
