#!/usr/bin/env bash
# ============================================================
# 02-launch-cluster.sh
# Run from the HEAD node as your normal user (not root).
# Starts a Ray head container locally, a Ray worker container
# on the second GX10 over SSH, then launches vLLM with
# tensor parallelism across both GB10s.
#
# Usage:
#   ./02-launch-cluster.sh            # launch with cluster.env settings
#   ./02-launch-cluster.sh stop       # tear down both containers
#   MODEL=Qwen/Qwen3-72B ./02-launch-cluster.sh   # override model
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "${SCRIPT_DIR}/cluster.env" && -f "${SCRIPT_DIR}/cluster.env.example" ]]; then
    cp "${SCRIPT_DIR}/cluster.env.example" "${SCRIPT_DIR}/cluster.env"
fi
# shellcheck source=cluster.env
source "${SCRIPT_DIR}/cluster.env"
# shellcheck source=/etc/gx10-cluster.conf
source /etc/gx10-cluster.conf   # CX7_IFACE, NODE_ROLE, NODE_IP from setup script

log()  { echo -e "\n\033[1;32m==> $*\033[0m"; }
fail() { echo -e "\033[1;31m[FAIL] $*\033[0m"; exit 1; }

[[ "${NODE_ROLE}" == "head" ]] || fail "Run this from the HEAD node only."

CONTAINER_NAME="vllm-node"
SSH_WORKER="ssh ${CLUSTER_USER}@${WORKER_SSH_HOST}"

# ------------------------------------------------------------
# Teardown mode
# ------------------------------------------------------------
if [[ "${1:-}" == "stop" ]]; then
    log "Stopping cluster"
    docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
    ${SSH_WORKER} "docker rm -f ${CONTAINER_NAME}" 2>/dev/null || true
    log "Cluster stopped."
    exit 0
fi

# ------------------------------------------------------------
# Preflight
# ------------------------------------------------------------
log "Preflight checks"
ping -c 2 -W 2 "${WORKER_IP}" >/dev/null || fail "Cannot ping worker at ${WORKER_IP} over the CX7 link."
${SSH_WORKER} "true" || fail "Passwordless SSH to worker failed. Run: ssh-copy-id ${CLUSTER_USER}@${WORKER_SSH_HOST}"

WORKER_IFACE="$(${SSH_WORKER} "source /etc/gx10-cluster.conf && echo \${CX7_IFACE}")"
[[ -n "${WORKER_IFACE}" ]] || fail "Worker has no /etc/gx10-cluster.conf. Run 01-node-setup.sh worker on it first."
log "Head iface: ${CX7_IFACE} (${HEAD_IP})  |  Worker iface: ${WORKER_IFACE} (${WORKER_IP})"

# ------------------------------------------------------------
# UMA hygiene: drop page cache on both nodes so CUDA can claim memory
# ------------------------------------------------------------
# Free unified memory before load. This is hygiene, not correctness, so a
# missing sudo password must not abort the launch. Try passwordless; warn if unavailable.
log "Dropping page caches on both nodes (GB10 unified memory)"
if sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; then
    echo "Head caches dropped."
else
    echo "Skipped head cache drop (no passwordless sudo). Not required; continuing."
fi
if ${SSH_WORKER} "sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'" 2>/dev/null; then
    echo "Worker caches dropped."
else
    echo "Skipped worker cache drop (no passwordless sudo). Not required; continuing."
fi

# ------------------------------------------------------------
# Pull image on both nodes
# ------------------------------------------------------------
# Locally-built images (e.g. gx10/vllm-ray) have no registry to pull from.
# Only pull if the image is genuinely absent, and never abort on it.
log "Ensuring ${VLLM_IMAGE} is present on both nodes"
docker image inspect "${VLLM_IMAGE}" >/dev/null 2>&1 || docker pull "${VLLM_IMAGE}" || true
${SSH_WORKER} "docker image inspect ${VLLM_IMAGE} >/dev/null 2>&1 || docker pull ${VLLM_IMAGE}" || true

# ------------------------------------------------------------
# Clean any stale containers
# ------------------------------------------------------------
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
${SSH_WORKER} "docker rm -f ${CONTAINER_NAME}" 2>/dev/null || true

# ------------------------------------------------------------
# Shared docker flags
# ------------------------------------------------------------
# --network host       : Ray + NCCL need the real interfaces, no bridge NAT
# --device/-v infiniband: expose CX7 RDMA devices to NCCL
# --ulimit memlock=-1  : RDMA pinned memory registration
# --shm-size 32g       : prevents bus errors during tensor-parallel ops
common_flags() {
    local host_ip="$1" iface="$2"
    cat <<EOF
--gpus all \
--network host \
--ipc host \
--shm-size 32g \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
-v /dev/infiniband:/dev/infiniband \
--device /dev/infiniband \
-v ${HF_CACHE_DIR}:/root/.cache/huggingface \
-e HF_TOKEN=${HF_TOKEN} \
-e VLLM_HOST_IP=${host_ip} \
-e NCCL_SOCKET_IFNAME=${iface} \
-e GLOO_SOCKET_IFNAME=${iface} \
-e UCX_NET_DEVICES=${iface} \
-e NCCL_IB_DISABLE=0 \
-e NCCL_DEBUG=WARN \
-e RAY_memory_monitor_refresh_ms=0
EOF
}

# ------------------------------------------------------------
# Start Ray head (local)
# ------------------------------------------------------------
log "Starting Ray head container on ${HEAD_IP}"
# shellcheck disable=SC2046
docker run -d --name "${CONTAINER_NAME}" \
    $(common_flags "${HEAD_IP}" "${CX7_IFACE}") \
    --entrypoint /bin/bash \
    "${VLLM_IMAGE}" \
    -c "ray start --head --node-ip-address=${HEAD_IP} --port=${RAY_PORT} \
        --dashboard-host=0.0.0.0 --dashboard-port=${RAY_DASHBOARD_PORT} \
        --block"

sleep 8

# ------------------------------------------------------------
# Start Ray worker (remote over SSH)
# ------------------------------------------------------------
log "Starting Ray worker container on ${WORKER_IP}"
WORKER_FLAGS="$(common_flags "${WORKER_IP}" "${WORKER_IFACE}" | tr '\n' ' ')"
${SSH_WORKER} "docker run -d --name ${CONTAINER_NAME} \
    ${WORKER_FLAGS} \
    --entrypoint /bin/bash \
    ${VLLM_IMAGE} \
    -c 'ray start --address=${HEAD_IP}:${RAY_PORT} --node-ip-address=${WORKER_IP} --block'"

# ------------------------------------------------------------
# Wait for both GPUs to register with Ray
# ------------------------------------------------------------
log "Waiting for 2 GPUs to register with Ray"
for i in $(seq 1 30); do
    GPU_COUNT="$(docker exec "${CONTAINER_NAME}" python3 -c \
        "import ray; ray.init(address='auto', logging_level='ERROR'); print(int(ray.cluster_resources().get('GPU', 0)))" \
        2>/dev/null || echo 0)"
    if [[ "${GPU_COUNT}" == "${TENSOR_PARALLEL}" ]]; then
        log "Ray cluster ready: ${GPU_COUNT} GPUs"
        break
    fi
    [[ $i -eq 30 ]] && fail "Ray never reached ${TENSOR_PARALLEL} GPUs (saw ${GPU_COUNT}). Check: docker logs ${CONTAINER_NAME}"
    sleep 5
done

# ------------------------------------------------------------
# Launch vLLM
# ------------------------------------------------------------
log "Launching vLLM: ${MODEL} (TP=${TENSOR_PARALLEL}, max_len=${MAX_MODEL_LEN}, gpu_util=${GPU_MEM_UTIL})"
docker exec -d "${CONTAINER_NAME}" bash -c "\
    vllm serve '${MODEL}' \
        --host 0.0.0.0 \
        --port ${API_PORT} \
        --tensor-parallel-size ${TENSOR_PARALLEL} \
        --distributed-executor-backend ray \
        --gpu-memory-utilization ${GPU_MEM_UTIL} \
        --max-model-len ${MAX_MODEL_LEN} \
        > /var/log/vllm.log 2>&1"

cat <<EOF

============================================================
Cluster is starting. Model load can take several minutes
on first run while weights download to ${HF_CACHE_DIR}.

Watch progress:    docker exec ${CONTAINER_NAME} tail -f /var/log/vllm.log
Ray dashboard:     http://${HEAD_IP}:${RAY_DASHBOARD_PORT}
API endpoint:      http://<head-LAN-or-tailscale-ip>:${API_PORT}/v1
Verify:            ./03-verify.sh
Stop everything:   ./02-launch-cluster.sh stop
============================================================
EOF
