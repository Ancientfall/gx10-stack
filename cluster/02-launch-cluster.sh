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
valid_repo() { [[ "$1" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]]; }
valid_uint() { [[ "$1" =~ ^[0-9]+$ && "$1" -ge "$2" && "$1" -le "$3" ]]; }
valid_gpu_util() { [[ "$1" =~ ^[0-9]+(\.[0-9]+)?$ ]] && awk -v v="$1" 'BEGIN { exit !(v >= 0.1 && v <= 0.95) }'; }
valid_image_ref() { [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._/-]*(:[A-Za-z0-9][A-Za-z0-9._-]*)?$ ]]; }
valid_abs_path() { [[ "$1" == /* && ! "$1" =~ [[:space:]\'\"\\\`\$\;\&\|\<\>\*\?\(\)\[\]\{\}\!] ]]; }
quote_cmd() {
    local out="" arg
    for arg in "$@"; do
        printf -v out '%s%q ' "${out}" "${arg}"
    done
    printf '%s' "${out}"
}

[[ "${NODE_ROLE}" == "head" ]] || fail "Run this from the HEAD node only."

CONTAINER_NAME="vllm-node"
SSH_TARGET="${CLUSTER_USER}@${WORKER_SSH_HOST}"

valid_image_ref "${VLLM_IMAGE}" || fail "Invalid VLLM_IMAGE: ${VLLM_IMAGE}"
valid_abs_path "${HF_CACHE_DIR}" || fail "Invalid HF_CACHE_DIR: ${HF_CACHE_DIR}"
valid_uint "${API_PORT}" 1 65535 || fail "Invalid API_PORT: ${API_PORT}"
valid_uint "${RAY_PORT}" 1 65535 || fail "Invalid RAY_PORT: ${RAY_PORT}"
valid_uint "${RAY_DASHBOARD_PORT}" 1 65535 || fail "Invalid RAY_DASHBOARD_PORT: ${RAY_DASHBOARD_PORT}"
valid_uint "${TENSOR_PARALLEL}" 1 8 || fail "Invalid TENSOR_PARALLEL: ${TENSOR_PARALLEL}"
valid_uint "${MAX_MODEL_LEN}" 512 1048576 || fail "Invalid MAX_MODEL_LEN: ${MAX_MODEL_LEN}"
valid_gpu_util "${GPU_MEM_UTIL}" || fail "Invalid GPU_MEM_UTIL: ${GPU_MEM_UTIL}"
[[ -z "${MODEL:-}" ]] || valid_repo "${MODEL}" || fail "Invalid MODEL repo id: ${MODEL}"

# ------------------------------------------------------------
# Teardown mode
# ------------------------------------------------------------
if [[ "${1:-}" == "stop" ]]; then
    log "Stopping cluster"
    docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
    ssh "${SSH_TARGET}" docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
    log "Cluster stopped."
    exit 0
fi

# ------------------------------------------------------------
# Preflight
# ------------------------------------------------------------
log "Preflight checks"
ping -c 2 -W 2 "${WORKER_IP}" >/dev/null || fail "Cannot ping worker at ${WORKER_IP} over the CX7 link."
ssh "${SSH_TARGET}" true || fail "Passwordless SSH to worker failed. Run: ssh-copy-id ${CLUSTER_USER}@${WORKER_SSH_HOST}"

WORKER_IFACE="$(ssh "${SSH_TARGET}" "source /etc/gx10-cluster.conf && echo \${CX7_IFACE}")"
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
if ssh "${SSH_TARGET}" "sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'" 2>/dev/null; then
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
remote_pull="$(quote_cmd docker image inspect "${VLLM_IMAGE}") >/dev/null 2>&1 || $(quote_cmd docker pull "${VLLM_IMAGE}")"
ssh "${SSH_TARGET}" "${remote_pull}" || true

# ------------------------------------------------------------
# Clean any stale containers
# ------------------------------------------------------------
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
ssh "${SSH_TARGET}" docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

# ------------------------------------------------------------
# Shared docker flags
# ------------------------------------------------------------
# --network host        : Ray + NCCL need the real interfaces, no bridge NAT
# --device /dev/infiniband + --cap-add=IPC_LOCK : RDMA verbs need the devices AND
#                         the capability to pin memory (memlock ulimit alone isn't enough)
# --ipc host            : shared memory for tensor-parallel ops (replaces --shm-size)
# NCCL block            : GB10 has no GPUDirect RDMA, so NCCL_NET_PLUGIN=none forces the
#                         native IB-verbs path; pin NCCL_IB_HCA/GID or it falls back to TCP.
#                         MERGE_NICS + QPS tuning give dual-rail ~24 GB/s (vs ~14 single).
common_flags_array() {
    local -n _out="$1"
    local host_ip="$2" iface="$3"
    _out=(
        --gpus all
        --network host
        --ipc host
        --ulimit memlock=-1
        --ulimit stack=67108864
        --ulimit nofile=1048576
        --cap-add=IPC_LOCK
        --device /dev/infiniband
        -v "${HF_CACHE_DIR}:/root/.cache/huggingface"
        -e "HF_TOKEN=${HF_TOKEN:-}"
        -e "VLLM_HOST_IP=${host_ip}"
        -e "NCCL_SOCKET_IFNAME=${iface}"
        -e "GLOO_SOCKET_IFNAME=${iface}"
        -e "UCX_NET_DEVICES=${iface}"
        -e NCCL_IB_DISABLE=0
        -e NCCL_NET_PLUGIN=none
        -e "NCCL_IB_HCA=${IB_HCA:-rocep1s0f1}"
        -e "NCCL_IB_GID_INDEX=${IB_GID_INDEX:-3}"
        -e "NCCL_IB_MERGE_NICS=${NCCL_IB_MERGE_NICS:-1}"
        -e "NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-1}"
        -e "NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION:-4}"
        -e "NCCL_IB_SPLIT_DATA_ON_QPS=${NCCL_IB_SPLIT_DATA_ON_QPS:-1}"
        -e "NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}"
        -e "NCCL_DEBUG=${NCCL_DEBUG:-WARN}"
        -e RAY_memory_monitor_refresh_ms=0
    )
}

# ------------------------------------------------------------
# Start Ray head (local)
# ------------------------------------------------------------
log "Starting Ray head container on ${HEAD_IP}"
HEAD_FLAGS=()
common_flags_array HEAD_FLAGS "${HEAD_IP}" "${CX7_IFACE}"
docker run -d --name "${CONTAINER_NAME}" \
    "${HEAD_FLAGS[@]}" \
    --entrypoint /bin/bash \
    "${VLLM_IMAGE}" \
    -lc 'exec ray start --head --node-ip-address="$1" --port="$2" --dashboard-host=0.0.0.0 --dashboard-port="$3" --block' \
    ray-head "${HEAD_IP}" "${RAY_PORT}" "${RAY_DASHBOARD_PORT}"

sleep 8

# ------------------------------------------------------------
# Start Ray worker (remote over SSH)
# ------------------------------------------------------------
log "Starting Ray worker container on ${WORKER_IP}"
WORKER_FLAGS=()
common_flags_array WORKER_FLAGS "${WORKER_IP}" "${WORKER_IFACE}"
remote_run="$(quote_cmd docker run -d --name "${CONTAINER_NAME}" \
    "${WORKER_FLAGS[@]}" \
    --entrypoint /bin/bash \
    "${VLLM_IMAGE}" \
    -lc 'exec ray start --address="$1:$2" --node-ip-address="$3" --block' \
    ray-worker "${HEAD_IP}" "${RAY_PORT}" "${WORKER_IP}")"
ssh "${SSH_TARGET}" "${remote_run}"

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
# ------------------------------------------------------------
# Launch vLLM (unless starting idle)
# ------------------------------------------------------------
# Cluster comes up idle if MODEL is empty or NO_AUTOLOAD=1 is set.
# Ray and both nodes are ready; load a model from the panel when you want one.
if [[ "${NO_AUTOLOAD:-0}" == "1" || -z "${MODEL:-}" ]]; then
    cat <<EOF

============================================================
Cluster is up and IDLE (Ray ready on ${TENSOR_PARALLEL} GPU(s), no model loaded).
Load a model whenever you want from the panel, or:
    MODEL='openai/gpt-oss-120b' ./02-launch-cluster.sh cluster

Ray dashboard:     http://${HEAD_IP}:${RAY_DASHBOARD_PORT}
Stop everything:   ./02-launch-cluster.sh stop
============================================================
EOF
    exit 0
fi

log "Launching vLLM: ${MODEL} (TP=${TENSOR_PARALLEL}, max_len=${MAX_MODEL_LEN}, gpu_util=${GPU_MEM_UTIL})"
docker exec -d "${CONTAINER_NAME}" bash -lc \
    'exec vllm serve "$1" --host 0.0.0.0 --port "$2" --tensor-parallel-size "$3" --distributed-executor-backend ray --gpu-memory-utilization "$4" --max-model-len "$5" > /var/log/vllm.log 2>&1' \
    vllm-serve "${MODEL}" "${API_PORT}" "${TENSOR_PARALLEL}" "${GPU_MEM_UTIL}" "${MAX_MODEL_LEN}"

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
