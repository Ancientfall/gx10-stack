#!/usr/bin/env bash
# ============================================================
# 02-launch-cluster.sh
# Run from the HEAD node as your normal user (not root).
#
# Two-node (default): starts a Ray head container locally and a Ray
# worker container on the second GX10 over SSH, then launches vLLM with
# tensor parallelism across both GB10s.
#
# Single-node: starts only the head container and serves on the head GPU
# (multiprocessing backend, no Ray worker), leaving the second box idle.
# Best for models that fit on one GB10.
#
# Usage:
#   ./02-launch-cluster.sh            # mode from TENSOR_PARALLEL (1 = single, else two-node)
#   ./02-launch-cluster.sh single     # force single-node (head only, worker idle)
#   ./02-launch-cluster.sh cluster    # force two-node
#   ./02-launch-cluster.sh stop       # tear down containers on both boxes
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
SSH_WORKER="ssh -o BatchMode=yes -o ConnectTimeout=5 ${CLUSTER_USER}@${WORKER_SSH_HOST}"

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
# Node mode: explicit arg wins, otherwise derive from TENSOR_PARALLEL
#   single  -> head GPU only, mp backend, worker left idle
#   cluster -> both GB10s, Ray, tensor parallel across the pair
# ------------------------------------------------------------
TP="${TENSOR_PARALLEL:-2}"
case "${1:-}" in
    single)  MODE="single" ;;
    cluster) MODE="cluster" ;;
    "")      if [[ "${TP}" == "1" ]]; then MODE="single"; else MODE="cluster"; fi ;;
    *)       fail "Unknown argument '${1}'. Use: single | cluster | stop" ;;
esac

if [[ "${MODE}" == "single" ]]; then
    TP_SIZE=1; BACKEND="mp"; EXPECT_GPUS=1
else
    TP_SIZE="${TENSOR_PARALLEL}"; BACKEND="ray"; EXPECT_GPUS="${TENSOR_PARALLEL}"
fi
log "Mode: ${MODE} (TP=${TP_SIZE}, backend=${BACKEND})"

# ------------------------------------------------------------
# Preflight
# ------------------------------------------------------------
log "Preflight checks"
if [[ "${MODE}" == "cluster" ]]; then
    ping -c 2 -W 2 "${WORKER_IP}" >/dev/null || fail "Cannot ping worker at ${WORKER_IP} over the CX7 link."
    ${SSH_WORKER} "true" || fail "Passwordless SSH to worker failed. Run: ssh-copy-id ${CLUSTER_USER}@${WORKER_SSH_HOST}"

    WORKER_IFACE="$(${SSH_WORKER} "source /etc/gx10-cluster.conf && echo \${CX7_IFACE}")"
    [[ -n "${WORKER_IFACE}" ]] || fail "Worker has no /etc/gx10-cluster.conf. Run 01-node-setup.sh worker on it first."
    log "Head iface: ${CX7_IFACE} (${HEAD_IP})  |  Worker iface: ${WORKER_IFACE} (${WORKER_IP})"
else
    log "Single-node mode: serving on the head GPU only (${HEAD_IP}); worker left idle."
fi

# ------------------------------------------------------------
# UMA hygiene: drop page cache so CUDA can claim memory
# ------------------------------------------------------------
# Free unified memory before load. This is hygiene, not correctness, so a
# missing sudo password must not abort the launch. Try passwordless; warn if unavailable.
log "Dropping page caches (GB10 unified memory)"
if sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; then
    echo "Head caches dropped."
else
    echo "Skipped head cache drop (no passwordless sudo). Not required; continuing."
fi
if [[ "${MODE}" == "cluster" ]]; then
    if ${SSH_WORKER} "sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'" 2>/dev/null; then
        echo "Worker caches dropped."
    else
        echo "Skipped worker cache drop (no passwordless sudo). Not required; continuing."
    fi
fi

# ------------------------------------------------------------
# Pull image
# ------------------------------------------------------------
# Locally-built images (e.g. gx10/vllm-ray) have no registry to pull from.
# Only pull if the image is genuinely absent, and never abort on it.
log "Ensuring ${VLLM_IMAGE} is present"
docker image inspect "${VLLM_IMAGE}" >/dev/null 2>&1 || docker pull "${VLLM_IMAGE}" || true
if [[ "${MODE}" == "cluster" ]]; then
    ${SSH_WORKER} "docker image inspect ${VLLM_IMAGE} >/dev/null 2>&1 || docker pull ${VLLM_IMAGE}" || true
fi

# ------------------------------------------------------------
# Clean any stale containers
# ------------------------------------------------------------
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
# Always clear a stale worker container (best effort). In single mode this
# frees the worker so it can idle; in cluster mode it makes room for a fresh start.
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
# Start the head container (local)
# ------------------------------------------------------------
# Runs the Ray head, which also keeps the container alive. In cluster mode
# the worker joins this head; in single mode vLLM uses the mp backend and
# ignores Ray, but the head keeps the container up for `docker exec`.
log "Starting head container on ${HEAD_IP}"
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
# Start Ray worker (cluster mode only)
# ------------------------------------------------------------
if [[ "${MODE}" == "cluster" ]]; then
    log "Starting Ray worker container on ${WORKER_IP}"
    WORKER_FLAGS="$(common_flags "${WORKER_IP}" "${WORKER_IFACE}" | tr '\n' ' ')"
    ${SSH_WORKER} "docker run -d --name ${CONTAINER_NAME} \
        ${WORKER_FLAGS} \
        --entrypoint /bin/bash \
        ${VLLM_IMAGE} \
        -c 'ray start --address=${HEAD_IP}:${RAY_PORT} --node-ip-address=${WORKER_IP} --block'"
fi

# ------------------------------------------------------------
# Wait for the expected GPUs to register with Ray
# ------------------------------------------------------------
log "Waiting for ${EXPECT_GPUS} GPU(s) to register with Ray"
for i in $(seq 1 30); do
    GPU_COUNT="$(docker exec "${CONTAINER_NAME}" python3 -c \
        "import ray; ray.init(address='auto', logging_level='ERROR'); print(int(ray.cluster_resources().get('GPU', 0)))" \
        2>/dev/null || echo 0)"
    if [[ "${GPU_COUNT}" == "${EXPECT_GPUS}" ]]; then
        log "Ray ready: ${GPU_COUNT} GPU(s)"
        break
    fi
    [[ $i -eq 30 ]] && fail "Ray never reached ${EXPECT_GPUS} GPU(s) (saw ${GPU_COUNT}). Check: docker logs ${CONTAINER_NAME}"
    sleep 5
done

# ------------------------------------------------------------
# Launch vLLM
# ------------------------------------------------------------
log "Launching vLLM: ${MODEL} (TP=${TP_SIZE}, backend=${BACKEND}, max_len=${MAX_MODEL_LEN}, gpu_util=${GPU_MEM_UTIL})"
docker exec -d "${CONTAINER_NAME}" bash -c "\
    vllm serve '${MODEL}' \
        --host 0.0.0.0 \
        --port ${API_PORT} \
        --tensor-parallel-size ${TP_SIZE} \
        --distributed-executor-backend ${BACKEND} \
        --gpu-memory-utilization ${GPU_MEM_UTIL} \
        --max-model-len ${MAX_MODEL_LEN} \
        > /var/log/vllm.log 2>&1"

cat <<EOF

============================================================
Cluster is starting in ${MODE} mode (TP=${TP_SIZE}, backend=${BACKEND}).
Model load can take several minutes on first run while weights
download to ${HF_CACHE_DIR}.

Watch progress:    docker exec ${CONTAINER_NAME} tail -f /var/log/vllm.log
Ray dashboard:     http://${HEAD_IP}:${RAY_DASHBOARD_PORT}
API endpoint:      http://<head-LAN-or-tailscale-ip>:${API_PORT}/v1
Verify:            ./03-verify.sh
Stop everything:   ./02-launch-cluster.sh stop
============================================================
EOF

if [[ "${MODE}" == "single" ]]; then
    echo "Single-node mode: the worker box is idle. './03-verify.sh' will flag the"
    echo "fabric / 2-GPU checks as down — that is expected. Switch to two-node with"
    echo "'./02-launch-cluster.sh cluster' (or set TENSOR_PARALLEL=2 in cluster.env)."
fi
