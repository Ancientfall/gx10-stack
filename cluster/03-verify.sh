#!/usr/bin/env bash
# ============================================================
# 03-verify.sh
# Run from the HEAD node. Checks fabric, Ray, and the API.
#   ./03-verify.sh           # standard checks
#   ./03-verify.sh --bw      # also run ib_write_bw bandwidth test
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/cluster.env"
source /etc/gx10-cluster.conf

PASS=0; FAILED=0
ok()   { echo -e "\033[1;32m[PASS]\033[0m $*"; ((PASS++)); }
bad()  { echo -e "\033[1;31m[FAIL]\033[0m $*"; ((FAILED++)); }
info() { echo -e "\033[1;36m[INFO]\033[0m $*"; }

SSH_WORKER="ssh ${CLUSTER_USER}@${WORKER_SSH_HOST}"

echo "=== 1. ConnectX-7 link state ==="
LINK="$(ibdev2netdev 2>/dev/null | grep -c '(Up)' || true)"
[[ "${LINK}" -ge 1 ]] && ok "CX7 link Up on head (${CX7_IFACE})" || bad "No CX7 link Up on head"

MTU="$(cat /sys/class/net/${CX7_IFACE}/mtu 2>/dev/null || echo 0)"
[[ "${MTU}" == "9000" ]] && ok "MTU 9000 on ${CX7_IFACE}" || bad "MTU is ${MTU}, expected 9000"

echo
echo "=== 2. Fabric connectivity ==="
if ping -c 3 -W 2 -M do -s 8972 "${WORKER_IP}" >/dev/null 2>&1; then
    ok "Jumbo-frame ping to worker ${WORKER_IP}"
elif ping -c 3 -W 2 "${WORKER_IP}" >/dev/null 2>&1; then
    bad "Worker reachable but jumbo frames failing; check MTU on both ends"
else
    bad "Worker ${WORKER_IP} unreachable over CX7 link"
fi

if [[ "${1:-}" == "--bw" ]]; then
    echo
    echo "=== 2b. RDMA verbs bandwidth (ib_write_bw, ~10s) ==="
    DEV="$(ibdev2netdev | awk '/\(Up\)/{print $1; exit}')"
    ${SSH_WORKER} "nohup ib_write_bw -d ${DEV} >/dev/null 2>&1 &" </dev/null || true
    sleep 2
    ib_write_bw -d "${DEV}" "${WORKER_IP}" || bad "ib_write_bw failed"
    ${SSH_WORKER} "pkill ib_write_bw" </dev/null 2>/dev/null || true
    info "Raw verbs BW only. On GB10 (no GPUDirect RDMA) this does NOT equal NCCL throughput -- see 2c."

    echo
    echo "=== 2c. NCCL all-reduce over RDMA (authoritative; ~20s) ==="
    # The real test of what TP serving sees: a 2-rank all-reduce. Confirms NET/IB
    # (not TCP fallback) and measures busbw (single-rail ~14, dual-rail ~24 GB/s).
    if docker ps --format '{{.Names}}' | grep -q '^vllm-node$'; then
        AR='import time,datetime,torch,torch.distributed as dist
dist.init_process_group(backend="nccl",timeout=datetime.timedelta(seconds=45))
r=dist.get_rank();torch.cuda.set_device(0)
x=torch.ones(1024*1024*1024//4,device="cuda")
for _ in range(5): dist.all_reduce(x)
torch.cuda.synchronize();N=20;t=time.time()
for _ in range(N): dist.all_reduce(x)
torch.cuda.synchronize();bw=x.numel()*4/((time.time()-t)/N)/1e9
print(f"NCCL_BUSBW={bw:.2f}") if r==0 else None
dist.destroy_process_group()'
        docker exec -i vllm-node bash -c "cat > /tmp/_ar.py" <<<"$AR"
        ${SSH_WORKER} "docker exec -i vllm-node bash -c 'cat > /tmp/_ar.py'" <<<"$AR"
        AR_ENV="-e WORLD_SIZE=2 -e MASTER_ADDR=${HEAD_IP} -e MASTER_PORT=29600 -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=NET"
        ${SSH_WORKER} "docker exec -e RANK=1 ${AR_ENV} vllm-node python3 /tmp/_ar.py" </dev/null >/tmp/_ar_w.log 2>&1 &
        sleep 3
        timeout 90 docker exec -e RANK=0 ${AR_ENV} vllm-node python3 /tmp/_ar.py >/tmp/_ar_h.log 2>&1
        wait
        BW="$(grep -oE 'NCCL_BUSBW=[0-9.]+' /tmp/_ar_h.log | cut -d= -f2)"
        if grep -q 'Using network IB' /tmp/_ar_h.log; then
            ok "NCCL on NET/IB (RDMA), busbw ${BW:-?} GB/s  (single-rail ~14, dual-rail ~24)"
        else
            bad "NCCL NOT on NET/IB (likely TCP fallback), busbw ${BW:-?} GB/s -- check NCCL_IB_HCA / NCCL_NET_PLUGIN=none"
        fi
        docker exec vllm-node rm -f /tmp/_ar.py 2>/dev/null
        ${SSH_WORKER} "docker exec vllm-node rm -f /tmp/_ar.py" </dev/null 2>/dev/null || true
        rm -f /tmp/_ar_h.log /tmp/_ar_w.log
    else
        info "vllm-node not running; start the cluster (02-launch-cluster.sh) before the NCCL bw test."
    fi
fi

echo
echo "=== 3. Ray cluster ==="
GPU_COUNT="$(docker exec vllm-node python3 -c \
    "import ray; ray.init(address='auto', logging_level='ERROR'); print(int(ray.cluster_resources().get('GPU', 0)))" \
    2>/dev/null || echo 0)"
[[ "${GPU_COUNT}" == "${TENSOR_PARALLEL}" ]] \
    && ok "Ray sees ${GPU_COUNT}/${TENSOR_PARALLEL} GPUs" \
    || bad "Ray sees ${GPU_COUNT}/${TENSOR_PARALLEL} GPUs (run: docker logs vllm-node)"

echo
echo "=== 4. vLLM API ==="
MODELS_JSON="$(curl -s --max-time 5 "http://localhost:${API_PORT}/v1/models" || true)"
if echo "${MODELS_JSON}" | jq -e '.data[0].id' >/dev/null 2>&1; then
    SERVED="$(echo "${MODELS_JSON}" | jq -r '.data[0].id')"
    ok "API serving model: ${SERVED}"

    info "Running inference smoke test..."
    REPLY="$(curl -s --max-time 120 "http://localhost:${API_PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\": \"${SERVED}\", \"max_tokens\": 32, \"messages\": [{\"role\": \"user\", \"content\": \"Reply with the single word: online\"}]}" \
        | jq -r '.choices[0].message.content' 2>/dev/null || true)"
    [[ -n "${REPLY}" && "${REPLY}" != "null" ]] \
        && ok "Inference response: ${REPLY}" \
        || bad "Chat completion returned nothing; model may still be loading (tail /var/log/vllm.log in container)"
else
    bad "API not responding on :${API_PORT}. Model may still be loading. Check: docker exec vllm-node tail -50 /var/log/vllm.log"
fi

echo
echo "============================================"
echo "Results: ${PASS} passed, ${FAILED} failed"
echo "============================================"
exit "${FAILED}"
