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
    echo "=== 2b. RDMA bandwidth (ib_write_bw, ~10s) ==="
    ${SSH_WORKER} "nohup ib_write_bw -d \$(ibdev2netdev | awk '/(Up)/{print \$1; exit}') >/dev/null 2>&1 &" || true
    sleep 2
    ib_write_bw -d "$(ibdev2netdev | awk '/\(Up\)/{print $1; exit}')" "${WORKER_IP}" || bad "ib_write_bw failed"
    info "Expect roughly 180+ Gb/s average on a healthy 200G DAC link"
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
