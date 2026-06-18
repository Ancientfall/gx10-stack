#!/usr/bin/env bash
# ============================================================
# 04-make-persistent.sh
# Makes the cluster survive a reboot. Run on EACH box with its role:
#   sudo ./04-make-persistent.sh head     # on myspark
#   sudo ./04-make-persistent.sh worker   # on myspark2
#
# What it does:
#   1. Writes a real netplan so the CX7 fabric IP + MTU 9000 persist
#   2. Re-asserts /etc/gx10-cluster.conf from the live interface
#   3. Sets Docker to start on boot
#   4. Verifies the vLLM/Ray containers carry a restart policy
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "${SCRIPT_DIR}/cluster.env" && -f "${SCRIPT_DIR}/cluster.env.example" ]]; then
    cp "${SCRIPT_DIR}/cluster.env.example" "${SCRIPT_DIR}/cluster.env"
fi
source "${SCRIPT_DIR}/cluster.env"

ROLE="${1:-}"
[[ "${ROLE}" == "head" || "${ROLE}" == "worker" ]] || { echo "Usage: sudo $0 [head|worker]"; exit 1; }
[[ $EUID -eq 0 ]] || { echo "Run with sudo."; exit 1; }

if [[ "${ROLE}" == "head" ]]; then NODE_IP="${HEAD_IP}"; else NODE_IP="${WORKER_IP}"; fi

log()  { echo -e "\n\033[1;32m==> $*\033[0m"; }
warn() { echo -e "\033[1;33m[WARN] $*\033[0m"; }

# 1. Detect the live CX7 interface (robust match on "(Up)")
CX7_IFACE="$(ibdev2netdev 2>/dev/null | awk '/\(Up\)/ {print $(NF-1); exit}')"
[[ -n "${CX7_IFACE}" ]] || { echo "No CX7 interface is Up. Bring the link up before persisting."; exit 1; }
log "Live CX7 interface: ${CX7_IFACE} -> ${NODE_IP}/${CX7_SUBNET_PREFIX:-24}"

# 2. Persistent netplan. IPv6 disabled, MTU 9000, static IP -> survives reboot.
log "Writing persistent netplan (60-cx7-cluster.yaml)"
cat > /etc/netplan/60-cx7-cluster.yaml <<EOF
network:
  version: 2
  ethernets:
    ${CX7_IFACE}:
      dhcp4: false
      dhcp6: false
      link-local: []
      accept-ra: false
      addresses:
        - ${NODE_IP}/${CX7_SUBNET_PREFIX:-24}
      mtu: 9000
EOF
chmod 600 /etc/netplan/60-cx7-cluster.yaml

# Archive the NVIDIA default if it lingers, to avoid the conflict from earlier
if [[ -f /etc/netplan/40-cx7.yaml ]]; then
    mv /etc/netplan/40-cx7.yaml /etc/netplan/40-cx7.yaml.disabled
    echo "Archived stale 40-cx7.yaml"
fi
netplan apply
sleep 2
ip -br addr show "${CX7_IFACE}"

# 3. Durable node conf
log "Writing /etc/gx10-cluster.conf"
cat > /etc/gx10-cluster.conf <<EOF
CX7_IFACE=${CX7_IFACE}
NODE_ROLE=${ROLE}
NODE_IP=${NODE_IP}
EOF
cat /etc/gx10-cluster.conf

# 4. Docker on boot
log "Enabling Docker on boot"
systemctl enable docker >/dev/null 2>&1 || true

# 5. Make running containers restart on boot (if up). Idempotent.
for c in vllm-node open-webui; do
    if docker inspect "${c}" >/dev/null 2>&1; then
        docker update --restart unless-stopped "${c}" >/dev/null 2>&1 \
            && echo "Set restart policy on ${c}" \
            || warn "Could not set restart policy on ${c}"
    fi
done

log "Persistence applied for ${ROLE}."
cat <<EOF

NOTE
----
The vLLM container restarts on boot, but the Ray *cluster* (head+worker join)
is established by 02-launch-cluster.sh, not by a single container. After a full
power cycle of both boxes, re-run the launch once:
    ./02-launch-cluster.sh
The panel's Start button does the same thing from the browser.

To verify persistence without a reboot, you can run:
    sudo netplan get ${CX7_IFACE}
EOF
