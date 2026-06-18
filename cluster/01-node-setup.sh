#!/usr/bin/env bash
# ============================================================
# 01-node-setup.sh
# Run ONCE on EACH GX10. Usage:
#   On the head box:   sudo ./01-node-setup.sh head
#   On the worker box: sudo ./01-node-setup.sh worker
#
# What it does:
#   1. Full OS update (DGX OS / Ubuntu ARM64)
#   2. Updates ConnectX-7 / mlx5 firmware support packages
#   3. Verifies Docker + NVIDIA Container Toolkit
#   4. Detects the live ConnectX-7 QSFP interface (ibdev2netdev)
#   5. Writes netplan: static IP on the CX7 link, MTU 9000
#   6. Applies sysctl tuning for the 200G link
#   7. Generates an SSH key and prints next steps
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# On a fresh clone only cluster.env.example exists. Seed the real file once.
if [[ ! -f "${SCRIPT_DIR}/cluster.env" && -f "${SCRIPT_DIR}/cluster.env.example" ]]; then
    cp "${SCRIPT_DIR}/cluster.env.example" "${SCRIPT_DIR}/cluster.env"
    echo "Created cluster.env from example. Edit it before continuing if you have not already."
fi
# shellcheck source=cluster.env
source "${SCRIPT_DIR}/cluster.env"

ROLE="${1:-}"
if [[ "${ROLE}" != "head" && "${ROLE}" != "worker" ]]; then
    echo "Usage: sudo $0 [head|worker]"
    exit 1
fi

if [[ $EUID -ne 0 ]]; then
    echo "Run with sudo."
    exit 1
fi

if [[ "${ROLE}" == "head" ]]; then
    NODE_IP="${HEAD_IP}"
else
    NODE_IP="${WORKER_IP}"
fi

log()  { echo -e "\n\033[1;32m==> $*\033[0m"; }
warn() { echo -e "\033[1;33m[WARN] $*\033[0m"; }
fail() { echo -e "\033[1;31m[FAIL] $*\033[0m"; exit 1; }

# ------------------------------------------------------------
# 1. System update
# ------------------------------------------------------------
log "Updating system packages (this can take a while on first run)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get upgrade -y

# ------------------------------------------------------------
# 2. ConnectX-7 firmware/support packages + tools
# ------------------------------------------------------------
log "Installing/updating ConnectX-7 support packages and diagnostics"
# dgx-spark-mlnx-hotplug ships on DGX OS images for GB10; install if available
apt-get install -y infiniband-diags ibverbs-utils perftest \
    net-tools jq curl git || true

if apt-cache show dgx-spark-mlnx-hotplug >/dev/null 2>&1; then
    apt-get install -y dgx-spark-mlnx-hotplug
else
    warn "dgx-spark-mlnx-hotplug not found in apt sources. If the CX7 link misbehaves after sleep/hotplug, check ASUS/NVIDIA support packages for your image."
fi

# ------------------------------------------------------------
# 3. Docker + NVIDIA Container Toolkit
# ------------------------------------------------------------
log "Verifying Docker"
if ! command -v docker >/dev/null 2>&1; then
    log "Docker not found, installing"
    curl -fsSL https://get.docker.com | sh
fi
systemctl enable --now docker

log "Verifying NVIDIA Container Toolkit"
if ! command -v nvidia-ctk >/dev/null 2>&1; then
    log "Installing NVIDIA Container Toolkit"
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -y
    apt-get install -y nvidia-container-toolkit
fi
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# Add the cluster user to the docker group so launch scripts work without sudo
if id "${CLUSTER_USER}" >/dev/null 2>&1; then
    usermod -aG docker "${CLUSTER_USER}"
fi

log "GPU sanity check"
docker run --rm --gpus all "${VLLM_IMAGE}" nvidia-smi >/dev/null 2>&1 \
    && echo "GPU visible inside container: OK" \
    || warn "Could not run nvidia-smi in ${VLLM_IMAGE} yet (image may not be pulled). Will pull during launch."

# ------------------------------------------------------------
# 4. Detect the live ConnectX-7 interface
# ------------------------------------------------------------
log "Detecting ConnectX-7 QSFP interface"
if ! command -v ibdev2netdev >/dev/null 2>&1; then
    fail "ibdev2netdev not found. infiniband-diags install failed."
fi

# Pick the first CX7 port reporting link Up. Cable must be plugged into the
# SAME port number on both boxes before running this.
CX7_IFACE="$(ibdev2netdev 2>/dev/null | awk '/Up/ {print $5; exit}')"

if [[ -z "${CX7_IFACE}" ]]; then
    echo
    ibdev2netdev || true
    fail "No CX7 port is Up. Plug the QSFP cable into the SAME port number on both GX10s, then re-run."
fi
echo "Active CX7 interface: ${CX7_IFACE}"

# Persist for the launch script
echo "CX7_IFACE=${CX7_IFACE}" > /etc/gx10-cluster.conf
echo "NODE_ROLE=${ROLE}" >> /etc/gx10-cluster.conf
echo "NODE_IP=${NODE_IP}" >> /etc/gx10-cluster.conf

# ------------------------------------------------------------
# 5. Netplan: static IP + jumbo frames on the CX7 link
# ------------------------------------------------------------
log "Writing netplan for ${CX7_IFACE} -> ${NODE_IP}/${CX7_SUBNET_PREFIX}, MTU 9000"
cat > /etc/netplan/60-cx7-cluster.yaml <<EOF
network:
  version: 2
  ethernets:
    ${CX7_IFACE}:
      dhcp4: false
      addresses:
        - ${NODE_IP}/${CX7_SUBNET_PREFIX}
      mtu: 9000
EOF
chmod 600 /etc/netplan/60-cx7-cluster.yaml
netplan apply
sleep 3
ip -br addr show "${CX7_IFACE}"

# ------------------------------------------------------------
# 6. Sysctl tuning for the 200G fabric
# ------------------------------------------------------------
log "Applying network sysctl tuning"
cat > /etc/sysctl.d/90-gx10-cluster.conf <<'EOF'
net.core.rmem_max = 536870912
net.core.wmem_max = 536870912
net.ipv4.tcp_rmem = 4096 87380 268435456
net.ipv4.tcp_wmem = 4096 65536 268435456
net.core.netdev_max_backlog = 250000
net.ipv4.tcp_mtu_probing = 1
EOF
sysctl --system >/dev/null

# Unlimited locked memory for RDMA (host side)
cat > /etc/security/limits.d/90-gx10-memlock.conf <<EOF
${CLUSTER_USER} soft memlock unlimited
${CLUSTER_USER} hard memlock unlimited
root soft memlock unlimited
root hard memlock unlimited
EOF

# ------------------------------------------------------------
# 7. Model cache dir + SSH key
# ------------------------------------------------------------
log "Creating model cache directory ${HF_CACHE_DIR}"
mkdir -p "${HF_CACHE_DIR}"
chown -R "${CLUSTER_USER}:${CLUSTER_USER}" "${HF_CACHE_DIR}"

USER_HOME="$(getent passwd "${CLUSTER_USER}" | cut -d: -f6)"
if [[ ! -f "${USER_HOME}/.ssh/id_ed25519" ]]; then
    log "Generating SSH key for ${CLUSTER_USER}"
    sudo -u "${CLUSTER_USER}" ssh-keygen -t ed25519 -N "" -f "${USER_HOME}/.ssh/id_ed25519"
fi

# ------------------------------------------------------------
# Done
# ------------------------------------------------------------
log "Node setup complete: role=${ROLE}, ip=${NODE_IP}, iface=${CX7_IFACE}"
cat <<EOF

NEXT STEPS
----------
1. Run this script on the other GX10 with the opposite role.
2. From the HEAD node, copy the SSH key to the worker (once both are set up):
       ssh-copy-id ${CLUSTER_USER}@${WORKER_IP}
3. Test the fabric from the head:
       ping -c 3 ${WORKER_IP}
4. Reboot both nodes if the kernel or firmware packages were upgraded.
5. Run ./02-launch-cluster.sh from the HEAD node.
EOF
