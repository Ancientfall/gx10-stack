#!/usr/bin/env bash
# ============================================================
# install.sh  -  set up the GX10 control panel on the HEAD node
#
#   ./install.sh
#
# Creates a venv, installs deps, adds a narrow sudoers drop-in
# so the panel can run the privileged fixes without a password,
# installs and starts the systemd service, and prints the URL.
# ============================================================
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PANEL_USER="${SUDO_USER:-$USER}"

# ------------------------------------------------------------
# Options
#   --engine native|spark   native = 02-launch-cluster.sh; spark = eugr/spark-vllm-docker
#   --orch-dir DIR          where spark-vllm-docker lives (default: <kit>/spark-vllm-docker)
#   --nodes single|cluster  single uses eugr's --solo start
#   --port N                panel port (default 8080)
# ------------------------------------------------------------
ENGINE="native"; ORCH_DIR=""; NODES=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --engine)   ENGINE="$2"; shift 2;;
        --orch-dir) ORCH_DIR="$2"; shift 2;;
        --nodes)    NODES="$2"; shift 2;;
        --port)     export GX10_PANEL_PORT="$2"; shift 2;;
        -h|--help)  echo "Usage: ./install.sh [--engine native|spark] [--orch-dir DIR] [--nodes single|cluster] [--port N]"; exit 0;;
        *)          echo "Unknown argument: $1"; exit 1;;
    esac
done

# Locate the cluster kit (cluster.env)
KIT_DIR="${GX10_KIT_DIR:-}"
if [[ -z "${KIT_DIR}" ]]; then
    for cand in "$DIR/../cluster" "$HOME/gx10-cluster" "$DIR/../gx10-cluster" "$DIR"; do
        if [[ -f "${cand}/cluster.env" || -f "${cand}/cluster.env.example" ]]; then
            KIT_DIR="$(cd "${cand}" && pwd)"; break
        fi
    done
fi
[[ -n "${KIT_DIR}" ]] || { echo "Could not find the cluster kit. Set GX10_KIT_DIR and re-run."; exit 1; }
# Seed cluster.env from the example on first install
if [[ ! -f "${KIT_DIR}/cluster.env" && -f "${KIT_DIR}/cluster.env.example" ]]; then
    cp "${KIT_DIR}/cluster.env.example" "${KIT_DIR}/cluster.env"
    echo "Created ${KIT_DIR}/cluster.env from example. Edit it to set MODEL, HF_TOKEN, etc."
fi

log() { echo -e "\n\033[1;32m==> $*\033[0m"; }

# ------------------------------------------------------------
# 0. Engine env (panel.env): native = defaults; spark = drive eugr/spark-vllm-docker
# ------------------------------------------------------------
PANEL_ENV="${DIR}/panel.env"
if [[ "${ENGINE}" == "spark" ]]; then
    ORCH_DIR="${ORCH_DIR:-${KIT_DIR}/spark-vllm-docker}"
    if [[ "${NODES}" == "single" ]]; then START_CMD="./launch-cluster.sh --solo start"; else START_CMD="./launch-cluster.sh start"; fi
    cat > "${PANEL_ENV}" <<EOF
# Written by install.sh --engine spark. Drives eugr/spark-vllm-docker.
GX10_CONTAINER=vllm_node
GX10_ORCH_DIR=${ORCH_DIR}
GX10_START_CMD=${START_CMD}
GX10_STOP_CMD=./launch-cluster.sh stop
EOF
    log "Engine: spark-vllm-docker (orchestrator dir: ${ORCH_DIR})"
else
    cat > "${PANEL_ENV}" <<EOF
# Native engine (02-launch-cluster.sh). No orchestrator overrides.
EOF
    log "Engine: native (gx10/vllm-ray)"
fi

# ------------------------------------------------------------
# 1. Python venv + deps
# ------------------------------------------------------------
log "Creating virtualenv and installing dependencies"
command -v python3 >/dev/null || { echo "python3 not found"; exit 1; }
python3 -m venv "${DIR}/.venv"
"${DIR}/.venv/bin/pip" install --quiet --upgrade pip
"${DIR}/.venv/bin/pip" install --quiet -r "${DIR}/requirements.txt"

# ------------------------------------------------------------
# 2. Sudoers drop-in: only the exact privileged commands the panel needs
# ------------------------------------------------------------
log "Installing sudoers drop-in for ${PANEL_USER}"
SUDO_FILE="/tmp/gx10-panel.sudoers"
cat > "${SUDO_FILE}" <<EOF
# Allow the GX10 panel to run only these privileged fixes without a password
${PANEL_USER} ALL=(root) NOPASSWD: /usr/sbin/netplan apply
${PANEL_USER} ALL=(root) NOPASSWD: /usr/bin/sysctl *
${PANEL_USER} ALL=(root) NOPASSWD: /usr/sbin/ip link set * mtu *
${PANEL_USER} ALL=(root) NOPASSWD: /usr/bin/mv /etc/netplan/40-cx7.yaml /etc/netplan/40-cx7.yaml.disabled
${PANEL_USER} ALL=(root) NOPASSWD: /usr/bin/sh -c sync; echo 3 > /proc/sys/vm/drop_caches
EOF
if visudo -cf "${SUDO_FILE}" >/dev/null 2>&1; then
    sudo cp "${SUDO_FILE}" /etc/sudoers.d/gx10-panel
    sudo chmod 440 /etc/sudoers.d/gx10-panel
    echo "sudoers drop-in installed."
else
    echo "WARNING: sudoers file failed validation, skipping. Optimize fixes that need root may report permission errors."
fi
rm -f "${SUDO_FILE}"

# ------------------------------------------------------------
# 3. systemd unit
# ------------------------------------------------------------
log "Installing systemd service"
UNIT="/tmp/gx10-panel.service"
PANEL_PORT="${GX10_PANEL_PORT:-8080}"
sed -e "s#__USER__#${PANEL_USER}#g" \
    -e "s#__DIR__#${DIR}#g" \
    -e "s#__KIT_DIR__#${KIT_DIR}#g" \
    -e "s#__PANEL_PORT__#${PANEL_PORT}#g" \
    "${DIR}/gx10-panel.service" > "${UNIT}"
sudo cp "${UNIT}" /etc/systemd/system/gx10-panel.service
rm -f "${UNIT}"
sudo systemctl daemon-reload
sudo systemctl enable --now gx10-panel.service
sleep 2
sudo systemctl --no-pager --lines=0 status gx10-panel.service || true

# ------------------------------------------------------------
# 4. Access URL
# ------------------------------------------------------------
log "Done"
TS_IP="$(tailscale ip -4 2>/dev/null | head -n1 || true)"
echo "Kit dir:  ${KIT_DIR}"
echo "Engine:   ${ENGINE}  (env file: ${PANEL_ENV})"
echo "Local:    http://localhost:${PANEL_PORT}"
if [[ -n "${TS_IP}" ]]; then
    echo "Tailscale: http://${TS_IP}:${PANEL_PORT}  (reachable from your Mac/phone)"
    echo
    echo "Optional, for a clean HTTPS URL over Tailscale:"
    echo "    sudo tailscale serve --bg ${PANEL_PORT}"
fi
echo
echo "Manage:   sudo systemctl restart gx10-panel   |   journalctl -u gx10-panel -f"
