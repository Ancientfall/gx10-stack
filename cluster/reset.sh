#!/usr/bin/env bash
# ============================================================
# reset.sh - tear the GX10 cluster software back to a clean slate.
# Removes containers, the panel (service/sudoers/venv/panel.env/db), the
# config/netplan/sysctl/limits files node-setup wrote, a nested repo clone,
# vLLM images + build cache, spark build caches, and (by default) the HF
# model cache. Does NOT touch the OS. Re-run ./deploy.sh for a fresh start.
#
#   ./reset.sh                 # nuke this node (asks for confirmation)
#   ./reset.sh -y              # no prompt
#   ./reset.sh --keep-models   # keep the HF model cache (skip the big re-download)
#   ./reset.sh --keep-images   # keep docker images
#   ./reset.sh --volumes       # also prune unreferenced docker volumes
#   ./reset.sh -c              # also reset the worker over SSH
# ============================================================
set -uo pipefail

: "${USER:=$(id -un)}"
: "${HOME:=$(getent passwd "${USER}" | cut -d: -f6)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PANEL_DIR="$(cd "${SCRIPT_DIR}/../panel" 2>/dev/null && pwd || true)"
# shellcheck source=cluster.env
[[ -f "${SCRIPT_DIR}/cluster.env" ]] && source "${SCRIPT_DIR}/cluster.env" || true
HF_CACHE_DIR="${HF_CACHE_DIR:-/data/hf-cache}"
CLUSTER_USER="${CLUSTER_USER:-$USER}"
WORKER_SSH_HOST="${WORKER_SSH_HOST:-}"

KEEP_MODELS=0; KEEP_IMAGES=0; PRUNE_VOLUMES=0; YES=0; WORKER=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --keep-models) KEEP_MODELS=1; shift;;
        --keep-images) KEEP_IMAGES=1; shift;;
        --volumes)     PRUNE_VOLUMES=1; shift;;
        -y|--yes)      YES=1; shift;;
        -c|--worker|--all-nodes) WORKER=1; shift;;
        -h|--help)     sed -n '2,17p' "$0"; exit 0;;
        *)             echo "Unknown argument: $1"; exit 1;;
    esac
done

say()  { printf "\n\033[1;36m== %s\033[0m\n" "$*"; }
warn() { printf "\033[1;33m[!] %s\033[0m\n" "$*"; }

# refuse to rm a dangerous cache path
case "${HF_CACHE_DIR}" in ""|"/"|"/root"|"/home") warn "Unsafe HF_CACHE_DIR='${HF_CACHE_DIR}'; will NOT delete models."; KEEP_MODELS=1;; esac

SUDO=""; [[ $EUID -ne 0 ]] && SUDO="sudo"

plan() {
    echo "This will remove on $(hostname):"
    echo "  - containers: vllm-node, vllm_node"
    echo "  - panel: gx10-panel service, sudoers drop-in, venv, panel.env, metrics db"
    echo "  - files: /etc/gx10-cluster.conf, /etc/netplan/60-cx7-cluster.yaml,"
    echo "           /etc/sysctl.d/90-gx10-cluster.conf, /etc/security/limits.d/90-gx10-memlock.conf"
    [[ "${KEEP_IMAGES}" == 1 ]] && echo "  - images: KEPT" || echo "  - ALL vLLM images + docker build cache"
    [[ "${KEEP_MODELS}" == 1 ]] && echo "  - model cache: KEPT" || echo "  - model cache: ${HF_CACHE_DIR} + spark caches (DELETED)"
    [[ "${PRUNE_VOLUMES}" == 1 ]] && echo "  - docker volumes: unreferenced volumes pruned"
    [[ "${WORKER}" == 1 ]] && echo "  - and the same on worker ${WORKER_SSH_HOST:-<unset>}"
}

wipe_local() {
    say "Containers"
    ${SUDO} docker rm -f vllm-node vllm_node 2>/dev/null || true

    say "Panel service"
    ${SUDO} systemctl stop gx10-panel 2>/dev/null || true
    ${SUDO} systemctl disable gx10-panel 2>/dev/null || true
    ${SUDO} rm -f /etc/systemd/system/gx10-panel.service
    ${SUDO} systemctl daemon-reload 2>/dev/null || true
    ${SUDO} rm -f /etc/sudoers.d/gx10-panel
    [[ -n "${PANEL_DIR}" ]] && rm -rf "${PANEL_DIR}/.venv" "${PANEL_DIR}/panel.env" "${PANEL_DIR}"/*.db 2>/dev/null || true

    say "Config / netplan / sysctl / limits"
    ${SUDO} rm -f /etc/gx10-cluster.conf /etc/netplan/60-cx7-cluster.yaml \
                  /etc/sysctl.d/90-gx10-cluster.conf /etc/security/limits.d/90-gx10-memlock.conf

    # remove a nested duplicate clone (e.g. ~/gx10-stack/gx10-stack), never the active repo
    local nested="$(cd "${SCRIPT_DIR}/.." && pwd)/gx10-stack"
    [[ -d "${nested}/.git" ]] && { say "Nested clone ${nested}"; rm -rf "${nested}"; }

    if [[ "${KEEP_IMAGES}" != 1 ]]; then
        say "vLLM images + build cache"
        local imgs; imgs="$(${SUDO} docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep -E 'vllm|vllm-ray|vllm-node' || true)"
        [[ -n "${imgs}" ]] && echo "${imgs}" | xargs -r ${SUDO} docker rmi -f 2>/dev/null || true
        ${SUDO} docker image prune -af 2>/dev/null || true
        ${SUDO} docker builder prune -af 2>/dev/null || true
    fi

    if [[ "${PRUNE_VOLUMES}" == 1 ]]; then
        say "Docker volumes (unreferenced)"
        ${SUDO} docker volume prune -f 2>/dev/null || true
    fi

    if [[ "${KEEP_MODELS}" != 1 ]]; then
        say "Model + build caches"
        ${SUDO} rm -rf "${HF_CACHE_DIR:?}" 2>/dev/null || true
        rm -rf "${HOME}/.cache/vllm" "${HOME}/.cache/flashinfer" "${HOME}/.triton" 2>/dev/null || true
    fi
    say "Local reset done."
}

wipe_worker() {
    [[ -z "${WORKER_SSH_HOST}" ]] && { warn "WORKER_SSH_HOST not set; skipping worker."; return; }
    local target="${CLUSTER_USER}@${WORKER_SSH_HOST}"
    say "Worker: ${target}"
    local tmp; tmp="$(mktemp)"
    cat > "${tmp}" <<EOF
#!/usr/bin/env bash
set -uo pipefail
docker rm -f vllm-node vllm_node 2>/dev/null || true
systemctl stop gx10-panel 2>/dev/null || true
systemctl disable gx10-panel 2>/dev/null || true
rm -f /etc/systemd/system/gx10-panel.service; systemctl daemon-reload 2>/dev/null || true
rm -f /etc/sudoers.d/gx10-panel
rm -f /etc/gx10-cluster.conf /etc/netplan/60-cx7-cluster.yaml /etc/sysctl.d/90-gx10-cluster.conf /etc/security/limits.d/90-gx10-memlock.conf
if [ "${KEEP_IMAGES}" != "1" ]; then
  docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep -E 'vllm|vllm-ray|vllm-node' | xargs -r docker rmi -f 2>/dev/null || true
  docker image prune -af 2>/dev/null || true
  docker builder prune -af 2>/dev/null || true
fi
if [ "${PRUNE_VOLUMES}" = "1" ]; then
  docker volume prune -f 2>/dev/null || true
fi
if [ "${KEEP_MODELS}" != "1" ]; then
  rm -rf "${HF_CACHE_DIR:?}" 2>/dev/null || true
  rm -rf /home/${CLUSTER_USER}/.cache/vllm /home/${CLUSTER_USER}/.cache/flashinfer /home/${CLUSTER_USER}/.triton 2>/dev/null || true
fi
echo "Worker reset done."
EOF
    scp -q "${tmp}" "${target}:/tmp/gx10-reset.$$.sh" && rm -f "${tmp}" \
        && ssh -t "${target}" "sudo bash /tmp/gx10-reset.$$.sh; rm -f /tmp/gx10-reset.$$.sh" \
        || warn "Worker reset needs attention (SSH/sudo)."
}

plan
if [[ "${YES}" != 1 ]]; then
    read -rp $'\nType NUKE to proceed: ' ans || true
    [[ "${ans}" == "NUKE" ]] || { echo "Aborted."; exit 1; }
fi
wipe_local
[[ "${WORKER}" == 1 ]] && wipe_worker
say "Done. A reboot is recommended (clears the removed netplan/sysctl), then: ./deploy.sh"
