#!/usr/bin/env bash
# deploy.sh - pull the latest on both nodes and restart the panel. Idempotent.
# With the panel sudoers drop-in (install.sh) this runs unattended.
#   ./deploy.sh            # pull both nodes + restart panel
#   ./deploy.sh --no-worker # head only
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "${HERE}/cluster/cluster.env" ]] && source "${HERE}/cluster/cluster.env" 2>/dev/null || true
log() { printf '\033[0;32m==>\033[0m %s\n' "$*"; }

log "Pull (head)"
git -C "${HERE}" pull --ff-only

WH="${WORKER_SSH_HOST:-}"
if [[ "${1:-}" != "--no-worker" && -n "${WH}" ]]; then
    log "Pull (worker ${WH})"
    ssh -o ConnectTimeout=10 "${WH}" "cd ~/gx10-stack && git pull --ff-only" || log "worker pull failed (continuing)"
fi

log "Restart panel"
sudo -n systemctl restart gx10-panel.service 2>/dev/null || sudo systemctl restart gx10-panel.service
sleep 2
if systemctl is-active --quiet gx10-panel.service; then
    log "Panel restarted and active."
else
    log "WARNING: panel is not active — check: journalctl -u gx10-panel -n 50"
    exit 1
fi
