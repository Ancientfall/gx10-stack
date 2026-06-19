#!/usr/bin/env bash
# ============================================================
# deploy.sh - one-command GX10 cluster + panel deploy wizard.
#
# Run on the HEAD node (myspark) from the repo root:
#     ./deploy.sh                 # guided
#     ./deploy.sh -y --engine spark --nodes cluster   # fewer prompts
#
# It will: init the spark-vllm-docker submodule, collect a few settings,
# set up passwordless SSH to the worker, sync the repo there, run node
# setup on both boxes, build the image, launch, and install the panel.
#
# Idempotent: safe to re-run. Hardware-dependent steps warn (not abort)
# so one hiccup doesn't kill the run.
# ============================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER="${ROOT}/cluster"
PANEL="${ROOT}/panel"
ENVF="${CLUSTER}/cluster.env"

# ---- logging ----
say()  { printf "\n\033[1;36m== %s\033[0m\n" "$*"; }
ok()   { printf "\033[1;32m%s\033[0m\n" "$*"; }
warn() { printf "\033[1;33m[!] %s\033[0m\n" "$*"; }
die()  { printf "\033[1;31m[x] %s\033[0m\n" "$*" >&2; exit 1; }

ASSUME_YES=0; ENGINE=""; NODES=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -y|--yes)    ASSUME_YES=1; shift;;
        --engine)    ENGINE="$2"; shift 2;;
        --nodes)     NODES="$2"; shift 2;;
        --single)    NODES="single"; shift;;
        -h|--help)   sed -n '2,17p' "$0"; exit 0;;
        *)           die "Unknown argument: $1";;
    esac
done

ask() {   # ask VAR "Prompt" "default"
    local __var="$1" __p="$2" __d="${3:-}" __a
    if [[ "${ASSUME_YES}" == "1" ]]; then printf -v "$__var" '%s' "$__d"; return; fi
    read -rp "$__p${__d:+ [$__d]}: " __a || true
    printf -v "$__var" '%s' "${__a:-$__d}"
}
confirm() { [[ "${ASSUME_YES}" == "1" ]] && return 0; local a; read -rp "$1 [y/N] " a || true; [[ "$a" =~ ^[Yy] ]]; }

getv()  { grep -E "^$1=" "${ENVF}" 2>/dev/null | head -1 | cut -d= -f2- | tr -d '"' || true; }
setv()  { local k="$1" v="$2"
    if grep -qE "^$k=" "${ENVF}" 2>/dev/null; then
        sed -i "s|^$k=.*|$k=\"$v\"|" "${ENVF}"
    else echo "$k=\"$v\"" >> "${ENVF}"; fi
}

# ------------------------------------------------------------
say "GX10 deploy wizard"
command -v git >/dev/null || die "git is required."
[[ -f "${CLUSTER}/01-node-setup.sh" ]] || die "Run this from the gx10-stack repo root on the HEAD node."

# 1) submodule (spark-vllm-docker)
say "Fetching submodules"
git -C "${ROOT}" submodule update --init --recursive || warn "submodule init failed (check network); 'spark' engine needs it."

# 2) config
[[ -f "${ENVF}" ]] || cp "${CLUSTER}/cluster.env.example" "${ENVF}"
say "Cluster settings"
ask CLUSTER_USER     "SSH user on both boxes"                 "$(getv CLUSTER_USER)"
ask WORKER_SSH_HOST  "Worker host/IP for SSH from head"       "$(getv WORKER_SSH_HOST)"
ask MODEL            "Model to serve"                         "$(getv MODEL)"
ask HF_TOKEN         "HuggingFace token (blank if none)"      "$(getv HF_TOKEN)"
[[ -n "${ENGINE}" ]] || ask ENGINE "Engine: native or spark (broadest models)" "native"
[[ -n "${NODES}"  ]] || ask NODES  "Nodes: cluster (two) or single"            "cluster"
setv CLUSTER_USER    "${CLUSTER_USER}"
setv WORKER_SSH_HOST "${WORKER_SSH_HOST}"
setv MODEL           "${MODEL}"
setv HF_TOKEN        "${HF_TOKEN}"
[[ "${NODES}" == "single" ]] && setv TENSOR_PARALLEL 1 || setv TENSOR_PARALLEL 2
ok "Wrote ${ENVF}"

WORKER="${CLUSTER_USER}@${WORKER_SSH_HOST}"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=6"
REPO_URL="$(git -C "${ROOT}" remote get-url origin 2>/dev/null || echo https://github.com/Ancientfall/gx10-stack.git)"

# 3) passwordless SSH to worker
say "Worker SSH"
if ${SSH} "${WORKER}" true 2>/dev/null; then
    ok "Passwordless SSH to ${WORKER} works."
else
    warn "No passwordless SSH to ${WORKER} yet."
    if confirm "Generate a key (if needed) and run ssh-copy-id ${WORKER}?"; then
        [[ -f ~/.ssh/id_ed25519.pub || -f ~/.ssh/id_rsa.pub ]] || ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
        ssh-copy-id "${WORKER}" || warn "ssh-copy-id failed; set up keys manually, then re-run."
    fi
fi

# 4) sync repo + env to the worker
say "Worker repo"
${SSH} "${WORKER}" "test -d gx10-stack/.git || git clone --recurse-submodules ${REPO_URL} gx10-stack" \
    || warn "Could not clone the repo on the worker."
scp -q "${ENVF}" "${WORKER}:gx10-stack/cluster/cluster.env" 2>/dev/null \
    && ok "Synced cluster.env to worker." || warn "Could not copy cluster.env to worker."

# 5) node setup (privileged; may update OS/firmware and need a reboot)
say "Node setup"
if confirm "Run node setup with sudo on head and worker now? (OS/firmware update; may want a reboot after)"; then
    sudo "${CLUSTER}/01-node-setup.sh" head || warn "Head node setup reported issues."
    ssh -t "${WORKER}" "cd gx10-stack/cluster && sudo ./01-node-setup.sh worker" || warn "Worker node setup needs attention."
    warn "If the kernel/firmware updated, reboot both boxes and log out/in (docker group) before launching."
else
    warn "Skipped node setup. Run 'sudo ./cluster/01-node-setup.sh head|worker' before launching."
fi

# 6) build + launch
say "Build & launch (engine: ${ENGINE})"
if [[ "${ENGINE}" == "spark" ]]; then
    [[ -f "${CLUSTER}/spark-vllm-docker/build-and-copy.sh" ]] || die "spark-vllm-docker submodule missing; re-run after 'git submodule update --init'."
    ( cd "${CLUSTER}/spark-vllm-docker" && ./build-and-copy.sh -c ) || warn "spark image build/copy needs attention."
    START="./launch-cluster.sh start"; [[ "${NODES}" == "single" ]] && START="./launch-cluster.sh --solo start"
    ( cd "${CLUSTER}/spark-vllm-docker" && bash -lc "${START}" ) || warn "spark launch needs attention."
else
    ( cd "${CLUSTER}" && ./00-build-image.sh ) || warn "native image build needs attention."
    ( cd "${CLUSTER}" && ./02-launch-cluster.sh "${NODES}" ) || warn "native launch needs attention."
    ( cd "${CLUSTER}" && ./03-verify.sh ) || true
fi

# 7) panel
say "Control panel"
if [[ "${ENGINE}" == "spark" ]]; then
    "${PANEL}/install.sh" --engine spark --orch-dir "${CLUSTER}/spark-vllm-docker" --nodes "${NODES}" \
        || warn "Panel install needs attention."
else
    "${PANEL}/install.sh" || warn "Panel install needs attention."
fi

say "Done"
ok "Deploy finished. The panel URL is printed above (http://<tailscale-ip>:8080)."
echo "Re-run ./deploy.sh any time; it is idempotent."
