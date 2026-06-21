# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`gx10-stack` turns two ASUS Ascent GX10 boxes (NVIDIA GB10, 128GB UMA each) into model-serving infrastructure: bash setup scripts plus a FastAPI web control panel. It serves models over distributed vLLM across the ConnectX-7 200G link (tensor-parallel, 256GB pooled), or single-node via llama.cpp (GGUF) or vLLM + DFlash speculative decoding — all driven from the panel.

Two top-level pieces:
- `cluster/` — numbered, run-once-per-node bash scripts that bring the two nodes up and launch engines.
- `panel/` — a single-file FastAPI backend (`app.py`) + a single static HTML page that runs on the head node and shells out to the cluster scripts, docker, ssh, and networking tools.

## Commands

Panel (run on the head node, `myspark`):

```bash
cd panel && ./install.sh                 # venv + sudoers drop-in + systemd service on :8090
GX10_KIT_DIR=~/gx10-cluster ./.venv/bin/python app.py   # run without systemd
sudo systemctl restart gx10-panel        # manage the service
journalctl -u gx10-panel -f              # tail panel logs
```

Cluster bring-up (in order, on the indicated node):

```bash
sudo ./01-node-setup.sh head             # on myspark; worker on myspark2
./02-launch-cluster.sh                    # from head: Ray head + worker over SSH, vLLM TP=2
./03-verify.sh [--bw]                      # fabric/Ray/API checks; --bw adds an RDMA bandwidth test
sudo ./04-make-persistent.sh head|worker  # survive reboot
```

Deploy/update (one command on the head node, runs unattended via the sudoers drop-in):

```bash
./deploy.sh                  # git pull on both nodes + restart panel
./deploy.sh --no-worker      # head only
```

There is no test suite and no linter configured. Dependencies are only `fastapi` + `uvicorn[standard]` (`panel/requirements.txt`); the cluster scripts are plain bash. `app.py` runs directly with `python3`.

## Architecture

### Panel backend (`panel/app.py`, ~2850 lines, single file)

All backend logic lives in this one module — helper functions at the top, the `@app.*` route handlers near the bottom (~line 2300+), `uvicorn.run` in `__main__`. The UI is one file, `panel/static/index.html` (~100KB, no build step). When changing behavior, expect to touch both `app.py` (an endpoint) and `static/index.html` (the caller); there is no separate frontend toolchain.

### Three serving engines behind one API port

The panel manages start/stop/switch across three mutually-exclusive serving paths, all exposing the OpenAI-compatible API on `API_PORT` (default 8000):
- **multi-node vLLM** — the Ray cluster from `02-launch-cluster.sh`, container `vllm-node` (constant `CONTAINER`). For safetensors models that need both boxes.
- **single-node llama.cpp** — `cluster/llama-serve.sh`, for GGUF models (and bleeding-edge ones vLLM can't load yet).
- **single-node vLLM + DFlash** — `cluster/vllm-dflash-serve.sh`, container `vllm-dflash` (`DFLASH_CONTAINER`), speculative decoding. Triggered only for base models in `DFLASH_PAIRS` (hardcoded, currently just `Qwen/Qwen3.6-27B`) when the draft model is cached. Because it serves on the same API port, it is detected as the "vllm" engine.

Engine selection is by model file type: GGUF → llama.cpp, safetensors → vLLM (`detect_engine`). `/v1/*` on the panel is a **passthrough** that forwards to whichever engine is currently live, so external clients (Open WebUI, etc.) point at one stable `http://<panel>/v1` and never reconnect on a model swap.

### Config resolution

Settings live in `cluster/cluster.env` (gitignored — holds the HF token; only `cluster.env.example` is tracked, seeded automatically on first run). `cfg()` returns a merged dict: hardcoded defaults ← `cluster.env` ← `/etc/gx10-cluster.conf` (node conf written by setup). The panel finds the kit via `GX10_KIT_DIR`, then sibling `../cluster`, then `~/gx10-cluster` (`_resolve_kit_dir`). Single-key edits go through `write_env_value`, which is lock-serialized and atomic (temp file + `os.replace`).

### Metrics, persistence, and self-heal

- A background collector thread (`_collector_loop`, every ~10s) samples GPU/engine metrics and writes to a SQLite DB (`gx10-metrics.db`, gitignored). Serving-quality numbers are parsed from the engine's Prometheus endpoint — full histograms for vLLM (`_parse_prom_hist`), averages for llama.cpp. History/cost/telemetry endpoints read back from this DB.
- The currently-served engine/model is saved (`save_serving_state`) and restored after a crash/reboot (`_maybe_restore_serving`), backing the watchdog that complements the containers' `--restart unless-stopped` policy.
- Long operations (downloads, model loads, NAS copies) run in background threads tracked by a single-slot job system (`_claim_job`/`_job_is_current`/`_release_job`) so the UI can poll progress and only one heavy task runs at a time.

### Concurrency

Shared state is guarded by module-level locks: `_action_lock` (serializes start/stop so they can't overlap), `_env_lock` (env writes), `_activity_lock` (the activity ring buffer). Read-heavy probes use a small TTL cache (`cached`/`invalidate_cache`). Keep new long-running or state-mutating work inside this locking/job model rather than adding ad-hoc threads.

## Constraints and gotchas

- **No auth, by design.** The panel exposes full cluster control and binds `0.0.0.0:8090`. Keep it Tailscale-only; never port-forward it publicly. The sudoers drop-in (`install.sh`) grants only the specific privileged fixes plus restart of the panel's own service — keep it that narrow.
- **Port 8090, not 8080.** 8080 is deliberately skipped (Open WebUI's default); `_pick_port` sticks to 8090 and falls back 8090→8091→8092, retrying the preferred port briefly so it stays stable across restarts instead of crash-looping on a collision.
- **GB10 UMA.** CPU and GPU share the 128GB. The launch path drops page caches and caps `--gpu-memory-utilization` at 0.80; OOM-at-load usually means lowering `GPU_MEM_UTIL` or `MAX_MODEL_LEN`, not a code bug.
- **CX7 fabric is order-sensitive.** The 200G link only comes up when the QSFP cable is in the *same port number* on both boxes; Ray/NCCL must bind the CX7 interface (`VLLM_HOST_IP`, `NCCL_SOCKET_IFNAME`), not the LAN. `03-verify.sh` and the panel's config checks diagnose this.
- The NGC vLLM image no longer ships the `ray` CLI, so `00-build-image.sh` layers it back on — multi-node clustering depends on that rebuilt image.

## Workflow note

This repo is built in numbered phases on `main`. "Continue the build" means ship the next phase. Work happens in a git worktree, then fast-forward merges to `main`.
