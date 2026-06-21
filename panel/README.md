# GX10 Cluster Control Panel

A small web control panel for the dual GB10 cluster. Runs on the head node (`myspark`), serves a single page you open over Tailscale, and drives the real cluster: start, stop, verify, and fix configuration.

## What it does

- **Live fabric view**: the two nodes and the ConnectX-7 link, lit green at 200G when healthy, amber when degraded, red when down.
- **Request latency telemetry**: live TTFT, per-output-token, end-to-end, and queue-time latency (avg + p50/p95/p99) plus KV-cache use and successful-request count, parsed from the serving engine's Prometheus metrics (vLLM full histograms; llama.cpp averages).
- **Benchmark**: fire N concurrent streaming requests at the active model and read back aggregate tok/s and average TTFT, with a history of past runs.
- **History trends**: hour/day/week charts for throughput, temperature, power, memory, and serving quality (TTFT, inter-token and end-to-end latency, KV-cache use), plus a token-cost summary.
- **Start / Stop**: runs your `02-launch-cluster.sh` and streams output into the activity log.
- **Configuration checks**: link state and MTU, fabric reachability with jumbo-frame probe, netplan conflicts, worker SSH, Ray GPU registration, vLLM API.
- **Optimize & fix** (idempotent): resolves the `40-cx7.yaml` netplan conflict, sets MTU 9000, disables IPv6 on the fabric, reloads sysctl tuning, drops page caches on both nodes, checks the Docker runtime.
- **Serving config**: edit model, tensor parallel, GPU mem util, max length. Save writes to `cluster.env`; Reload swaps the model in the running cluster without a full restart.
- **Playground**: a streaming, multi-turn chat against whatever engine is serving (vLLM or llama.cpp), with time-to-first-token and tok/s measured live in the browser off the token stream.
- **Engines**: serve safetensors models on multi-node vLLM (Ray) or single-node, GGUF models on llama.cpp, and Qwen3.6-27B single-node on **vLLM + DFlash speculative decoding** (Library shows "Load (DFlash)"). The panel manages start/stop/switch for each.
- **OpenAI `/v1` passthrough**: a single stable endpoint (`http://<panel>/v1`) that forwards to whichever engine is live — point Open WebUI (or any OpenAI client) at it and it always sees the current model, no reconnecting.
- **Self-heal**: serving containers run `--restart unless-stopped`, and a panel watchdog restores the last-served engine after a crash/reboot if the container policy didn't.
- **Check RDMA**: confirms NCCL is using RDMA (NET/IB) and not falling back to TCP sockets.

## Install (on the head node)

```bash
# put this folder next to your gx10-cluster kit, then:
cd gx10-panel
./install.sh
```

The installer creates a venv, adds a narrow sudoers drop-in (the privileged fixes
plus `systemctl start/stop/restart gx10-panel`, so `deploy.sh` runs unattended),
installs a systemd service that starts on boot, and prints the URL.

The panel serves on **`:8090`** (8080 is left for Open WebUI). Open it from your Mac
at `http://myspark:8090` over Tailscale. If its port is taken it auto-falls-back to the
next free one and logs where it landed, so it never crash-loops on a conflict.

Update later with one command on the head node: `./deploy.sh` (pulls both nodes and
restarts the panel).

## Requirements

- Runs as your normal user, who must be in the `docker` group and have passwordless SSH to the worker (both already set up by `01-node-setup.sh`).
- Finds the kit via `GX10_KIT_DIR`, then `~/gx10-cluster`. Set `GX10_KIT_DIR` if yours lives elsewhere.

## Security

The panel exposes cluster control with no auth, so keep it on Tailscale only. It binds `0.0.0.0:8090`; do not port-forward it to the public internet. Reach it from anywhere by running Tailscale on the client device (`http://myspark:8090`) — no public exposure needed. The sudoers drop-in grants only the specific fix commands plus restarting the panel's own service, nothing broader.

## Run without systemd

```bash
GX10_KIT_DIR=~/gx10-cluster ./.venv/bin/python app.py
```
