# GX10 Cluster Control Panel

A small web control panel for the dual GB10 cluster. Runs on the head node (`myspark`), serves a single page you open over Tailscale, and drives the real cluster: start, stop, verify, and fix configuration.

## What it does

- **Live fabric view**: the two nodes and the ConnectX-7 link, lit green at 200G when healthy, amber when degraded, red when down.
- **Request latency telemetry**: live TTFT, per-output-token, end-to-end, and queue-time latency (avg + p50/p95/p99) plus KV-cache use and successful-request count, parsed from the serving engine's Prometheus metrics (vLLM full histograms; llama.cpp averages).
- **Benchmark**: fire N concurrent streaming requests at the active model and read back aggregate tok/s and average TTFT, with a history of past runs.
- **Start / Stop**: runs your `02-launch-cluster.sh` and streams output into the activity log.
- **Configuration checks**: link state and MTU, fabric reachability with jumbo-frame probe, netplan conflicts, worker SSH, Ray GPU registration, vLLM API.
- **Optimize & fix** (idempotent): resolves the `40-cx7.yaml` netplan conflict, sets MTU 9000, disables IPv6 on the fabric, reloads sysctl tuning, drops page caches on both nodes, checks the Docker runtime.
- **Serving config**: edit model, tensor parallel, GPU mem util, max length. Save writes to `cluster.env`; Reload swaps the model in the running cluster without a full restart.
- **Playground**: a streaming, multi-turn chat against whatever engine is serving (vLLM or llama.cpp), with time-to-first-token and tok/s measured live in the browser off the token stream.
- **Check RDMA**: confirms NCCL is using RDMA (NET/IB) and not falling back to TCP sockets.

## Install (on the head node)

```bash
# put this folder next to your gx10-cluster kit, then:
cd gx10-panel
./install.sh
```

The installer creates a venv, adds a narrow sudoers drop-in for the few privileged fixes, installs a systemd service that starts on boot, and prints the URL.

Open it from your Mac at `http://<tailscale-ip>:8080`, or run `sudo tailscale serve --bg 8080` for a clean HTTPS URL.

## Requirements

- Runs as your normal user, who must be in the `docker` group and have passwordless SSH to the worker (both already set up by `01-node-setup.sh`).
- Finds the kit via `GX10_KIT_DIR`, then `~/gx10-cluster`. Set `GX10_KIT_DIR` if yours lives elsewhere.

## Security

The panel exposes cluster control with no auth, so keep it on Tailscale only. It binds `0.0.0.0:8080`; do not port-forward it to the public internet. `tailscale serve` keeps it inside your tailnet. The sudoers drop-in grants only the specific fix commands, nothing broader.

## Run without systemd

```bash
GX10_KIT_DIR=~/gx10-cluster ./.venv/bin/python app.py
```
