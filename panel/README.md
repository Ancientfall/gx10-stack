# GX10 Cluster Control Panel

A web control panel for the dual GB10 cluster. Runs on the head node (`myspark`), serves a single page you open over Tailscale, and drives the real cluster: start, stop, verify, fix configuration, manage models, and monitor live.

## What it does

The page is organized top-to-bottom by what you're actually doing — status, live monitoring, control, trends, then admin:

- **Fabric & status (top)**: the signature live view of the two nodes and the ConnectX-7 link — green and animated at 200G when healthy, amber when degraded, red when down. An overall verdict line and the primary **Start / Stop / Optimize & fix** actions sit right below. The browser tab title and favicon also reflect cluster health, so a backgrounded tab shows status at a glance.
- **Cluster health**: the configuration checks that drive the overall verdict — link state and MTU, fabric reachability with a jumbo-frame probe, netplan conflicts, worker SSH, Ray GPU registration, and the vLLM API.
- **Live performance**: real-time serving throughput (tokens/sec), in-flight and queued requests, and total generated tokens, plus per-node gauges for GPU utilization, unified memory, power, temperature, and SM clock (head locally, worker over SSH).
- **Models & serving** — the main workspace:
  - **Favorites**: a curated, GB10-friendly model list with a rough memory-fit estimate (fits a single node / needs both nodes / too large) against your pooled-memory budget.
  - **Discover**: live Hugging Face search with footprint estimates parsed from the model name.
  - **Library**: models already in your local cache (`HF_CACHE_DIR`), so switching skips the download.
  - **Test**: send a prompt to the live model and see tokens/sec, token count, and latency.
  - Loading a model runs a pre-flight check (is it cached? does it replace what's serving? will it fit?), then downloads it with a live progress bar and/or loads it, watching the vLLM log until it's serving. Context length is auto-corrected once if the model caps it, and a failed load **rolls back** to the previously working model.
  - **Serving mode** flips between **Single node** (TP=1 on the head GPU with the multiprocessing backend — best for models that fit one box; faster to start and leaves the second GX10 free) and **Both nodes** (TP=2 sharded across the pair via Ray). The panel recommends one based on the model's memory fit, and switching reloads the live model in place.
  - **Serving parameters** live here too: edit model, tensor parallel, GPU mem util, and max length. **Save** writes to `cluster.env` for the next start; **Reload** swaps the model in the running cluster without a full restart; **Unload** stops serving to free memory.
- **Activity & logs**: a live activity console for start/stop/optimize/model output, with a toggle to tail the raw vLLM log.
- **History & cost**: sparkline charts of throughput, temperature, power, and memory over the last **hour / day / week**, sampled to a local SQLite database, plus a cost-equivalent estimate (tokens generated × a configurable $/1M-token rate) for today, this week, and all time. Token counting is robust to model reloads (vLLM's cumulative counter resets on restart).
- **vLLM version & image** (advanced, collapsible): pull a newer NVIDIA-validated NGC image, rebuild the Ray layer on both nodes, switch the cluster to it, and roll back — tucked into a collapsed section until you need it.
- **Optimize & fix** (idempotent): resolves the `40-cx7.yaml` netplan conflict, sets MTU 9000, disables IPv6 on the fabric, reloads sysctl tuning, drops page caches on both nodes, and checks the Docker runtime.
- **Check RDMA**: confirms NCCL is using RDMA (NET/IB) and not falling back to TCP sockets.

### Accessibility & polish

- Tabs (Models and the history window) are keyboard-navigable (arrow / Home / End) with correct ARIA roles, and your last-selected Models tab and history window are remembered across reloads.
- Status is conveyed by text and labels, not color alone; toasts are color-coded by outcome and announced to screen readers.
- Honors `prefers-reduced-motion` (calms the spinner, progress bars, and gauges).

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
- Finds the kit via `GX10_KIT_DIR`, then the sibling `../cluster`, then `~/gx10-cluster`. Set `GX10_KIT_DIR` if yours lives elsewhere.

## Configuration (environment)

| Variable | Default | Purpose |
|---|---|---|
| `GX10_KIT_DIR` | auto-detected | Where `cluster.env` and `02-launch-cluster.sh` live. |
| `GX10_PANEL_HOST` | `0.0.0.0` | Bind address for the panel. |
| `GX10_PANEL_PORT` | `8080` | Port for the panel. |
| `GX10_DB` | `gx10-metrics.db` (in the panel dir) | SQLite file for telemetry history. |
| `GX10_COST_PER_MTOK` | `10.0` | $ per 1M generated tokens, used for the cost-equivalent estimate. |

Serving settings (`MODEL`, `TENSOR_PARALLEL`, `MAX_MODEL_LEN`, `GPU_MEM_UTIL`, `HF_TOKEN`, `HF_CACHE_DIR`, …) are read from the cluster kit's `cluster.env`.

## Security

The panel exposes cluster control with no auth, so keep it on Tailscale only. It binds `0.0.0.0:8080`; do not port-forward it to the public internet. `tailscale serve` keeps it inside your tailnet. The sudoers drop-in grants only the specific fix commands, nothing broader.

## Run without systemd

```bash
GX10_KIT_DIR=~/gx10-cluster ./.venv/bin/python app.py
```
