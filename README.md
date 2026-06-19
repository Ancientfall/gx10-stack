# gx10-stack

Setup scripts and a web control panel for a dual ASUS Ascent GX10 (NVIDIA GB10) cluster running distributed vLLM over the ConnectX-7 200G link.

## Layout

```
gx10-stack/
  cluster/    one-time node setup, cluster launch, and verify scripts
  panel/      FastAPI web control panel (status, telemetry, models, history)
```

Each folder has its own README with details. Start with `cluster/` to bring the
two nodes up, then `panel/` for the web UI.

The panel is a single page that covers the whole workflow: a live fabric/health
view, start/stop/optimize, real-time per-node telemetry, model management
(curated favorites, Hugging Face search, local library, and a prompt tester with
download/load progress and rollback), serving-parameter editing, history & cost
charts, and an optional vLLM image/version manager. See `panel/README.md`.

## Quickstart

On each GX10, once the repo is cloned:

```bash
# 1. configure (edit model, HF token, worker host)
cd cluster
cp cluster.env.example cluster.env   # scripts also do this automatically
nano cluster.env

# 2. node setup (run on each box with its role)
sudo ./01-node-setup.sh head      # on myspark
sudo ./01-node-setup.sh worker    # on myspark2

# 3. launch from the head node
./02-launch-cluster.sh
./03-verify.sh

# 4. optional: install the web panel on the head node
cd ../panel
./install.sh
```

## Remote deploy over SSH

From your Mac, with the boxes reachable on Tailscale:

```bash
# head node
ssh nealasmothers@myspark  'git clone git@github.com:Ancientfall/gx10-stack.git'
# worker node
ssh nealasmothers@myspark2 'git clone git@github.com:Ancientfall/gx10-stack.git'
```

To pull updates later:

```bash
ssh nealasmothers@myspark  'cd gx10-stack && git pull'
ssh nealasmothers@myspark2 'cd gx10-stack && git pull'
```

## Notes

- `cluster.env` holds your HF token and is gitignored. Only `cluster.env.example`
  is tracked, so secrets never enter git history.
- The panel expects the kit at `../cluster` (this repo layout) or `~/gx10-cluster`,
  or set `GX10_KIT_DIR`.
- Keep the panel on Tailscale only. It has no auth by design.
- For the broadest model/quant coverage on GB10, you can run the
  [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) engine and drive
  it from the panel via a few `GX10_*` env vars — see `cluster/SPARK-VLLM.md`.
