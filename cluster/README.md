# GX10 Dual-Node vLLM Cluster Kit

Turns two ASUS Ascent GX10s (NVIDIA GB10, 128GB each) into a single 256GB tensor-parallel vLLM inference server over the ConnectX-7 QSFP link, exposing an OpenAI-compatible API.

## Files

| File | Where to run | Purpose |
|---|---|---|
| `cluster.env` | edit first | All settings: IPs, model, image, token |
| `01-node-setup.sh` | both nodes, sudo | OS update, Docker, CX7 detection, netplan, tuning |
| `02-launch-cluster.sh` | head node, no sudo | Start Ray head + worker, launch vLLM TP=2 |
| `03-verify.sh` | head node | Link, fabric, Ray, API checks |

## Run order

1. Rack both boxes, connect the QSFP DAC cable into the **same port number** on each unit. The link will not come up on mismatched ports.
2. Edit `cluster.env` (set `CLUSTER_USER`, `HF_TOKEN`, `MODEL`, and `WORKER_SSH_HOST` to the worker's current LAN IP).
3. Copy this folder to both boxes.
4. On box 1: `sudo ./01-node-setup.sh head`
5. On box 2: `sudo ./01-node-setup.sh worker`
6. Reboot both if kernel/firmware updated. Log out/in so the docker group applies.
7. From head: `ssh-copy-id <user>@<worker>` then `ping 192.168.100.11`
8. From head: `./02-launch-cluster.sh`
9. From head: `./03-verify.sh` (add `--bw` for an RDMA bandwidth test)

## Single-node mode

Models that fit on one GB10 run faster per token on a single box (no cross-node
NCCL), and you can leave the second unit idle. The launch script supports this:

```bash
./02-launch-cluster.sh single     # head GPU only (TP=1, mp backend), worker left idle
./02-launch-cluster.sh cluster    # both boxes (TP=2, Ray) - the default
./02-launch-cluster.sh            # picks single when TENSOR_PARALLEL=1, else two-node
```

Single mode never starts the worker container (and clears any stale one), so the
second GX10 sits idle. `03-verify.sh` will flag the fabric / 2-GPU checks as down
in this mode - that is expected. The web panel exposes the same choice as a
**Single node / Both nodes** toggle and recommends one from the model's size.

`01-node-setup.sh` no longer hard-requires the fabric: if no ConnectX-7 link is
up it warns and continues (and skips `netplan apply`, so it won't bounce your SSH),
so you can set up and serve on one box before — or without — cabling the link.

## Alternative engine: spark-vllm-docker

For the broadest model and quantization coverage on GB10 (latest vLLM +
FlashInfer, NVFP4 FP4-MoE, AWQ/GPTQ, recipes and per-model mods), you can run
[`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker) as the
orchestrator and still drive it from the GX10 panel. See
[`SPARK-VLLM.md`](SPARK-VLLM.md). The native kit here remains the fallback.

> Note: the spark engine's autodiscovery expects **both** ConnectX-7 ports cabled
> between the boxes (2 active interfaces). The native kit needs only one cable.

## Cleaning up disk

vLLM images (~19GB each), Docker build cache, and the HuggingFace model cache add
up fast. `cleanup.sh` reports usage and reclaims the safe stuff:

```bash
./cleanup.sh             # report only (images, build cache, model cache, disk)
./cleanup.sh --prune     # + stopped containers, dangling images, build cache
./cleanup.sh --prune -c  # also run on the worker over SSH
```

It prints (but never auto-runs) the bigger node-specific wins: removing old images
with `docker rmi`, deleting cached models under `$HF_CACHE_DIR/hub/models--*`, and
clearing `~/.cache/vllm` / `~/.cache/flashinfer` / `~/.triton` from spark builds.

## Start fresh (reset)

To wipe the cluster software back to a clean slate **without reinstalling the OS**:

```bash
./reset.sh                    # this node (shows a plan, asks before deleting)
./reset.sh -c                 # also reset the worker over SSH
./reset.sh -y --keep-models   # no prompt, but keep downloaded models
```

It removes the containers, the panel (service, sudoers drop-in, venv, `panel.env`,
metrics db), the config/netplan/sysctl/limits files node-setup wrote, a nested
duplicate clone, all vLLM images + build cache, and — unless `--keep-models` — the
HF model cache and spark build caches. Reboot afterwards, then `./deploy.sh`.

## Hardware reality check

Each QSFP port hangs off a PCIe Gen5 x4 link (~100 Gb/s to the host), so a **single cable gives ~100 Gb/s** and the rated **200 Gb/s is the two ports combined — not 400**. Reaching the full ~200 needs both cables with per-port IPs over RoCE/RDMA (the spark engine's autodiscovery handles this; see `SPARK-VLLM.md`). Either way it's plenty: tensor-parallel traffic for 70B–120B class models saturates well below 100 Gb/s, so a single cable works fine for the native kit.

## Remote access

The API binds to all interfaces on the head node, port 8000. Since you already run Tailscale, install it on the head GX10's management side (the 10GbE RJ-45 network, not the CX7 link) and hit `http://<tailscale-ip>:8000/v1` from your Mac. Works as a drop-in OpenAI base URL for LM Studio clients, Open WebUI, or your own apps.

## Memory notes (GB10 UMA)

CPU and GPU share the 128GB. The launch script drops page caches before starting and caps `--gpu-memory-utilization` at 0.80. If you see OOMs during load:
- lower `GPU_MEM_UTIL` to 0.70
- lower `MAX_MODEL_LEN`
- stop anything else running on the boxes

## Model sizing guide (TP=2, ~205GB usable pooled)

- FP8 120B class (gpt-oss-120b, Qwen3 large variants): comfortable
- 70B FP16 / 405B heavily quantized: possible with tight max-model-len
- Anything that fits on one box alone: skip the cluster, run single-node, it is faster per token

## Common failures

| Symptom | Cause | Fix |
|---|---|---|
| Ray stuck "pending placement group" | Ray bound to LAN instead of CX7 | Already handled via `VLLM_HOST_IP` + `NCCL_SOCKET_IFNAME`; verify IPs in `/etc/gx10-cluster.conf` |
| No CX7 link Up | Cable in different port numbers | Move to matching ports |
| NCCL timeout | Wrong iface env or firewall | `ibdev2netdev` on both nodes, confirm iface names match the conf files |
| OOM at model load | Page cache hogging UMA | Re-run launch script (it drops caches), lower `GPU_MEM_UTIL` |
| API up but slow first reply | Weights downloading | `docker exec vllm-node tail -f /var/log/vllm.log` |
