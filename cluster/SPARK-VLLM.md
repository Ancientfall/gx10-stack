# Running on spark-vllm-docker (broadest model support)

[`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker) is a
community engine purpose-built for DGX Spark / GB10 (your ASUS Ascent GX10s),
tracking the latest vLLM + FlashInfer with the broadest model and quantization
coverage (NVFP4 incl. the FP4-MoE backend, FP8, AWQ, GPTQ, INT4-AutoRound,
experimental mxfp4), plus GB10-specific fixes (NCCL multi-node hang fix,
`fastsafetensors` fast loading, GiB memory reservation) and per-model "mods".

This guide adopts it as the orchestrator while keeping the **GX10 panel** as the
control surface. The native `02-launch-cluster.sh` kit stays as a fallback.

> **Heads up:** the steps below run on the GB10s (GPU + a from-source vLLM
> build), so they have not been validated from CI. Treat the first run as a
> bring-up and keep the native `gx10/vllm-ray` image as a known-good fallback.

> **Fastest path:** `./deploy.sh --engine spark` from the repo root does all of
> the below (submodule, build, launch, panel wiring). The steps here are the
> manual equivalent / reference.

## 1. Get it on both nodes

It's vendored as a git submodule at `cluster/spark-vllm-docker`. If you cloned
gx10-stack with `--recurse-submodules` it's already there; otherwise:

```bash
git submodule update --init --recursive
cd cluster/spark-vllm-docker
```

Passwordless SSH head→worker is already set up by `01-node-setup.sh`.

> **Networking is eugr's job on this path, not ours.** Reaching the full ~200 Gb/s
> needs *both* ConnectX-7 ports cabled with per-port IPs and the right topology —
> that's configured by `./launch-cluster.sh --setup` here. Our `01-node-setup.sh`
> only handles the base (Docker, NVIDIA toolkit, memlock, cache dir); its
> single-port netplan is for the *native* kit, so let eugr own the fabric here.
> (One cable ≈ ~100 Gb/s — each QSFP port is a PCIe Gen5 x4 link; the rated 200 is
> the two ports combined, not per port.)

## 2. Build the image and copy to the worker

From the **head**:

```bash
cd ~/spark-vllm-docker
./build-and-copy.sh -c            # build, then distribute to the worker
# useful flags: --rebuild-vllm  --gpu-arch 12.1a  --exp-mxfp4  --tf5  -j 16
./launch-cluster.sh --setup status   # one-time: write .env (CLUSTER_NODES, ETH_IF, IB_IF, ...)
```

The default image tag is `vllm-node` and the per-node container is `vllm_node`.

## 3. Start the cluster

```bash
./launch-cluster.sh start         # idle containers on both nodes (+ Ray head/worker)
./launch-cluster.sh --solo start  # single node (head only) — pairs with the panel's Single mode
./launch-cluster.sh status
./launch-cluster.sh stop
```

## 4. Serve a model

```bash
# two-node, tensor parallel across both GB10s
./launch-cluster.sh exec vllm serve QuantTrio/MiniMax-M2-AWQ \
  --host 0.0.0.0 --port 8000 -tp 2 \
  --distributed-executor-backend ray \
  --gpu-memory-utilization 0.7 --max-model-len 128000 \
  --load-format fastsafetensors

# single node
./launch-cluster.sh --solo -p 8000:8000 exec vllm serve \
  QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ \
  --host 0.0.0.0 --port 8000 --load-format fastsafetensors
```

Recipes and mods (the broad-model conveniences) live here too:

```bash
./run-recipe.sh --list
./run-recipe.sh glm-4.7-flash-awq --solo --setup
./launch-cluster.sh --apply-mod ./mods/fix-Salyut1-GLM-4.7-NVFP4 \
  exec vllm serve Salyut1/GLM-4.7-NVFP4 --host 0.0.0.0 --port 8000
./hf-download.sh QuantTrio/MiniMax-M2-AWQ -c --copy-parallel   # download + distribute
```

## 5. Point the GX10 panel at it

The panel is orchestrator-agnostic. `deploy.sh --engine spark` and
`panel/install.sh --engine spark` wire this up automatically by writing
`panel/panel.env` (loaded by the systemd unit):

```
GX10_CONTAINER=vllm_node
GX10_ORCH_DIR=<repo>/cluster/spark-vllm-docker
GX10_START_CMD=./launch-cluster.sh start     # or '--solo start' for single node
GX10_STOP_CMD=./launch-cluster.sh stop
```

To convert an existing native install: `panel/install.sh --engine spark`
(add `--nodes single` for solo), then `sudo systemctl restart gx10-panel`.
With these set:

- **Start / Stop** run eugr's `launch-cluster.sh` (in `GX10_ORCH_DIR`) and stream
  its output into the activity log.
- **Telemetry, health checks, the Test tab, and model load/unload** keep working
  — they `docker exec` into `vllm_node`, read `/v1/models` + `/metrics`, and run
  `nvidia-smi` exactly as before.
- The **Single / Both nodes** toggle still sets `TENSOR_PARALLEL`; pair it with
  `--solo` when you start from eugr for true single-node.

For single-node, set `GX10_START_CMD=./launch-cluster.sh --solo start`.

## Known seams (verify on first bring-up)

- **Model load from the panel** runs `vllm serve … > /var/log/vllm.log` directly
  in `vllm_node`, which bypasses eugr's `exec` wrapper (so recipe/mod env isn't
  applied). For models that need a mod or special env, launch them with
  `./run-recipe.sh` / `--apply-mod` on the box; the panel will still show them
  serving and monitor them. Simple models load fine straight from the panel.
- **vLLM log view**: panel-initiated loads log to `/var/log/vllm.log` (visible in
  the panel). Models launched via eugr's `exec` log to the container — use
  `docker logs vllm_node` or `./launch-cluster.sh` output for those.
- **Downloads**: the panel's download button tries `hf` then `huggingface-cli`
  inside the container. For cluster-wide pre-staging, prefer eugr's
  `./hf-download.sh <model> -c`.

## Reverting to the native kit

Unset the four `GX10_*` env vars (or `systemctl revert gx10-panel`), restart the
panel, and it drives `02-launch-cluster.sh` against the `gx10/vllm-ray` image
again. Nothing about the native path is removed.
