#!/usr/bin/env python3
# ============================================================
# gx10-panel  -  control panel backend for the dual GB10 cluster
#
# Runs on the HEAD node (myspark) as your normal user.
# Serves the web UI and shells out to the real commands:
#   the launch script, docker, ibdev2netdev, netplan, ssh to worker.
#
# Start manually:   python3 app.py
# Or via systemd:   see gx10-panel.service
# ============================================================
import os
import re
import glob
import time
import json
import shlex
import threading
import subprocess
import collections
from pathlib import Path

from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# ------------------------------------------------------------
# Paths and configuration
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# Where the cluster kit lives (cluster.env, 02-launch-cluster.sh).
# Override with GX10_KIT_DIR. Defaults to ~/gx10-cluster, then the panel dir.
def _resolve_kit_dir() -> Path:
    env = os.environ.get("GX10_KIT_DIR")
    if env and Path(env).is_dir():
        return Path(env)
    # sibling layout from the consolidated repo: repo/panel + repo/cluster
    sibling = BASE_DIR.parent / "cluster"
    if (sibling / "cluster.env").exists() or (sibling / "cluster.env.example").exists():
        if not (sibling / "cluster.env").exists():
            try:
                (sibling / "cluster.env").write_text((sibling / "cluster.env.example").read_text())
            except Exception:
                pass
        return sibling
    home_kit = Path.home() / "gx10-cluster"
    if (home_kit / "cluster.env").exists():
        return home_kit
    return BASE_DIR

KIT_DIR = _resolve_kit_dir()
CLUSTER_ENV = KIT_DIR / "cluster.env"
NODE_CONF = Path("/etc/gx10-cluster.conf")
CONTAINER = "vllm-node"
LAUNCH_SCRIPT = KIT_DIR / "02-launch-cluster.sh"

PANEL_HOST = os.environ.get("GX10_PANEL_HOST", "0.0.0.0")
PANEL_PORT = int(os.environ.get("GX10_PANEL_PORT", "8080"))

# Fields exposed in the config editor and their casts
CONFIG_FIELDS = {
    "MODEL": str,
    "TENSOR_PARALLEL": int,
    "MAX_MODEL_LEN": int,
    "GPU_MEM_UTIL": float,
}

# ------------------------------------------------------------
# Small key/value .env reader and writer (keeps quotes/format sane)
# ------------------------------------------------------------
def read_env(path: Path) -> dict:
    values = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        values[key] = val
    return values


def write_env_value(path: Path, key: str, value: str) -> None:
    """Update a single KEY="value" line in place, preserving the rest of the file."""
    lines = path.read_text().splitlines() if path.exists() else []
    pattern = re.compile(rf'^\s*{re.escape(key)}\s*=')
    replaced = False
    for i, line in enumerate(lines):
        if pattern.match(line):
            lines[i] = f'{key}="{value}"'
            replaced = True
            break
    if not replaced:
        lines.append(f'{key}="{value}"')
    path.write_text("\n".join(lines) + "\n")


def cfg() -> dict:
    """Merged config: cluster.env plus node conf, with safe defaults."""
    data = {
        "HEAD_IP": "192.168.100.10",
        "WORKER_IP": "192.168.100.11",
        "WORKER_SSH_HOST": "myspark2",
        "CLUSTER_USER": os.environ.get("USER", "neal"),
        "API_PORT": "8000",
        "RAY_DASHBOARD_PORT": "8265",
        "TENSOR_PARALLEL": "2",
        "GPU_MEM_UTIL": "0.80",
        "MAX_MODEL_LEN": "65536",
        "MODEL": "openai/gpt-oss-120b",
        "VLLM_IMAGE": "nvcr.io/nvidia/vllm:26.01-py3",
        "HF_TOKEN": "",
        "CX7_IFACE": "enp1s0f1np1",
    }
    data.update(read_env(CLUSTER_ENV))
    data.update(read_env(NODE_CONF))
    return data


# ------------------------------------------------------------
# Command runner + activity ring buffer
# ------------------------------------------------------------
_activity = collections.deque(maxlen=600)
_activity_lock = threading.Lock()
_action_lock = threading.Lock()      # guards start/stop so they cannot overlap
_current_action = {"name": None}     # what long task is running, if any


def log_line(text: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    with _activity_lock:
        _activity.append({"t": stamp, "line": text})


def run(cmd, timeout=20, shell=False, capture_to_activity=False):
    """Run a command, return (rc, stdout, stderr). Never raises on non-zero."""
    try:
        proc = subprocess.run(
            cmd if shell else (cmd if isinstance(cmd, list) else shlex.split(cmd)),
            shell=shell,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if capture_to_activity:
            for stream in (proc.stdout, proc.stderr):
                for ln in (stream or "").splitlines():
                    if ln.strip():
                        log_line(ln)
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except subprocess.TimeoutExpired:
        return 124, "", "timed out"
    except FileNotFoundError as exc:
        return 127, "", str(exc)


def ssh_worker(remote_cmd: str, timeout=20):
    c = cfg()
    target = f'{c["CLUSTER_USER"]}@{c["WORKER_SSH_HOST"]}'
    base = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", target, remote_cmd]
    return run(base, timeout=timeout)


# ------------------------------------------------------------
# Read-only status checks
# ------------------------------------------------------------
def _iface_speed_mtu(iface: str):
    speed = mtu = None
    try:
        mtu = (Path(f"/sys/class/net/{iface}/mtu").read_text().strip())
    except Exception:
        pass
    try:
        speed = (Path(f"/sys/class/net/{iface}/speed").read_text().strip())
    except Exception:
        pass
    return speed, mtu


def check_link(c) -> dict:
    iface = c.get("CX7_IFACE", "enp1s0f1np1")
    rc, out, _ = run("ibdev2netdev")
    up = False
    if rc == 0:
        for line in out.splitlines():
            if iface in line and "Up" in line:
                up = True
                break
    speed, mtu = _iface_speed_mtu(iface)
    if not up:
        return {"status": "error", "label": "ConnectX-7 link",
                "detail": f"{iface} is down. Check cable and netplan."}
    status = "ok"
    notes = []
    if speed and speed != "200000":
        status = "warn"; notes.append(f"speed {speed}Mb/s (expected 200000)")
    if mtu and mtu != "9000":
        status = "warn"; notes.append(f"MTU {mtu} (expected 9000)")
    detail = f"{iface} up at {speed or '?'}Mb/s, MTU {mtu or '?'}"
    if notes:
        detail += " - " + ", ".join(notes)
    return {"status": status, "label": "ConnectX-7 link", "detail": detail}


def check_fabric(c) -> dict:
    worker = c["WORKER_IP"]
    rc, _, _ = run(["ping", "-c", "1", "-W", "2", worker], timeout=6)
    if rc != 0:
        return {"status": "error", "label": "Fabric reachability",
                "detail": f"No response from worker {worker} over the link."}
    # jumbo-frame probe
    rcj, _, _ = run(["ping", "-c", "1", "-W", "2", "-M", "do", "-s", "8972", worker], timeout=6)
    if rcj != 0:
        return {"status": "warn", "label": "Fabric reachability",
                "detail": f"Worker {worker} reachable, but jumbo frames fail. Check MTU on both ends."}
    return {"status": "ok", "label": "Fabric reachability",
            "detail": f"Worker {worker} reachable, jumbo frames OK."}


def check_netplan() -> dict:
    cx7_files = []
    for f in glob.glob("/etc/netplan/*.yaml"):
        try:
            txt = Path(f).read_text()
        except Exception:
            continue
        if "cx7" in f.lower() or re.search(r"en[pP]\dp?\d?s0f\dnp\d", txt):
            cx7_files.append(os.path.basename(f))
    if len(cx7_files) > 1:
        return {"status": "warn", "label": "Netplan config",
                "detail": f"Conflicting files: {', '.join(cx7_files)}. Run Optimize to resolve."}
    if not cx7_files:
        return {"status": "warn", "label": "Netplan config",
                "detail": "No CX7 netplan file found."}
    return {"status": "ok", "label": "Netplan config", "detail": f"Single config: {cx7_files[0]}"}


def check_worker_ssh(c) -> dict:
    rc, _, err = ssh_worker("true", timeout=8)
    if rc == 0:
        return {"status": "ok", "label": "Worker SSH",
                "detail": f'Passwordless SSH to {c["WORKER_SSH_HOST"]} OK.'}
    return {"status": "error", "label": "Worker SSH",
            "detail": f'Cannot reach {c["WORKER_SSH_HOST"]}: {err or "failed"}'}


def check_ray(c) -> dict:
    expected = int(c.get("TENSOR_PARALLEL", "2"))
    py = ("import ray;ray.init(address='auto',logging_level='ERROR');"
          "print(int(ray.cluster_resources().get('GPU',0)))")
    rc, out, _ = run(["docker", "exec", CONTAINER, "python3", "-c", py], timeout=20)
    if rc != 0:
        return {"status": "idle", "label": "Ray cluster",
                "detail": "Cluster not running."}
    try:
        gpus = int(out.strip().splitlines()[-1])
    except Exception:
        gpus = 0
    if gpus == expected:
        return {"status": "ok", "label": "Ray cluster",
                "detail": f"{gpus}/{expected} GPUs registered."}
    return {"status": "warn", "label": "Ray cluster",
            "detail": f"{gpus}/{expected} GPUs registered."}


def check_api(c) -> dict:
    port = c["API_PORT"]
    rc, out, _ = run(["curl", "-s", "--max-time", "4", f"http://localhost:{port}/v1/models"], timeout=6)
    if rc != 0 or not out:
        return {"status": "idle", "label": "vLLM API", "detail": "API not serving yet."}
    try:
        model = json.loads(out)["data"][0]["id"]
        return {"status": "ok", "label": "vLLM API", "detail": f"Serving: {model}"}
    except Exception:
        return {"status": "warn", "label": "vLLM API", "detail": "API up but no model loaded yet."}


def _parse_prom(text: str, name: str):
    """Sum all samples of a Prometheus metric family, ignoring labels."""
    total = 0.0
    found = False
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if line.split("{")[0].split(" ")[0] == name:
            try:
                total += float(line.rsplit(" ", 1)[1])
                found = True
            except (ValueError, IndexError):
                pass
    return total if found else None


def gather_metrics() -> dict:
    """Live serving throughput from vLLM /metrics plus per-node memory/util."""
    c = cfg()
    out = {"serving": {}, "nodes": {}}

    # vLLM exposes Prometheus metrics on the same port as the API
    rc, body, _ = run(["curl", "-s", "--max-time", "4",
                       f"http://localhost:{c['API_PORT']}/metrics"], timeout=6)
    if rc == 0 and body:
        gen = _parse_prom(body, "vllm:generation_tokens_total")
        prompt = _parse_prom(body, "vllm:prompt_tokens_total")
        running = _parse_prom(body, "vllm:num_requests_running")
        waiting = _parse_prom(body, "vllm:num_requests_waiting")
        out["serving"] = {
            "gen_tokens_total": gen,
            "prompt_tokens_total": prompt,
            "requests_running": running,
            "requests_waiting": waiting,
        }

    # Per-node: GPU utilization (nvidia-smi) and unified memory (free).
    # On GB10 UMA, host memory IS the GPU memory, so we report free -m.
    def node_stats(local: bool):
        if local:
            rc1, util, _ = run(["nvidia-smi", "--query-gpu=utilization.gpu",
                                "--format=csv,noheader,nounits"], timeout=6)
            rc2, mem, _ = run(["sh", "-c",
                "free -m | awk '/^Mem:/{print $2,$3,$7}'"], timeout=6)
        else:
            rc1, util, _ = ssh_worker(
                "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits", timeout=8)
            rc2, mem, _ = ssh_worker("free -m | awk '/^Mem:/{print $2,$3,$7}'", timeout=8)
        stats = {"gpu_util": None, "mem_total": None, "mem_used": None, "mem_avail": None}
        if rc1 == 0 and util.strip():
            try:
                stats["gpu_util"] = int(util.strip().splitlines()[0])
            except (ValueError, IndexError):
                pass
        if rc2 == 0 and mem.strip():
            try:
                total, used, avail = mem.strip().split()
                stats.update(mem_total=int(total), mem_used=int(used), mem_avail=int(avail))
            except ValueError:
                pass
        return stats

    out["nodes"]["head"] = node_stats(local=True)
    out["nodes"]["worker"] = node_stats(local=False)
    return out


def gather_status() -> dict:
    c = cfg()
    checks = [
        check_link(c),
        check_fabric(c),
        check_netplan(),
        check_worker_ssh(c),
        check_ray(c),
        check_api(c),
    ]
    # overall: error if any error, warn if any warn, running if api ok
    statuses = [c_["status"] for c_ in checks]
    api_ok = checks[-1]["status"] == "ok"
    ray_ok = checks[-2]["status"] == "ok"
    if "error" in statuses:
        overall = "error"
    elif "warn" in statuses:
        overall = "warn"
    else:
        overall = "ok"
    running = api_ok or ray_ok
    return {
        "overall": overall,
        "running": running,
        "action": _current_action["name"],
        "checks": checks,
        "nodes": {
            "head": {"ip": c["HEAD_IP"], "iface": c.get("CX7_IFACE", "")},
            "worker": {"ip": c["WORKER_IP"], "host": c["WORKER_SSH_HOST"]},
        },
        "config": {k: cfg().get(k, "") for k in CONFIG_FIELDS},
    }


# ------------------------------------------------------------
# Optimize / fix  (idempotent, uses sudo -n for privileged ops)
# ------------------------------------------------------------
def _sudo(cmd_list, timeout=30):
    return run(["sudo", "-n", *cmd_list], timeout=timeout, capture_to_activity=True)


def optimize() -> list:
    c = cfg()
    iface = c.get("CX7_IFACE", "enp1s0f1np1")
    results = []

    def record(name, status, detail):
        results.append({"name": name, "status": status, "detail": detail})
        log_line(f"[optimize] {name}: {detail}")

    # 1. Resolve the netplan conflict (archive the NVIDIA default if both exist)
    default_np = Path("/etc/netplan/40-cx7.yaml")
    ours_np = Path("/etc/netplan/60-cx7-cluster.yaml")
    if default_np.exists() and ours_np.exists():
        rc, _, err = _sudo(["mv", str(default_np), f"/etc/netplan/40-cx7.yaml.disabled"])
        if rc == 0:
            _sudo(["netplan", "apply"])
            record("Netplan conflict", "fixed", "Archived 40-cx7.yaml, applied clean config.")
        else:
            record("Netplan conflict", "error", f"Could not archive: {err or 'permission denied'}")
    else:
        record("Netplan conflict", "ok", "No conflict.")

    # 2. MTU 9000 on the CX7 interface
    _, mtu = _iface_speed_mtu(iface)[1], None
    _, mtu = _iface_speed_mtu(iface)
    if mtu != "9000":
        rc, _, err = _sudo(["ip", "link", "set", iface, "mtu", "9000"])
        record("Jumbo frames", "fixed" if rc == 0 else "error",
               f"Set MTU 9000 on {iface}" if rc == 0 else f"Failed: {err}")
    else:
        record("Jumbo frames", "ok", f"{iface} already MTU 9000.")

    # 3. Disable IPv6 on the CX7 interface (keeps RoCE GID indices consistent)
    key = f"net.ipv6.conf.{iface}.disable_ipv6=1"
    rc, _, err = _sudo(["sysctl", "-w", key])
    record("IPv6 on fabric", "fixed" if rc == 0 else "warn",
           f"Disabled IPv6 on {iface}" if rc == 0 else f"Could not set: {err}")

    # 4. Ensure network sysctl tuning file is loaded
    tuning = Path("/etc/sysctl.d/90-gx10-cluster.conf")
    if tuning.exists():
        _sudo(["sysctl", "--system"])
        record("Network tuning", "ok", "sysctl tuning present and reloaded.")
    else:
        record("Network tuning", "warn", "Tuning file missing. Re-run 01-node-setup.sh.")

    # 5. Drop page caches on both nodes (UMA hygiene before a run)
    rc, _, _ = _sudo(["sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"])
    rcw, _, _ = ssh_worker("sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'")
    record("Free unified memory", "fixed" if rc == 0 else "warn",
           "Dropped caches on head" + (" and worker." if rcw == 0 else "; worker skipped."))

    # 6. Docker + NVIDIA runtime present
    rc, _, _ = run(["docker", "info"], timeout=10)
    record("Docker runtime", "ok" if rc == 0 else "error",
           "Docker reachable." if rc == 0 else "Docker not reachable.")

    return results


def rdma_check() -> dict:
    """On-demand: is NCCL using RDMA (NET/IB) or falling back to sockets?"""
    log_line("[rdma] checking NCCL transport...")
    py = ("import os;os.environ['NCCL_DEBUG']='INFO';import ray;"
          "ray.init(address='auto',logging_level='ERROR');print('ray-ok')")
    rc, out, err = run(["docker", "exec", CONTAINER, "python3", "-c", py], timeout=20)
    blob = (out + "\n" + err)
    if "NET/IB" in blob:
        log_line("[rdma] NET/IB detected - RDMA active.")
        return {"status": "ok", "detail": "NCCL is using RDMA (NET/IB)."}
    if "NET/Socket" in blob:
        log_line("[rdma] NET/Socket detected - TCP fallback.")
        return {"status": "warn",
                "detail": "NCCL fell back to TCP sockets. Check /dev/infiniband mount and NCCL build."}
    return {"status": "idle",
            "detail": "Could not determine transport. Start the cluster, then re-check."}


# ------------------------------------------------------------
# Start / stop  (long running, run in a thread)
# ------------------------------------------------------------
def _run_launch(arg=None):
    name = "stopping" if arg == "stop" else "starting"
    with _action_lock:
        _current_action["name"] = name
        try:
            if not LAUNCH_SCRIPT.exists():
                log_line(f"[error] launch script not found at {LAUNCH_SCRIPT}. "
                         f"Set GX10_KIT_DIR.")
                return
            cmd = ["bash", str(LAUNCH_SCRIPT)] + ([arg] if arg else [])
            log_line(f"[{name}] {' '.join(cmd)}")
            proc = subprocess.Popen(cmd, cwd=str(KIT_DIR), text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in proc.stdout:
                if line.strip():
                    log_line(line.rstrip())
            proc.wait()
            log_line(f"[{name}] finished (exit {proc.returncode}).")
        finally:
            _current_action["name"] = None


def start_cluster():
    if _current_action["name"]:
        return {"ok": False, "detail": f"Busy: {_current_action['name']}"}
    threading.Thread(target=_run_launch, daemon=True).start()
    return {"ok": True, "detail": "Start initiated. Watch the activity log."}


def stop_cluster():
    if _current_action["name"]:
        return {"ok": False, "detail": f"Busy: {_current_action['name']}"}
    threading.Thread(target=_run_launch, args=("stop",), daemon=True).start()
    return {"ok": True, "detail": "Stop initiated."}


def swap_model(model: str):
    c = cfg()
    write_env_value(CLUSTER_ENV, "MODEL", model)
    log_line(f"[model] switching to {model}")
    run(["docker", "exec", CONTAINER, "pkill", "-f", "vllm serve"], timeout=10)
    time.sleep(2)
    serve = (
        f"vllm serve '{model}' --host 0.0.0.0 --port {c['API_PORT']} "
        f"--tensor-parallel-size {c['TENSOR_PARALLEL']} --distributed-executor-backend ray "
        f"--gpu-memory-utilization {c['GPU_MEM_UTIL']} --max-model-len {c['MAX_MODEL_LEN']} "
        f"> /var/log/vllm.log 2>&1"
    )
    rc, _, err = run(["docker", "exec", "-d", CONTAINER, "bash", "-c", serve], timeout=15)
    if rc != 0:
        return {"ok": False, "detail": f"Relaunch failed: {err}. Is the cluster running?"}
    return {"ok": True, "detail": f"Reloading with {model}. Loading takes a few minutes."}


def vllm_logs(lines=120):
    rc, out, _ = run(["docker", "exec", CONTAINER, "tail", "-n", str(lines), "/var/log/vllm.log"], timeout=10)
    if rc != 0:
        return "No vLLM log yet. Container may not be running."
    return out


# ------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------
app = FastAPI(title="GX10 Cluster Panel")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def api_status():
    return JSONResponse(gather_status())


@app.get("/api/metrics")
def api_metrics():
    return JSONResponse(gather_metrics())


@app.get("/api/activity")
def api_activity():
    with _activity_lock:
        return JSONResponse({"lines": list(_activity)})


@app.post("/api/start")
def api_start():
    return JSONResponse(start_cluster())


@app.post("/api/stop")
def api_stop():
    return JSONResponse(stop_cluster())


@app.post("/api/optimize")
def api_optimize():
    return JSONResponse({"results": optimize()})


@app.post("/api/rdma-check")
def api_rdma():
    return JSONResponse(rdma_check())


@app.get("/api/logs")
def api_logs():
    return JSONResponse({"log": vllm_logs()})


@app.post("/api/config")
def api_config(payload: dict = Body(...)):
    applied = {}
    for key, cast in CONFIG_FIELDS.items():
        if key in payload and str(payload[key]) != "":
            try:
                value = cast(payload[key])
            except (ValueError, TypeError):
                return JSONResponse({"ok": False, "detail": f"Invalid value for {key}"}, status_code=400)
            write_env_value(CLUSTER_ENV, key, str(value))
            applied[key] = value
    log_line(f"[config] updated {', '.join(applied)} (restart to apply)")
    return JSONResponse({"ok": True, "applied": applied,
                         "detail": "Saved. Stop and start the cluster to apply."})


@app.post("/api/model")
def api_model(payload: dict = Body(...)):
    model = (payload.get("model") or "").strip()
    if not model:
        return JSONResponse({"ok": False, "detail": "No model provided"}, status_code=400)
    return JSONResponse(swap_model(model))


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    log_line(f"Panel starting. Kit dir: {KIT_DIR}")
    uvicorn.run(app, host=PANEL_HOST, port=PANEL_PORT, log_level="warning")
