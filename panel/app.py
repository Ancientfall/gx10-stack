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
import socket
import sqlite3
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
        "HF_CACHE_DIR": "/data/hf-cache",
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


# ------------------------------------------------------------
# Metrics store: SQLite history for throughput, GPU, memory, thermals, cost
# ------------------------------------------------------------
DB_PATH = Path(os.environ.get("GX10_DB", str(BASE_DIR / "gx10-metrics.db")))
_db_lock = threading.Lock()

# Cost comparison rate: $ per 1M generated tokens (configurable). Default ~ frontier model.
COST_PER_MTOK = float(os.environ.get("GX10_COST_PER_MTOK", "10.0"))


def db_init():
    with _db_lock, sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS samples (
                ts INTEGER PRIMARY KEY,
                tps REAL, gen_total REAL,
                head_gpu INTEGER, head_mem_used INTEGER, head_mem_total INTEGER,
                head_temp INTEGER, head_power REAL,
                worker_gpu INTEGER, worker_mem_used INTEGER, worker_mem_total INTEGER,
                worker_temp INTEGER, worker_power REAL,
                model TEXT
            )""")
        con.execute("""
            CREATE TABLE IF NOT EXISTS benchmarks (
                ts INTEGER PRIMARY KEY,
                model TEXT, concurrency INTEGER, prompt_tokens INTEGER,
                max_tokens INTEGER, ttft_ms REAL, tps REAL,
                total_tokens INTEGER, duration_s REAL, notes TEXT
            )""")
        con.commit()


def db_insert_sample(row: dict):
    cols = ("ts", "tps", "gen_total", "head_gpu", "head_mem_used", "head_mem_total",
            "head_temp", "head_power", "worker_gpu", "worker_mem_used", "worker_mem_total",
            "worker_temp", "worker_power", "model")
    with _db_lock, sqlite3.connect(DB_PATH) as con:
        con.execute(
            f"INSERT OR REPLACE INTO samples ({','.join(cols)}) VALUES ({','.join('?'*len(cols))})",
            tuple(row.get(c) for c in cols))
        con.commit()


def db_history(since_seconds: int, max_points: int = 240):
    """Return downsampled history for charts."""
    cutoff = int(time.time()) - since_seconds
    with _db_lock, sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM samples WHERE ts >= ? ORDER BY ts", (cutoff,)).fetchall()
    rows = [dict(r) for r in rows]
    # downsample evenly if too many points
    if len(rows) > max_points:
        step = len(rows) / max_points
        rows = [rows[int(i * step)] for i in range(max_points)]
    return rows


def db_cost_summary():
    """Tokens generated and cost-equivalent over windows.

    vLLM's generation_tokens_total is cumulative since the engine started and
    resets to 0 whenever the model is reloaded. A plain MAX-MIN would drop an
    entire segment after any restart (and can read as 0 across a reset), so sum
    the positive deltas between consecutive samples instead; a reset shows up as
    a negative delta we simply skip.
    """
    now = int(time.time())
    out = {"rate_per_mtok": COST_PER_MTOK}
    with _db_lock, sqlite3.connect(DB_PATH) as con:
        for label, secs in (("today", 86400), ("week", 604800), ("all", 10**12)):
            cutoff = now - secs
            rows = con.execute(
                "SELECT gen_total FROM samples WHERE ts >= ? AND gen_total IS NOT NULL ORDER BY ts",
                (cutoff,)).fetchall()
            tokens = 0.0
            prev = None
            for (g,) in rows:
                if prev is not None and g > prev:
                    tokens += g - prev
                prev = g
            out[label] = {
                "tokens": int(tokens),
                "cost_equiv": round(tokens / 1_000_000 * COST_PER_MTOK, 2),
            }
    return out


# Background collector: samples every interval into SQLite
_collector_stop = threading.Event()


def _collect_once():
    try:
        m = gather_metrics()
    except Exception:
        return
    s = m.get("serving", {})
    h = m.get("nodes", {}).get("head", {})
    w = m.get("nodes", {}).get("worker", {})
    # tps from gen-token delta vs the previous sample
    gen = s.get("gen_tokens_total")
    tps = None
    prev = getattr(_collect_once, "_prev", None)
    nowt = time.time()
    if gen is not None and prev is not None:
        dt = nowt - prev[1]
        if dt > 0:
            tps = max(0.0, (gen - prev[0]) / dt)
    if gen is not None:
        _collect_once._prev = (gen, nowt)
    db_insert_sample({
        "ts": int(nowt), "tps": tps, "gen_total": gen,
        "head_gpu": h.get("gpu_util"), "head_mem_used": h.get("mem_used"),
        "head_mem_total": h.get("mem_total"), "head_temp": h.get("temp"), "head_power": h.get("power"),
        "worker_gpu": w.get("gpu_util"), "worker_mem_used": w.get("mem_used"),
        "worker_mem_total": w.get("mem_total"), "worker_temp": w.get("temp"), "worker_power": w.get("power"),
        "model": cfg().get("MODEL", ""),
    })


def _collector_loop(interval=10):
    db_init()
    while not _collector_stop.wait(interval):
        _collect_once()


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
        # full nvidia-smi telemetry in one call
        fields = ("utilization.gpu,temperature.gpu,power.draw,power.limit,"
                  "clocks.sm,clocks.mem,utilization.memory,"
                  "memory.used,memory.total,fan.speed,pstate")
        smi_q = f"nvidia-smi --query-gpu={fields} --format=csv,noheader,nounits"
        if local:
            rc1, gpu, _ = run(["sh", "-c", smi_q], timeout=6)
            rc2, mem, _ = run(["sh", "-c",
                "free -m | awk '/^Mem:/{print $2,$3,$7}'"], timeout=6)
        else:
            rc1, gpu, _ = ssh_worker(smi_q, timeout=8)
            rc2, mem, _ = ssh_worker("free -m | awk '/^Mem:/{print $2,$3,$7}'", timeout=8)
        stats = {"gpu_util": None, "temp": None, "power": None, "power_limit": None,
                 "clock_sm": None, "clock_mem": None, "vram_util": None,
                 "vram_used": None, "vram_total": None, "fan": None, "pstate": None,
                 "mem_total": None, "mem_used": None, "mem_avail": None}
        if rc1 == 0 and gpu.strip():
            try:
                p = [x.strip() for x in gpu.strip().splitlines()[0].split(",")]
                def num(v, cast=float):
                    try:
                        return cast(float(v))
                    except (ValueError, TypeError):
                        return None
                stats["gpu_util"] = num(p[0], int)
                stats["temp"] = num(p[1], int)
                stats["power"] = num(p[2])
                stats["power_limit"] = num(p[3])
                stats["clock_sm"] = num(p[4], int)
                stats["clock_mem"] = num(p[5], int)
                stats["vram_util"] = num(p[6], int)
                stats["vram_used"] = num(p[7], int)
                stats["vram_total"] = num(p[8], int)
                stats["fan"] = num(p[9], int)
                stats["pstate"] = p[10] if len(p) > 10 and p[10] not in ("", "[N/A]") else None
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

    # recommend single vs two-node serving from the current model's memory fit
    model = c.get("MODEL", "")
    params = next((m["params_b"] for m in CURATED_MODELS if m["id"] == model), None) \
        or _guess_params_from_name(model)
    fit = _fit_label(params)["fit"] if params else "unknown"
    recommended = "single" if fit == "single" else ("cluster" if fit in ("cluster", "toobig") else "")
    try:
        cur_tp = int(c.get("TENSOR_PARALLEL", "2") or 2)
    except ValueError:
        cur_tp = 2

    return {
        "overall": overall,
        "running": running,
        "action": _current_action["name"],
        "checks": checks,
        "nodes": {
            "head": {"ip": c["HEAD_IP"], "iface": c.get("CX7_IFACE", ""), "host": socket.gethostname()},
            "worker": {"ip": c["WORKER_IP"], "host": c["WORKER_SSH_HOST"]},
        },
        "config": {k: cfg().get(k, "") for k in CONFIG_FIELDS},
        "serving_mode": {
            "current": "single" if cur_tp <= 1 else "cluster",
            "recommended": recommended,
            "fit": fit,
        },
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


# ------------------------------------------------------------
# Download manager: tracked background download, then auto-load
# ------------------------------------------------------------
_download = {
    "active": False, "model": None, "status": "idle", "percent": 0,
    "detail": "", "speed": "", "started": None, "error": None,
}
_download_lock = threading.Lock()


def _set_dl(**kw):
    with _download_lock:
        _download.update(kw)


def download_status() -> dict:
    with _download_lock:
        return dict(_download)


def _start_vllm(model: str, max_len_override: int = None) -> tuple:
    """Relaunch vLLM serve in the running container with the given model.

    Tensor parallel selects the execution backend: TP>=2 shards the model
    across both GB10s via Ray, while TP=1 runs single-node on the head GPU
    with the multiprocessing backend (no Ray). Single-node is faster to start
    and avoids Ray scheduling the lone shard onto the worker box.
    """
    c = cfg()
    max_len = max_len_override if max_len_override else c["MAX_MODEL_LEN"]
    try:
        tp = int(c.get("TENSOR_PARALLEL", "2") or 2)
    except ValueError:
        tp = 2
    backend = "ray" if tp > 1 else "mp"
    run(["docker", "exec", CONTAINER, "pkill", "-f", "vllm serve"], timeout=10)
    time.sleep(2)
    serve = (
        f"vllm serve '{model}' --host 0.0.0.0 --port {c['API_PORT']} "
        f"--tensor-parallel-size {tp} --distributed-executor-backend {backend} "
        f"--gpu-memory-utilization {c['GPU_MEM_UTIL']} --max-model-len {max_len} "
        f"> /var/log/vllm.log 2>&1"
    )
    return run(["docker", "exec", "-d", CONTAINER, "bash", "-c", serve], timeout=15)


def _explain_error(log: str) -> str:
    """Turn a raw vLLM error into a plain-language reason."""
    low = log.lower()
    if "keyerror" in low and ("_scale" in low or "experts" in low):
        return "Model uses a quantization format this vLLM build can't load (missing weight mapping). Needs a newer vLLM image."
    if "derived max_model_len" in low:
        return "Context length too high for this model (auto-correcting)."
    if "out of memory" in low or "oom" in low or "cuda error" in low:
        return "Out of memory. Lower GPU mem util or max model length, or pick a smaller model."
    if "unsupported" in low and "quant" in low:
        return "Quantization format not supported by this vLLM build."
    if "no module named" in low or "importerror" in low:
        return "Model needs a dependency this image doesn't have. Needs a newer vLLM image."
    if "trust_remote_code" in low:
        return "Model requires trust_remote_code, which isn't enabled."
    # fall back to the most specific error line
    for l in log.splitlines()[::-1]:
        ll = l.lower()
        if any(k in ll for k in ("error:", "valueerror", "runtimeerror", "keyerror", "assert")):
            return re.sub(r"^.*?((Value|Runtime|Key|OS)Error|AssertionError|Error)", r"\1", l).strip()[:200]
    return "Engine failed to start. See vLLM logs for details."


def _model_matches(served: str, requested: str) -> bool:
    """True if vLLM's served id is the model we asked for (tolerating path vs repo-id)."""
    if not served or not requested:
        return False
    s, r = served.strip().lower(), requested.strip().lower()
    return s == r or s.rsplit("/", 1)[-1] == r.rsplit("/", 1)[-1]


def _vllm_dead() -> bool:
    """True only when we're sure no vllm serve process is running.

    pgrep exit codes: 0 = a process matched, 1 = none matched, anything else
    (2/3 usage, 126/127 missing) = unknown. Only rc==1 is a confident "dead";
    on unknown we assume alive so a missing pgrep can't cause a false failure.
    """
    rc, _, _ = run(["docker", "exec", CONTAINER, "pgrep", "-f", "vllm serve"], timeout=8)
    return rc == 1


def _watch_vllm_until_ready(model: str, timeout_s: int = 1200, _retried=False) -> bool:
    """Poll until the model serves (True) or the engine dies / times out (False).

    Success is the API serving the requested model. Failure is declared only
    when the vllm serve process has actually exited - non-fatal tracebacks in
    the log (kernel fallbacks, quantization warnings) are common during load,
    especially for newer formats like NVFP4, and must not be mistaken for a
    crash or a model that loads fine gets reported as failed.
    """
    c = cfg()
    port = c["API_PORT"]
    start = time.time()
    phase = "loading"
    while time.time() - start < timeout_s:
        # 1) success: API is serving our model
        rc, out, _ = run(["curl", "-s", "--max-time", "3", f"http://localhost:{port}/v1/models"], timeout=5)
        if rc == 0 and out:
            try:
                served = json.loads(out)["data"][0]["id"]
                if _model_matches(served, model):
                    _set_dl(active=False, status="ready", percent=100,
                            detail=f"{model} is live and serving.", error=None)
                    log_line(f"[model] {model} is ready and serving")
                    return True
            except Exception:
                pass

        # 2) read the recent log for phase hints + a one-time context-length fix
        rc2, log, _ = run(["docker", "exec", CONTAINER, "tail", "-n", "60", "/var/log/vllm.log"], timeout=8)
        log = log if (rc2 == 0 and log) else ""
        low = log.lower()
        mlen = re.search(r"derived max_model_len \(max_position_embeddings=(\d+)", log)
        if mlen and not _retried:
            real_max = int(mlen.group(1))
            log_line(f"[model] {model} caps context at {real_max}; retrying")
            _set_dl(active=True, status="loading",
                    detail=f"{model} max context is {real_max}, retrying...")
            _start_vllm(model, max_len_override=real_max)
            _model_maxlen_cache[model] = real_max
            return _watch_vllm_until_ready(model, timeout_s=timeout_s, _retried=True)

        # 3) failure: only when the engine process has actually exited. Grace the
        #    first 25s so the pkill->relaunch gap and slow startup don't trip it.
        if (time.time() - start) > 25 and _vllm_dead():
            reason = _explain_error(log) if log else "Engine exited during startup. See vLLM logs."
            _set_dl(active=False, status="error", error=reason, detail=reason)
            log_line(f"[model] load FAILED: {reason}")
            return False

        if "capturing" in low or "graph" in low or "compile" in low:
            phase = "compiling"
        elif "loading safetensors" in low or "loading weights" in low:
            phase = "loading weights"
        elapsed = int(time.time() - start)
        _set_dl(active=True, status=phase, detail=f"{model}: {phase} ({elapsed}s)")
        time.sleep(5)
    _set_dl(active=False, status="error", error="Timed out waiting for the model to load.",
            detail="Load timed out. Check the vLLM logs.")
    return False


# remembers per-model context caps discovered via auto-correction
_model_maxlen_cache = {}


def _download_worker(model: str, auto_load: bool):
    """Download weights into the cache with progress, then optionally load + watch."""
    c = cfg()
    token = c.get("HF_TOKEN", "")
    _set_dl(active=True, model=model, status="downloading", percent=0,
            detail=f"Preparing to download {model}...", speed="", error=None, started=time.time())
    log_line(f"[download] starting {model}")

    env_prefix = f"HF_TOKEN={token} " if token else ""
    cmd = f"{env_prefix}hf download '{model}' 2>&1"

    proc = subprocess.Popen(
        ["docker", "exec", CONTAINER, "bash", "-lc", cmd],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    last_pct = 0
    pat_pct = re.compile(r'(\d+)%')
    pat_speed = re.compile(r'([\d.]+[KMG]B/s)')
    saw_error = None
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        low = line.lower()
        if "401" in line or "gated" in low or "awaiting a review" in low or "restricted" in low or "authentication" in low:
            saw_error = "Model is gated. Set HF_TOKEN in cluster.env and request access on Hugging Face."
        elif "404" in line or "repository not found" in low or "not found" in low:
            saw_error = "Model not found. Check the exact repo id."
        elif "no space left" in low:
            saw_error = "Out of disk space in the cache volume."
        m = pat_pct.search(line)
        if m:
            pct = int(m.group(1))
            if pct >= last_pct:
                last_pct = pct
                sp = pat_speed.search(line)
                _set_dl(percent=pct, status="downloading",
                        detail=f"Downloading {model}", speed=sp.group(1) if sp else "")
    proc.wait()

    if saw_error or proc.returncode != 0:
        msg = saw_error or f"Download failed (exit {proc.returncode}). See activity log."
        _set_dl(active=False, status="error", error=msg, detail=msg)
        log_line(f"[download] FAILED {model}: {msg}")
        return

    _set_dl(percent=100, status="downloaded", detail=f"{model} downloaded.")
    log_line(f"[download] complete {model}")

    if auto_load:
        write_env_value(CLUSTER_ENV, "MODEL", model)
        _set_dl(active=True, status="loading", detail=f"Starting {model} on the cluster...")
        log_line(f"[model] auto-loading {model}")
        rc, _, err = _start_vllm(model, _model_maxlen_cache.get(model))
        if rc != 0:
            _set_dl(active=False, status="error", error=f"Load failed: {err}")
            return
        _watch_vllm_until_ready(model)
    else:
        _set_dl(active=False, status="downloaded")


def currently_serving() -> str:
    """What the API is actually serving right now, or '' if nothing."""
    c = cfg()
    rc, out, _ = run(["curl", "-s", "--max-time", "3", f"http://localhost:{c['API_PORT']}/v1/models"], timeout=5)
    if rc == 0 and out:
        try:
            return json.loads(out)["data"][0]["id"]
        except Exception:
            return ""
    return ""


def _load_worker(model: str):
    """Load a cached model with rollback: if it fails, restore the previous one."""
    previous = currently_serving()
    write_env_value(CLUSTER_ENV, "MODEL", model)
    _set_dl(active=True, model=model, status="loading", percent=0,
            detail=f"Starting {model}...", speed="", error=None, started=time.time())
    log_line(f"[model] loading {model}" + (f" (was {previous})" if previous else ""))
    rc, _, err = _start_vllm(model, _model_maxlen_cache.get(model))
    if rc != 0:
        _set_dl(active=False, status="error", error=f"Could not start: {err}")
        return
    ok = _watch_vllm_until_ready(model)
    if ok:
        return
    # failed: roll back to the previous working model if there was one
    if previous and previous != model:
        log_line(f"[model] rolling back to {previous}")
        cur = download_status()
        _set_dl(active=True, model=previous, status="loading",
                detail=f"Load failed. Rolling back to {previous}...",
                error=cur.get("error"))
        write_env_value(CLUSTER_ENV, "MODEL", previous)
        rc2, _, _ = _start_vllm(previous, _model_maxlen_cache.get(previous))
        if rc2 == 0 and _watch_vllm_until_ready(previous):
            _set_dl(active=False, status="error",
                    error=f"{model} failed to load. Rolled back to {previous}.",
                    detail=f"{model} failed. {previous} restored and serving.")
            log_line(f"[model] rolled back to {previous} after {model} failed")


def start_download(model: str, auto_load: bool = True):
    with _download_lock:
        if _download["active"]:
            return {"ok": False, "detail": f"A job is already running: {_download['model']}"}
    threading.Thread(target=_download_worker, args=(model, auto_load), daemon=True).start()
    return {"ok": True, "detail": f"Download started for {model}."}


def swap_model(model: str):
    """Cached -> load + watch (with rollback). Not cached -> download + auto-load."""
    with _download_lock:
        if _download["active"]:
            return {"ok": False, "detail": f"Busy: {_download['model']} ({_download['status']})"}
    cached = {m["id"] for m in hf_cache_models()}
    if model in cached:
        threading.Thread(target=_load_worker, args=(model,), daemon=True).start()
        return {"ok": True, "detail": f"Loading cached {model}..."}
    threading.Thread(target=_download_worker, args=(model, True), daemon=True).start()
    return {"ok": True, "detail": f"Downloading then loading {model}..."}


def _ray_gpu_count() -> int:
    """How many GPUs are currently registered with the running Ray cluster."""
    py = ("import ray;ray.init(address='auto',logging_level='ERROR');"
          "print(int(ray.cluster_resources().get('GPU',0)))")
    rc, out, _ = run(["docker", "exec", CONTAINER, "python3", "-c", py], timeout=15)
    if rc != 0:
        return 0
    try:
        return int(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return 0


def _stop_worker_container() -> bool:
    """Best-effort: remove the vLLM/Ray container on the worker so its GPU idles."""
    c = cfg()
    target = f'{c["CLUSTER_USER"]}@{c["WORKER_SSH_HOST"]}'
    rc, _, _ = run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                    target, f"docker rm -f {CONTAINER}"], timeout=20)
    return rc == 0


def _apply_mode_reload(model: str, mode: str):
    """Reload the live model under the new mode; if single, idle the worker after."""
    _load_worker(model)
    if mode == "single" and currently_serving() == model and _stop_worker_container():
        log_line("[mode] worker container stopped - second node idle")


def set_serving_mode(mode: str) -> dict:
    """Switch between single-node (TP=1, mp, worker idle) and two-node (TP=2, ray).

    Persists TENSOR_PARALLEL. If a model is serving, reloads it under the new
    mode (rollback on failure). Single-node also stops the worker container so
    the second GX10 idles; switching back to two-node needs a Stop/Start to
    rebuild the Ray cluster, so we say so rather than attempt a doomed reload.
    """
    tp = 1 if mode == "single" else 2
    label = "single node" if mode == "single" else "both nodes (TP=2)"
    write_env_value(CLUSTER_ENV, "TENSOR_PARALLEL", str(tp))
    log_line(f"[mode] serving mode -> {label}")
    serving = currently_serving()

    # Going back to two-node needs the worker rejoined to Ray; a hot reload
    # can't do that, so ask for a relaunch when the worker is idle.
    if mode == "cluster" and serving and _ray_gpu_count() < 2:
        return {"ok": True, "tp": tp, "reloaded": False,
                "detail": "Set to both nodes. The worker is idle - Stop then Start to bring it back and shard across both."}

    if not serving:
        if mode == "single" and _stop_worker_container():
            log_line("[mode] worker container stopped - second node idle")
        return {"ok": True, "tp": tp, "reloaded": False,
                "detail": f"Set to {label}. Applies when you next load or start a model."}

    with _download_lock:
        if _download["active"]:
            return {"ok": False, "tp": tp,
                    "detail": f"Busy: {_download['model']} ({_download['status']}). Try again when it finishes."}
    threading.Thread(target=_apply_mode_reload, args=(serving, mode), daemon=True).start()
    extra = " and idling the worker" if mode == "single" else ""
    return {"ok": True, "tp": tp, "reloaded": True,
            "detail": f"Switching to {label}{extra} and reloading {serving}…"}


def preload_check(model: str) -> dict:
    """Tell the user what will happen before they load a model."""
    c = cfg()
    cached_ids = {m["id"] for m in hf_cache_models()}
    is_cached = model in cached_ids
    serving = currently_serving()
    # fit estimate from curated/known params or name guess
    params = next((m["params_b"] for m in CURATED_MODELS if m["id"] == model), None) \
        or _guess_params_from_name(model)
    fit = _fit_label(params) if params else {"fit": "unknown", "footprint_gb": None,
                                             "note": "Size unknown; check the model card."}
    warnings = []
    notes = []
    if not is_cached:
        notes.append("Not cached yet, it will download first (can take minutes).")
    else:
        notes.append("Already cached, loads without downloading.")
    if serving:
        warnings.append(f"This replaces the model currently serving ({serving}). It will be unloaded.")
    if fit["fit"] == "toobig":
        warnings.append(f"Estimated footprint (~{fit['footprint_gb']}GB) exceeds your pooled memory. Likely won't load.")
    elif fit["fit"] == "cluster":
        notes.append("Needs both nodes (TP=2).")
    elif fit["fit"] == "single":
        notes.append("Fits on a single node.")
    # known-risky formats
    low = model.lower()
    if "nvfp4" in low or "fp4" in low or "qwen3.5" in low or "qwen3.6" in low:
        warnings.append("Uses a newer quantization/architecture that this vLLM build may not support. May fail to load.")
    return {
        "model": model, "cached": is_cached, "currently_serving": serving,
        "fit": fit["fit"], "footprint_gb": fit["footprint_gb"],
        "warnings": warnings, "notes": notes,
        "ok_to_load": fit["fit"] != "toobig",
    }


def unload_model():
    """Stop vLLM serving (frees memory). Cluster/Ray stays up."""
    with _download_lock:
        if _download["active"]:
            return {"ok": False, "detail": "A job is running; wait for it to finish first."}
    rc, _, _ = run(["docker", "exec", CONTAINER, "pkill", "-f", "vllm serve"], timeout=10)
    _set_dl(active=False, status="idle", model=None, percent=0, detail="", error=None)
    log_line("[model] unloaded (vLLM serve stopped)")
    return {"ok": True, "detail": "Model unloaded. Memory freed. Load one to resume serving."}


def dismiss_status():
    """Clear a finished/errored job banner."""
    with _download_lock:
        if not _download["active"]:
            _download.update(status="idle", model=None, percent=0, detail="", error=None)
    return {"ok": True}


def vllm_logs(lines=120):
    rc, out, _ = run(["docker", "exec", CONTAINER, "tail", "-n", str(lines), "/var/log/vllm.log"], timeout=10)
    if rc != 0:
        return "No vLLM log yet. Container may not be running."
    return out


# ------------------------------------------------------------
# Model management: curated list, HF search, cache, fit, test
# ------------------------------------------------------------

# Curated GB10-friendly models. Edit freely. params_b = billions of params.
CURATED_MODELS = [
    {"id": "openai/gpt-oss-120b", "params_b": 120, "note": "MoE, mxfp4. Strong general + reasoning.", "gated": False},
    {"id": "openai/gpt-oss-20b", "params_b": 20, "note": "Smaller gpt-oss. Fast, single-node capable.", "gated": False},
    {"id": "Qwen/Qwen3-72B-Instruct", "params_b": 72, "note": "Strong all-rounder, long context.", "gated": False},
    {"id": "Qwen/Qwen3-32B", "params_b": 32, "note": "Fast, fits comfortably.", "gated": False},
    {"id": "Qwen/Qwen3-Coder-30B-A3B-Instruct", "params_b": 30, "note": "MoE coder, low active params.", "gated": False},
    {"id": "meta-llama/Llama-3.3-70B-Instruct", "params_b": 70, "note": "Meta flagship. Needs HF token.", "gated": True},
    {"id": "mistralai/Mistral-Small-3.2-24B-Instruct-2506", "params_b": 24, "note": "Efficient, capable.", "gated": False},
]

# Pooled memory budget across both GB10s (GB). 128 each, reserve headroom.
POOLED_GB = 256
USABLE_GB = 205          # after OS + KV cache + activation overhead
SINGLE_NODE_GB = 110     # what fits on one box alone


def _fit_label(params_b, quantized=True):
    """Rough footprint: ~1 byte/param for fp8/mxfp4, ~2 for fp16. Plus KV/activation slack."""
    bytes_per = 1.0 if quantized else 2.0
    weight_gb = params_b * bytes_per
    footprint = weight_gb * 1.25  # KV cache + activations headroom
    if footprint <= SINGLE_NODE_GB:
        return {"fit": "single", "footprint_gb": round(footprint), "note": "Fits on one node (fastest)."}
    if footprint <= USABLE_GB:
        return {"fit": "cluster", "footprint_gb": round(footprint), "note": "Needs both nodes (TP=2)."}
    return {"fit": "toobig", "footprint_gb": round(footprint), "note": "Exceeds pooled memory."}


def hf_cache_models() -> list:
    """List models already downloaded to HF_CACHE_DIR (so switching is fast)."""
    c = cfg()
    cache = Path(c.get("HF_CACHE_DIR", "")) / "hub"
    out = []
    if cache.is_dir():
        for d in cache.glob("models--*"):
            # models--org--name -> org/name
            name = d.name.replace("models--", "").replace("--", "/", 1).replace("--", "-")
            size_gb = None
            try:
                total = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                size_gb = round(total / 1e9, 1)
            except Exception:
                pass
            out.append({"id": name, "size_gb": size_gb})
    return out


def curated_models() -> list:
    cached = {m["id"] for m in hf_cache_models()}
    current = cfg().get("MODEL", "")
    result = []
    for m in CURATED_MODELS:
        fit = _fit_label(m["params_b"])
        result.append({**m, **fit, "cached": m["id"] in cached, "active": m["id"] == current})
    return result


def hf_search(query: str, limit: int = 60) -> list:
    """Live Hugging Face model search. Public API, no auth needed for public models."""
    if not query.strip():
        return []
    url = f"https://huggingface.co/api/models?search={query}&filter=text-generation&sort=downloads&direction=-1&limit={limit}"
    rc, body, _ = run(["curl", "-s", "--max-time", "8", url], timeout=12)
    if rc != 0 or not body:
        return []
    try:
        items = json.loads(body)
    except Exception:
        return []
    cached = {m["id"] for m in hf_cache_models()}
    results = []
    for it in items:
        mid = it.get("id") or it.get("modelId") or ""
        if not mid:
            continue
        # estimate params from the name (e.g. "-70B", "-7b", "-A3B")
        params_b = _guess_params_from_name(mid)
        fit = _fit_label(params_b) if params_b else {"fit": "unknown", "footprint_gb": None, "note": "Size unknown; check model card."}
        results.append({
            "id": mid,
            "downloads": it.get("downloads", 0),
            "likes": it.get("likes", 0),
            "gated": bool(it.get("gated", False)),
            "params_b": params_b,
            "cached": mid in cached,
            **fit,
        })
    return results


def _guess_params_from_name(name: str):
    """Best-effort parameter count from a model id like 'Qwen/Qwen3-72B-Instruct'."""
    import re as _re
    m = _re.search(r'(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])', name)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def test_prompt(prompt: str, max_tokens: int = 128) -> dict:
    """Send one prompt to the live model and time it, for a quick speed/health probe."""
    c = cfg()
    model = c.get("MODEL", "")
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    })
    t0 = time.time()
    rc, body, _ = run(["curl", "-s", "--max-time", "180",
                       f"http://localhost:{c['API_PORT']}/v1/chat/completions",
                       "-H", "Content-Type: application/json", "-d", payload], timeout=185)
    elapsed = time.time() - t0
    if rc != 0 or not body:
        return {"ok": False, "detail": "No response. Is a model loaded and serving?"}
    try:
        data = json.loads(body)
        text = data["choices"][0]["message"]["content"]
        completion = data.get("usage", {}).get("completion_tokens", max_tokens)
        tps = round(completion / elapsed, 1) if elapsed > 0 else 0
        return {"ok": True, "model": model, "text": text, "tokens": completion,
                "seconds": round(elapsed, 2), "tps": tps}
    except Exception as exc:
        return {"ok": False, "detail": f"Parse error: {exc}"}


# ------------------------------------------------------------
# Image / vLLM version manager: pull newer NGC base, rebuild, switch, rollback
# ------------------------------------------------------------
NGC_REPO = "nvcr.io/nvidia/vllm"
LOCAL_IMAGE_TAG = "gx10/vllm-ray"

_image_job = {"active": False, "status": "idle", "percent": 0, "detail": "", "error": None, "target": None}
_image_lock = threading.Lock()


def _set_img(**kw):
    with _image_lock:
        _image_job.update(kw)


def image_status() -> dict:
    with _image_lock:
        return dict(_image_job)


def current_vllm_version() -> dict:
    ver, base = None, cfg().get("VLLM_IMAGE", "")
    rc, out, _ = run(["docker", "exec", CONTAINER, "python3", "-c",
                      "import vllm; print(vllm.__version__)"], timeout=10)
    if rc == 0 and out.strip():
        ver = out.strip().splitlines()[-1]
    return {"vllm_version": ver, "image": base}


def list_local_images() -> list:
    rc, out, _ = run(["docker", "images", LOCAL_IMAGE_TAG, "--format", "{{.Tag}}|{{.Size}}|{{.CreatedSince}}"], timeout=10)
    imgs = []
    if rc == 0:
        for line in out.splitlines():
            parts = line.split("|")
            if len(parts) >= 3:
                imgs.append({"tag": parts[0], "size": parts[1], "created": parts[2]})
    return imgs


def check_ngc_tag(tag: str) -> dict:
    rc, out, err = run(["docker", "manifest", "inspect", f"{NGC_REPO}:{tag}"], timeout=25)
    blob = (out + err).lower()
    if rc == 0:
        return {"tag": tag, "available": True}
    if "no such manifest" in blob or "not found" in blob or "manifest unknown" in blob:
        return {"tag": tag, "available": False, "reason": "not released"}
    if "unauthorized" in blob or "denied" in blob:
        return {"tag": tag, "available": False, "reason": "needs docker login nvcr.io"}
    return {"tag": tag, "available": False, "reason": "unreachable"}


def _image_build_worker(ngc_tag: str):
    c = cfg()
    local = f"{LOCAL_IMAGE_TAG}:{ngc_tag}"
    base = f"{NGC_REPO}:{ngc_tag}"
    worker_host = c["WORKER_SSH_HOST"]
    user = c["CLUSTER_USER"]
    _set_img(active=True, status="pulling", percent=0, error=None, target=ngc_tag,
             detail=f"Pulling {base} on head...")
    log_line(f"[image] pulling {base} on both nodes")

    def pull(host_cmd, where):
        proc = subprocess.Popen(host_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        pat = re.compile(r'(\d+)%')
        for line in proc.stdout:
            m = pat.search(line)
            if m:
                _set_img(status="pulling", percent=int(m.group(1)), detail=f"Pulling base on {where}...")
        proc.wait()
        return proc.returncode

    if pull(["docker", "pull", base], "head") != 0:
        _set_img(active=False, status="error", error=f"Failed to pull {base} on head."); return
    _set_img(status="pulling", percent=50, detail="Pulling base on worker...")
    if pull(["ssh", "-o", "BatchMode=yes", f"{user}@{worker_host}", f"docker pull {base}"], "worker") != 0:
        _set_img(active=False, status="error", error=f"Failed to pull {base} on worker."); return

    _set_img(status="building", percent=70, detail="Building ray layer on head...")
    log_line(f"[image] building {local} on both nodes")
    dockerfile = str(KIT_DIR / "Dockerfile.ray")
    rcb1, _, eb1 = run(["docker", "build", "-f", dockerfile, "--build-arg", f"BASE={base}",
                        "-t", local, str(KIT_DIR)], timeout=900, capture_to_activity=True)
    if rcb1 != 0:
        _set_img(active=False, status="error", error=f"Build failed on head: {eb1[:120]}"); return
    _set_img(status="building", percent=88, detail="Building ray layer on worker...")
    run(["ssh", "-o", "BatchMode=yes", f"{user}@{worker_host}", "mkdir -p ~/.gx10-build"], timeout=15)
    run(["scp", dockerfile, f"{user}@{worker_host}:~/.gx10-build/Dockerfile.ray"], timeout=20)
    rcb2, _, _ = run(["ssh", "-o", "BatchMode=yes", f"{user}@{worker_host}",
                      f"docker build -f ~/.gx10-build/Dockerfile.ray --build-arg BASE={base} -t {local} ~/.gx10-build"],
                     timeout=900)
    if rcb2 != 0:
        _set_img(active=False, status="error", error="Build failed on worker."); return

    _set_img(active=False, status="built", percent=100,
             detail=f"{local} built on both nodes. Switch the cluster to use it.")
    log_line(f"[image] {local} ready on both nodes")


def start_image_update(ngc_tag: str):
    with _image_lock:
        if _image_job["active"]:
            return {"ok": False, "detail": "An image job is already running."}
    chk = check_ngc_tag(ngc_tag)
    if not chk["available"]:
        return {"ok": False, "detail": f"Tag {ngc_tag} unavailable: {chk.get('reason')}"}
    threading.Thread(target=_image_build_worker, args=(ngc_tag,), daemon=True).start()
    return {"ok": True, "detail": f"Updating to {ngc_tag}. Pulls ~19GB per node."}


def switch_image(ngc_tag: str):
    local = f"{LOCAL_IMAGE_TAG}:{ngc_tag}"
    rc, _, _ = run(["docker", "image", "inspect", local], timeout=10)
    if rc != 0:
        return {"ok": False, "detail": f"{local} not built yet. Update first."}
    write_env_value(CLUSTER_ENV, "VLLM_IMAGE", local)
    log_line(f"[image] switching cluster to {local} (relaunch required)")
    return {"ok": True, "detail": f"Set to {local}. Stop then Start to relaunch on the new image.",
            "needs_relaunch": True}


def image_overview() -> dict:
    cur = current_vllm_version()
    locals_ = list_local_images()
    cur_tag = ""
    m = re.search(r":(\d+\.\d+)-py3", cur.get("image", ""))
    if m:
        cur_tag = m.group(1)
    candidates = []
    if cur_tag:
        try:
            yy, mm = cur_tag.split(".")
            for delta in (1, 2):
                nm = int(mm) + delta
                ny, nmo = (int(yy), nm) if nm <= 12 else (int(yy) + 1, nm - 12)
                candidates.append(f"{ny:02d}.{nmo:02d}-py3")
        except ValueError:
            pass
    available = [check_ngc_tag(t) for t in candidates]
    return {"current": cur, "local_images": locals_, "current_ngc_tag": cur_tag, "candidates": available}


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


@app.get("/api/models/curated")
def api_models_curated():
    return JSONResponse({"models": curated_models(), "budget": {
        "pooled_gb": POOLED_GB, "usable_gb": USABLE_GB, "single_node_gb": SINGLE_NODE_GB}})


@app.get("/api/models/cached")
def api_models_cached():
    return JSONResponse({"models": hf_cache_models()})


@app.get("/api/models/search")
def api_models_search(q: str = ""):
    return JSONResponse({"results": hf_search(q)})


@app.post("/api/models/test")
def api_models_test(payload: dict = Body(...)):
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        return JSONResponse({"ok": False, "detail": "No prompt provided"}, status_code=400)
    return JSONResponse(test_prompt(prompt, int(payload.get("max_tokens", 128))))


@app.get("/api/download/status")
def api_download_status():
    return JSONResponse(download_status())


@app.post("/api/download/dismiss")
def api_download_dismiss():
    return JSONResponse(dismiss_status())


@app.post("/api/model/preload-check")
def api_preload_check(payload: dict = Body(...)):
    model = (payload.get("model") or "").strip()
    if not model:
        return JSONResponse({"ok": False, "detail": "No model provided"}, status_code=400)
    return JSONResponse(preload_check(model))


@app.post("/api/model/unload")
def api_model_unload():
    return JSONResponse(unload_model())


@app.post("/api/serving-mode")
def api_serving_mode(payload: dict = Body(...)):
    mode = (payload.get("mode") or "").strip()
    if mode not in ("single", "cluster"):
        return JSONResponse({"ok": False, "detail": "mode must be 'single' or 'cluster'"}, status_code=400)
    return JSONResponse(set_serving_mode(mode))


@app.post("/api/download")
def api_download(payload: dict = Body(...)):
    model = (payload.get("model") or "").strip()
    if not model:
        return JSONResponse({"ok": False, "detail": "No model provided"}, status_code=400)
    return JSONResponse(start_download(model, auto_load=payload.get("auto_load", True)))


@app.get("/api/history")
def api_history(window: str = "hour"):
    secs = {"hour": 3600, "day": 86400, "week": 604800}.get(window, 3600)
    return JSONResponse({"window": window, "samples": db_history(secs)})


@app.get("/api/cost")
def api_cost():
    return JSONResponse(db_cost_summary())


@app.get("/api/image/overview")
def api_image_overview():
    return JSONResponse(image_overview())


@app.get("/api/image/status")
def api_image_status():
    return JSONResponse(image_status())


@app.post("/api/image/update")
def api_image_update(payload: dict = Body(...)):
    tag = (payload.get("tag") or "").strip()
    if not tag:
        return JSONResponse({"ok": False, "detail": "No tag provided"}, status_code=400)
    return JSONResponse(start_image_update(tag))


@app.post("/api/image/switch")
def api_image_switch(payload: dict = Body(...)):
    tag = (payload.get("tag") or "").strip()
    if not tag:
        return JSONResponse({"ok": False, "detail": "No tag provided"}, status_code=400)
    return JSONResponse(switch_image(tag))


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    log_line(f"Panel starting. Kit dir: {KIT_DIR}")
    db_init()
    threading.Thread(target=_collector_loop, daemon=True).start()
    log_line(f"Metrics collector started (db: {DB_PATH.name}, cost rate ${COST_PER_MTOK}/Mtok)")
    uvicorn.run(app, host=PANEL_HOST, port=PANEL_PORT, log_level="warning")
