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
import sqlite3
import threading
import subprocess
import collections
from pathlib import Path

from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
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
# Single-node vLLM + DFlash speculative decoding (vllm-dflash-serve.sh) runs in its
# own container, distinct from the multi-node Ray cluster (CONTAINER above). When a
# base model in DFLASH_PAIRS is loaded and its draft is cached, the panel serves it
# this way (vLLM on the same API port, so it's detected as the vLLM engine).
DFLASH_CONTAINER = "vllm-dflash"
DFLASH_PAIRS = {"Qwen/Qwen3.6-27B": "z-lab/Qwen3.6-27B-DFlash"}

PANEL_HOST = os.environ.get("GX10_PANEL_HOST", "0.0.0.0")
PANEL_PORT = int(os.environ.get("GX10_PANEL_PORT", "8090"))  # 8080 is commonly Open WebUI

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


_env_lock = threading.Lock()


def write_env_value(path: Path, key: str, value: str) -> None:
    """Update a single KEY="value" line in place, preserving the rest of the file.
    Serialized + atomic (temp file then os.replace) so concurrent writers from
    different threads can't lose updates or leave a half-written file."""
    with _env_lock:
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
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text("\n".join(lines) + "\n")
        os.replace(tmp, path)


def _as_int(value, default: int) -> int:
    """Best-effort int coercion for request payloads; falls back to default on junk
    so a bad 'ctx'/'max_tokens' can't 500 an endpoint."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


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


# ------------------------------------------------------------
# Metrics store: SQLite history for throughput, GPU, memory, thermals, cost
# ------------------------------------------------------------
DB_PATH = Path(os.environ.get("GX10_DB", str(BASE_DIR / "gx10-metrics.db")))
_db_lock = threading.Lock()

# Cost comparison rate: $ per 1M generated tokens (configurable). Default ~ frontier model.
COST_PER_MTOK = float(os.environ.get("GX10_COST_PER_MTOK", "10.0"))


# Columns for the metrics 'samples' table. New columns are appended over time and
# migrated onto existing DBs via ALTER TABLE in db_init().
SAMPLE_COLS = [
    ("ts", "INTEGER PRIMARY KEY"), ("tps", "REAL"), ("gen_total", "REAL"),
    ("gen_delta", "REAL"),
    ("head_gpu", "INTEGER"), ("head_mem_used", "INTEGER"), ("head_mem_total", "INTEGER"),
    ("head_temp", "INTEGER"), ("head_power", "REAL"),
    ("worker_gpu", "INTEGER"), ("worker_mem_used", "INTEGER"), ("worker_mem_total", "INTEGER"),
    ("worker_temp", "INTEGER"), ("worker_power", "REAL"),
    ("ttft_ms", "REAL"), ("itl_ms", "REAL"), ("e2e_ms", "REAL"), ("queue_ms", "REAL"),
    ("kv_cache", "REAL"), ("req_success", "REAL"), ("req_error", "REAL"),
    ("model", "TEXT"),
]


def db_init():
    with _db_lock, sqlite3.connect(DB_PATH) as con:
        cols_sql = ", ".join(f"{n} {t}" for n, t in SAMPLE_COLS)
        con.execute(f"CREATE TABLE IF NOT EXISTS samples ({cols_sql})")
        # migrate older DBs: add any columns introduced after the table was created
        existing = {r[1] for r in con.execute("PRAGMA table_info(samples)").fetchall()}
        for n, t in SAMPLE_COLS:
            if n not in existing:
                con.execute(f"ALTER TABLE samples ADD COLUMN {n} {t.replace(' PRIMARY KEY', '')}")
        con.execute("""
            CREATE TABLE IF NOT EXISTS benchmarks (
                ts INTEGER PRIMARY KEY,
                model TEXT, concurrency INTEGER, prompt_tokens INTEGER,
                max_tokens INTEGER, ttft_ms REAL, tps REAL,
                total_tokens INTEGER, duration_s REAL, notes TEXT
            )""")
        con.commit()


def db_insert_sample(row: dict):
    cols = tuple(n for n, _ in SAMPLE_COLS)
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
    """Tokens generated and cost-equivalent over windows."""
    now = int(time.time())
    out = {"rate_per_mtok": COST_PER_MTOK}
    with _db_lock, sqlite3.connect(DB_PATH) as con:
        for label, secs in (("today", 86400), ("week", 604800), ("all", 10**12)):
            cutoff = now - secs
            # sum reset-aware per-sample deltas; correct across vLLM restarts (the old
            # MAX-MIN of the cumulative counter undercounted / broke on every reload)
            r = con.execute(
                "SELECT COALESCE(SUM(gen_delta), 0) FROM samples WHERE ts >= ? AND gen_delta IS NOT NULL",
                (cutoff,)).fetchone()
            tokens = int(r[0] or 0) if r else 0
            out[label] = {
                "tokens": tokens,
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
    gen = s.get("gen_tokens_total")
    nowt = time.time()
    prev = getattr(_collect_once, "_prev", None)
    tps = None
    gen_delta = None
    if gen is not None:
        if prev is not None:
            pg, pt = prev
            dt = nowt - pt
            # a vLLM/llama restart resets the cumulative counter to ~0; treat a drop
            # as a reset and count the new run's tokens so far (never negative)
            gen_delta = (gen - pg) if gen >= pg else gen
            if dt > 0:
                tps = max(0.0, gen_delta / dt)
        _collect_once._prev = (gen, nowt)

    def _avg(d):
        return (d or {}).get("avg_ms")
    db_insert_sample({
        "ts": int(nowt), "tps": tps, "gen_total": gen, "gen_delta": gen_delta,
        "head_gpu": h.get("gpu_util"), "head_mem_used": h.get("mem_used"),
        "head_mem_total": h.get("mem_total"), "head_temp": h.get("temp"), "head_power": h.get("power"),
        "worker_gpu": w.get("gpu_util"), "worker_mem_used": w.get("mem_used"),
        "worker_mem_total": w.get("mem_total"), "worker_temp": w.get("temp"), "worker_power": w.get("power"),
        "ttft_ms": _avg(s.get("ttft")), "itl_ms": _avg(s.get("itl")),
        "e2e_ms": _avg(s.get("e2e")), "queue_ms": _avg(s.get("queue")),
        "kv_cache": s.get("kv_cache_pct"), "req_success": s.get("requests_success_total"),
        "req_error": None,
        "model": cfg().get("MODEL", ""),
    })


def _collector_loop(interval=10):
    db_init()
    while not _collector_stop.wait(interval):
        _collect_once()
        try:
            _maybe_restore_serving()
        except Exception as e:
            log_line(f"[watchdog] error: {e}")


# ------------------------------------------------------------
# Serving state + watchdog. Remember what should be serving so we can bring it
# back after a crash/reboot if the container restart policy didn't (or the
# container was removed). Conservative: cooldown + bounded attempts, and it never
# fights an in-progress job or an explicit stop (which clears the state). All the
# functions it calls are defined later in the file but only invoked at run time.
# ------------------------------------------------------------
SERVING_STATE_FILE = Path(os.environ.get("GX10_STATE_DIR", str(Path.home()))) / ".gx10-serving-state.json"
_restore = {"attempts": 0, "last": 0.0}


def save_serving_state(engine, model, draft=None, repo=None):
    """Record the intended serving config (called when a load reaches 'serving')."""
    try:
        SERVING_STATE_FILE.write_text(json.dumps(
            {"engine": engine, "model": model, "draft": draft, "repo": repo, "ts": int(time.time())}))
        _restore["attempts"] = 0
    except Exception:
        pass


def clear_serving_state():
    """Forget the intended config (called on explicit stop/unload) so the watchdog
    won't resurrect a deliberately stopped engine."""
    _restore["attempts"] = 0
    try:
        SERVING_STATE_FILE.unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass


def _maybe_restore_serving():
    try:
        st = json.loads(SERVING_STATE_FILE.read_text())
    except Exception:
        return
    if not st:
        return
    if active_engine().get("model"):     # something is serving — all good
        _restore["attempts"] = 0
        return
    with _download_lock:                  # never fight an in-progress load
        if _download.get("active"):
            return
    now = time.time()
    if now - _restore["last"] < 90 or _restore["attempts"] >= 5:
        return
    _restore["last"] = now
    _restore["attempts"] += 1
    eng, model = st.get("engine"), st.get("model")
    log_line(f"[watchdog] nothing serving; restoring {eng}/{model} (attempt {_restore['attempts']}/5)")
    try:
        if eng == "dflash":
            start_dflash(model, st.get("draft") or DFLASH_PAIRS.get(model, ""))
        elif eng == "llamacpp":
            start_llama(st.get("repo") or "", model)
        elif eng == "vllm":
            route_model(model)
    except Exception as e:
        log_line(f"[watchdog] restore failed: {e}")


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


# Short-lived cache so rapid polling doesn't stack blocking subprocess calls.
# Each expensive gather (status, metrics) is computed at most once per TTL window;
# concurrent/rapid requests get the cached result instead of piling up.
_ttl_cache = {}
_ttl_locks = {}
_ttl_global = threading.Lock()

# Empty-but-correctly-shaped values returned on a cold cache while another thread
# is computing, so endpoints never emit a bare {} that the frontend would
# dereference into errors (e.g. data.checks.forEach, data.nodes.head.gpu_util).
_CACHE_SKELETONS = {
    "status": {"overall": "idle", "running": False, "action": None, "checks": [],
               "nodes": {"head": {}, "worker": {}}, "config": {}},
    "metrics": {"serving": {}, "engine": None, "nodes": {"head": {}, "worker": {}}},
    "telemetry": {"engine": None, "model": None, "serving": {}, "vllm": {}, "llamacpp": {}},
    "library": {"models": [], "storage": {}},
    "nas": [],
    "engine_active": {"engine": None, "model": None, "port": None},
    "image_overview": {"current": {}, "local_images": [], "current_ngc_tag": "", "candidates": []},
}


def _skeleton(key):
    import copy
    return copy.deepcopy(_CACHE_SKELETONS.get(key, {}))


def cached(key: str, ttl: float, producer):
    """Return a cached value if fresh, else compute it. Non-blocking for readers:
    if another thread is already computing, return the last known value (or a
    correctly-shaped empty skeleton, never a bare {})."""
    now = time.time()
    with _ttl_global:
        entry = _ttl_cache.get(key)
        if entry and (now - entry[1]) < ttl:
            return entry[0]
        lock = _ttl_locks.setdefault(key, threading.Lock())
    # only one thread computes; others return stale immediately
    if not lock.acquire(blocking=False):
        with _ttl_global:
            entry = _ttl_cache.get(key)
        return entry[0] if entry else _skeleton(key)
    try:
        value = producer()
        with _ttl_global:
            _ttl_cache[key] = (value, time.time())
        return value
    finally:
        lock.release()


def invalidate_cache(*keys):
    """Drop cached entries so the next read recomputes fresh. Call after any
    state change (load, unload, cancel, engine swap) so the UI doesn't see stale
    'active' status. With no args, clears everything."""
    with _ttl_global:
        if keys:
            for k in keys:
                _ttl_cache.pop(k, None)
        else:
            _ttl_cache.clear()


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


def detect_fabric_links() -> dict:
    """Auto-detect ConnectX-class fabric NICs (>=100G) and their link state, by
    reading /sys/class/net. Works even when CX7_IFACE is unset in cluster.env."""
    ports = []
    for p in sorted(glob.glob("/sys/class/net/*/speed")):
        d = Path(p).parent
        try:
            speed = int(Path(p).read_text().strip())
        except Exception:
            continue
        if speed < 100000:  # fabric-class only; skips eth/docker/veth
            continue
        try:
            oper = (d / "operstate").read_text().strip()
        except Exception:
            oper = "?"
        ports.append({"iface": d.name, "speed_mbps": speed, "up": oper == "up"})
    up = [x for x in ports if x["up"]]
    per = (up[0]["speed_mbps"] if up else (ports[0]["speed_mbps"] if ports else None))
    return {"ports": ports, "ports_up": len(up), "ports_total": len(ports),
            "per_port_gbps": round(per / 1000) if per else None}


def check_link(c) -> dict:
    iface = c.get("CX7_IFACE") or ""
    if not iface:  # cluster.env may leave it blank; fall back to auto-detection
        up = [x for x in detect_fabric_links()["ports"] if x["up"]]
        iface = up[0]["iface"] if up else "enp1s0f1np1"
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


def _parse_prom_hist(text: str, name: str):
    """Parse a Prometheus histogram family (name_bucket/_sum/_count), summed across
    label sets. Returns {'sum','count','buckets':[(le, cumulative_count)...]} or None."""
    buckets = {}
    total_sum = 0.0
    total_count = 0.0
    found = False
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        metric = line.split("{")[0].split(" ")[0]
        try:
            val = float(line.rsplit(" ", 1)[1])
        except (ValueError, IndexError):
            continue
        if metric == name + "_bucket":
            found = True
            m = re.search(r'le="([^"]+)"', line)
            if m:
                buckets[m.group(1)] = buckets.get(m.group(1), 0.0) + val
        elif metric == name + "_sum":
            found = True
            total_sum += val
        elif metric == name + "_count":
            found = True
            total_count += val
    if not found:
        return None

    def _le(le):
        return float("inf") if le in ("+Inf", "Inf") else float(le)
    bl = sorted(buckets.items(), key=lambda kv: _le(kv[0]))
    return {"sum": total_sum, "count": total_count,
            "buckets": [(_le(le), c) for le, c in bl]}


def _hist_avg(h):
    if not h or not h.get("count"):
        return None
    return h["sum"] / h["count"]


def _hist_quantile(h, q):
    """Approximate a quantile from cumulative histogram buckets via linear interp."""
    if not h:
        return None
    buckets = h.get("buckets") or []
    total = h.get("count") or 0
    if not buckets or total <= 0:
        return None
    target = q * total
    prev_le, prev_c = 0.0, 0.0
    for le, c in buckets:
        if c >= target:
            if le == float("inf"):
                return prev_le if prev_le > 0 else None
            if c == prev_c:
                return le
            return prev_le + (target - prev_c) / (c - prev_c) * (le - prev_le)
        prev_le, prev_c = le, c
    last = buckets[-1][0]
    return prev_le if last == float("inf") else last


def _latency_block(h):
    """Turn a seconds-histogram into a ms summary block for the UI."""
    if not h:
        return {"avg_ms": None, "p50_ms": None, "p95_ms": None, "p99_ms": None}
    ms = lambda v: round(v * 1000, 1) if v is not None else None
    return {"avg_ms": ms(_hist_avg(h)),
            "p50_ms": ms(_hist_quantile(h, 0.50)),
            "p95_ms": ms(_hist_quantile(h, 0.95)),
            "p99_ms": ms(_hist_quantile(h, 0.99))}


def _vllm_serving(body: str) -> dict:
    """Rich serving metrics from a vLLM /metrics body."""
    kv = _parse_prom(body, "vllm:gpu_cache_usage_perc")
    return {
        "tps_gauge": None,  # vLLM has no stable instant-throughput gauge; UI uses counter deltas
        "gen_tokens_total": _parse_prom(body, "vllm:generation_tokens_total"),
        "prompt_tokens_total": _parse_prom(body, "vllm:prompt_tokens_total"),
        "requests_running": _parse_prom(body, "vllm:num_requests_running"),
        "requests_waiting": _parse_prom(body, "vllm:num_requests_waiting"),
        "requests_success_total": _parse_prom(body, "vllm:request_success_total"),
        "kv_cache_pct": round(kv * 100, 1) if kv is not None else None,
        "ttft": _latency_block(_parse_prom_hist(body, "vllm:time_to_first_token_seconds")),
        "itl": _latency_block(_parse_prom_hist(body, "vllm:time_per_output_token_seconds")),
        "e2e": _latency_block(_parse_prom_hist(body, "vllm:e2e_request_latency_seconds")),
        "queue": _latency_block(_parse_prom_hist(body, "vllm:request_queue_time_seconds")),
    }


def _llama_serving(body: str) -> dict:
    """Rich serving metrics from a llama.cpp /metrics body. llama.cpp exposes fewer
    fields than vLLM (no latency histograms), so TTFT/ITL are averages where derivable."""
    gen = _parse_prom(body, "llamacpp:tokens_predicted_total")
    pred_secs = _parse_prom(body, "llamacpp:tokens_predicted_seconds_total")
    prompt_secs = _parse_prom(body, "llamacpp:prompt_seconds_total")
    n_decode = _parse_prom(body, "llamacpp:n_decode_total")
    kv = _parse_prom(body, "llamacpp:kv_cache_usage_ratio")
    none4 = {"avg_ms": None, "p50_ms": None, "p95_ms": None, "p99_ms": None}
    itl_avg = round(pred_secs / gen * 1000, 1) if (gen and pred_secs) else None
    ttft_avg = round(prompt_secs / n_decode * 1000, 1) if (n_decode and prompt_secs) else None
    # llama.cpp publishes an instantaneous generation-rate gauge; use it directly so
    # throughput reflects real output rate instead of noisy counter deltas.
    tps_gauge = _parse_prom(body, "llamacpp:predicted_tokens_seconds")
    return {
        "tps_gauge": round(tps_gauge, 1) if tps_gauge is not None else None,
        "gen_tokens_total": gen,
        "prompt_tokens_total": _parse_prom(body, "llamacpp:prompt_tokens_total"),
        "requests_running": _parse_prom(body, "llamacpp:requests_processing"),
        "requests_waiting": _parse_prom(body, "llamacpp:requests_deferred"),
        "requests_success_total": None,
        "kv_cache_pct": round(kv * 100, 1) if kv is not None else None,
        "ttft": {**none4, "avg_ms": ttft_avg},
        "itl": {**none4, "avg_ms": itl_avg},
        "e2e": dict(none4),
        "queue": dict(none4),
    }


def gather_metrics() -> dict:
    """Live serving throughput. Reads vLLM /metrics when vLLM serves, or
    llama.cpp /metrics when llama.cpp serves, so the stats reflect whichever
    engine is actually running. Plus per-node memory/util."""
    c = cfg()
    out = {"serving": {}, "engine": None, "nodes": {}}

    # vLLM exposes Prometheus metrics on the same port as the API
    rc, body, _ = run(["curl", "-s", "--max-time", "4",
                       f"http://localhost:{c['API_PORT']}/metrics"], timeout=6)
    if rc == 0 and body and "vllm:" in body:
        out["engine"] = "vllm"
        out["serving"] = _vllm_serving(body)
    else:
        # try llama.cpp metrics (port 8001). llama-server exposes Prometheus
        # metrics at /metrics when run with --metrics; field names differ.
        rcl, lbody, _ = run(["curl", "-s", "--max-time", "4",
                             f"http://localhost:{LLAMA_PORT}/metrics"], timeout=6)
        if rcl == 0 and lbody and "llamacpp:" in lbody:
            out["engine"] = "llamacpp"
            out["serving"] = _llama_serving(lbody)
        elif llama_serving():
            # llama.cpp is up but metrics not exposed (no --metrics flag);
            # report it as serving so the UI shows the engine, stats as 0
            out["engine"] = "llamacpp"
            out["serving"] = {"gen_tokens_total": None, "prompt_tokens_total": None,
                              "requests_running": 0, "requests_waiting": 0,
                              "metrics_unavailable": True}

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
    # the worker readout is an SSH round-trip (~0.7s) and changes slowly, so cache it
    # longer than the local stats — keeps live telemetry snappy without stale worker data.
    out["nodes"]["worker"] = cached("worker_stats", 6.0, lambda: node_stats(local=False))
    # physical fabric link speed (read locally from /sys; cheap)
    out["fabric"] = detect_fabric_links()
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


def gather_telemetry() -> dict:
    """Live telemetry snapshot for the Telemetry tab: latency blocks (TTFT/ITL/e2e/
    queue, avg + p50/p95/p99), KV-cache %, queue depth, success counts, and recent
    tokens/sec. Reuses the cached metrics fetch + the latest collector sample."""
    m = cached("metrics", 1.5, gather_metrics)
    serving = m.get("serving", {}) or {}
    tps = None
    try:
        with _db_lock, sqlite3.connect(DB_PATH) as con:
            r = con.execute(
                "SELECT tps FROM samples WHERE tps IS NOT NULL ORDER BY ts DESC LIMIT 1").fetchone()
            if r and r[0] is not None:
                tps = round(r[0], 1)
    except Exception:
        pass
    return {"engine": m.get("engine"), "tps": tps, "serving": serving}


# ------------------------------------------------------------
# Benchmark runner: fires concurrent streaming requests, measures TTFT + tok/s,
# records into the (previously unused) benchmarks table.
# ------------------------------------------------------------
_benchmark = {"active": False, "status": "idle", "detail": "", "result": None, "error": None}
_benchmark_lock = threading.Lock()


def _bench_one(prompt: str, max_tokens: int, port: str, model: str):
    """One streaming chat request. Returns (ttft_s, gen_chunks, total_s) or None."""
    import urllib.request
    payload = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": max_tokens, "stream": True}).encode()
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
                                 data=payload, headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    chunks = 0
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                if ttft is None:
                    ttft = time.time() - t0
                try:
                    delta = json.loads(data)["choices"][0].get("delta", {}).get("content")
                    if delta:
                        chunks += 1
                except Exception:
                    pass
    except Exception:
        return None
    return (ttft, chunks, time.time() - t0)


def _benchmark_worker(concurrency: int, prompt: str, max_tokens: int):
    eng = active_engine()
    c = cfg()
    port = eng.get("port") or c.get("API_PORT", "8000")
    model = eng.get("model") or ""
    if not model:
        with _benchmark_lock:
            _benchmark.update(active=False, status="error",
                              error="No model is serving. Load one first.", result=None)
        return
    with _benchmark_lock:
        _benchmark.update(active=True, status="running", error=None, result=None,
                          detail=f"Running {concurrency} concurrent requests against {model}...")
    log_line(f"[bench] start: {concurrency} concurrent vs {model}")
    results = []
    rlock = threading.Lock()

    def task():
        r = _bench_one(prompt, max_tokens, port, model)
        if r:
            with rlock:
                results.append(r)
    threads = [threading.Thread(target=task) for _ in range(concurrency)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    duration = time.time() - t0
    if not results:
        with _benchmark_lock:
            _benchmark.update(active=False, status="error",
                              error="All benchmark requests failed.", result=None)
        return
    ttfts = [r[0] for r in results if r[0] is not None]
    total_toks = sum(r[1] for r in results)
    avg_ttft = round(sum(ttfts) / len(ttfts) * 1000, 1) if ttfts else None
    agg_tps = round(total_toks / duration, 1) if duration > 0 else None
    summary = {"model": model, "concurrency": concurrency, "requests": len(results),
               "ttft_ms": avg_ttft, "tps": agg_tps, "total_tokens": total_toks,
               "duration_s": round(duration, 2), "max_tokens": max_tokens}
    try:
        with _db_lock, sqlite3.connect(DB_PATH) as con:
            con.execute(
                "INSERT OR REPLACE INTO benchmarks (ts,model,concurrency,prompt_tokens,"
                "max_tokens,ttft_ms,tps,total_tokens,duration_s,notes) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (int(time.time()), model, concurrency, len(prompt.split()), max_tokens,
                 avg_ttft, agg_tps, total_toks, round(duration, 2), ""))
            con.commit()
    except Exception:
        pass
    with _benchmark_lock:
        _benchmark.update(active=False, status="done", result=summary,
                          detail=f"{agg_tps} tok/s aggregate, TTFT {avg_ttft}ms")
    log_line(f"[bench] done: {agg_tps} tok/s, TTFT {avg_ttft}ms")


def start_benchmark(concurrency: int, prompt: str, max_tokens: int) -> dict:
    with _benchmark_lock:
        if _benchmark["active"]:
            return {"ok": False, "detail": "A benchmark is already running."}
        _benchmark.update(active=True, status="starting", error=None, result=None)
    threading.Thread(target=_benchmark_worker,
                     args=(concurrency, prompt, max_tokens), daemon=True).start()
    return {"ok": True, "detail": "Benchmark started."}


def benchmark_history(limit: int = 20) -> list:
    try:
        with _db_lock, sqlite3.connect(DB_PATH) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute("SELECT * FROM benchmarks ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


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
    """Runs the launch script. The caller has already acquired _action_lock and set
    _current_action; this worker releases the lock in its finally."""
    name = "stopping" if arg == "stop" else "starting"
    try:
        if not LAUNCH_SCRIPT.exists():
            log_line(f"[error] launch script not found at {LAUNCH_SCRIPT}. "
                     f"Set GX10_KIT_DIR.")
            return
        cmd = ["bash", str(LAUNCH_SCRIPT)] + ([arg] if arg else [])
        log_line(f"[{name}] {' '.join(cmd)}")
        # Start the cluster idle: Ray + both nodes up, but no model auto-loaded.
        # The user picks a model from the panel. (Stop is unaffected.)
        env = dict(os.environ)
        if arg != "stop":
            env["NO_AUTOLOAD"] = "1"
        proc = subprocess.Popen(cmd, cwd=str(KIT_DIR), text=True, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in proc.stdout:
            if line.strip():
                log_line(line.rstrip())
        proc.wait()
        log_line(f"[{name}] finished (exit {proc.returncode}).")
    finally:
        _current_action["name"] = None
        _action_lock.release()


def start_cluster():
    # Acquire the action lock in the request handler itself (non-blocking) so two
    # near-simultaneous Start/Stop clicks cannot both pass the guard. The worker
    # thread releases it when the launch finishes.
    if not _action_lock.acquire(blocking=False):
        return {"ok": False, "detail": f"Busy: {_current_action['name'] or 'a cluster action'}"}
    _current_action["name"] = "starting"
    threading.Thread(target=_run_launch, daemon=True).start()
    return {"ok": True, "detail": "Start initiated. Watch the activity log."}


def stop_cluster():
    if not _action_lock.acquire(blocking=False):
        return {"ok": False, "detail": f"Busy: {_current_action['name'] or 'a cluster action'}"}
    _current_action["name"] = "stopping"
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
# Monotonic job id. Bumped on every successful claim and on every cancel, so a
# worker thread that has been superseded or cancelled can detect it and stop
# writing state. Workers carry the id they were started with.
_job_id = 0
# Handle to the active download subprocess (the `docker exec ... hf download`),
# so cancel can terminate it directly instead of relying on pkill alone.
_download_proc = None


def _claim_job(model=None, status="starting", detail="") -> tuple:
    """Atomically claim the single job slot. Returns (job_id, None) on success or
    (None, busy_detail) if a job is already active. Setting active=True happens in
    the SAME critical section as the guard check, so two near-simultaneous requests
    cannot both pass (closes the check-then-act race)."""
    global _job_id
    with _download_lock:
        if _download["active"]:
            return None, f"Busy: {_download.get('model')} ({_download.get('status')})"
        _job_id += 1
        jid = _job_id
        _download.update(active=True, model=model, status=status, percent=0,
                         detail=detail or (f"Starting {model}..." if model else "Starting..."),
                         speed="", error=None, started=time.time())
    return jid, None


def _job_is_current(jid) -> bool:
    """True if this worker still owns the job slot (not cancelled/superseded)."""
    return jid == _job_id


def _release_job(jid) -> None:
    """Clear active only if we still own the slot, so we never clobber a newer job."""
    with _download_lock:
        if jid == _job_id and _download["active"]:
            _download["active"] = False


def _set_dl(_jid=None, **kw):
    """Update download state. If _jid is given and no longer the current job, the
    write is dropped — a cancelled or superseded worker can't resurrect stale state."""
    with _download_lock:
        if _jid is not None and _jid != _job_id:
            return
        _download.update(kw)


def download_status() -> dict:
    with _download_lock:
        return dict(_download)


def _start_vllm(model: str, max_len_override: int = None) -> tuple:
    """Relaunch vLLM serve in the running container with the given model."""
    c = cfg()
    max_len = max_len_override if max_len_override else c["MAX_MODEL_LEN"]
    run(["docker", "exec", CONTAINER, "pkill", "-f", "vllm serve"], timeout=10)
    time.sleep(2)
    serve = (
        f"vllm serve '{model}' --host 0.0.0.0 --port {c['API_PORT']} "
        f"--tensor-parallel-size {c['TENSOR_PARALLEL']} --distributed-executor-backend ray "
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


def _watch_vllm_until_ready(model: str, timeout_s: int = 1200, _retried=False, jid=None) -> bool:
    """Poll until the model serves (return True) or fails (return False)."""
    c = cfg()
    port = c["API_PORT"]
    start = time.time()
    phase = "loading"
    while time.time() - start < timeout_s:
        # cancelled or superseded by a newer job? stop watching immediately
        if jid is not None and not _job_is_current(jid):
            log_line(f"[model] watch for {model} stopped (cancelled/superseded)")
            return False
        rc, out, _ = run(["curl", "-s", "--max-time", "3", f"http://localhost:{port}/v1/models"], timeout=5)
        if rc == 0 and out:
            try:
                served = json.loads(out)["data"][0]["id"]
                if served == model:
                    _set_dl(_jid=jid, active=False, status="ready", percent=100,
                            detail=f"{model} is live and serving.", error=None)
                    log_line(f"[model] {model} is ready and serving")
                    return True
            except Exception:
                pass
        rc2, log, _ = run(["docker", "exec", CONTAINER, "tail", "-n", "40", "/var/log/vllm.log"], timeout=8)
        if rc2 == 0 and log:
            low = log.lower()
            # auto-correct context length once
            mlen = re.search(r"derived max_model_len \(max_position_embeddings=(\d+)", log)
            if mlen and not _retried:
                real_max = int(mlen.group(1))
                log_line(f"[model] {model} caps context at {real_max}; retrying")
                _set_dl(_jid=jid, active=True, status="loading",
                        detail=f"{model} max context is {real_max}, retrying...")
                _start_vllm(model, max_len_override=real_max)
                _model_maxlen_cache[model] = real_max
                return _watch_vllm_until_ready(model, timeout_s=timeout_s, _retried=True, jid=jid)
            # fatal error?
            if ("engine core initialization failed" in low or "traceback (most recent call last)" in low
                    or "raise runtimeerror" in low):
                reason = _explain_error(log)
                _set_dl(_jid=jid, active=False, status="error", error=reason, detail=reason)
                log_line(f"[model] load FAILED: {reason}")
                return False
            if "capturing" in low or "graph" in low or "compile" in low:
                phase = "compiling"
            elif "loading safetensors" in low or "loading weights" in low:
                phase = "loading weights"
        elapsed = int(time.time() - start)
        _set_dl(_jid=jid, active=True, status=phase, detail=f"{model}: {phase} ({elapsed}s)")
        time.sleep(5)
    _set_dl(_jid=jid, active=False, status="error", error="Timed out waiting for the model to load.",
            detail="Load timed out. Check the vLLM logs.")
    return False


# remembers per-model context caps discovered via auto-correction
_model_maxlen_cache = {}


def _download_worker(model: str, auto_load: bool, jid=None):
    """Download weights into the cache with progress, then optionally load + watch."""
    global _download_proc
    c = cfg()
    token = c.get("HF_TOKEN", "")
    _set_dl(_jid=jid, active=True, model=model, status="downloading", percent=0,
            detail=f"Preparing to download {model}...", speed="", error=None, started=time.time())
    log_line(f"[download] starting {model}")

    env_prefix = f"HF_TOKEN={token} " if token else ""
    cmd = f"{env_prefix}hf download '{model}' 2>&1"

    proc = subprocess.Popen(
        ["docker", "exec", CONTAINER, "bash", "-lc", cmd],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    with _download_lock:
        _download_proc = proc

    last_pct = 0
    pat_pct = re.compile(r'(\d+)%')
    pat_speed = re.compile(r'([\d.]+[KMG]B/s)')
    saw_error = None
    for line in proc.stdout:
        if jid is not None and not _job_is_current(jid):
            break  # cancelled/superseded: stop reading; cancel_load kills the proc
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
                _set_dl(_jid=jid, percent=pct, status="downloading",
                        detail=f"Downloading {model}", speed=sp.group(1) if sp else "")
    proc.wait()
    with _download_lock:
        if _download_proc is proc:
            _download_proc = None

    # cancelled/superseded while downloading: leave the final state to the canceller
    if jid is not None and not _job_is_current(jid):
        log_line(f"[download] {model} stopped (cancelled/superseded)")
        return

    if saw_error or proc.returncode != 0:
        msg = saw_error or f"Download failed (exit {proc.returncode}). See activity log."
        _set_dl(_jid=jid, active=False, status="error", error=msg, detail=msg)
        log_line(f"[download] FAILED {model}: {msg}")
        return

    _set_dl(_jid=jid, percent=100, status="downloaded", detail=f"{model} downloaded.")
    log_line(f"[download] complete {model}")

    if auto_load:
        write_env_value(CLUSTER_ENV, "MODEL", model)
        _set_dl(_jid=jid, active=True, status="loading", detail=f"Starting {model} on the cluster...")
        log_line(f"[model] auto-loading {model}")
        rc, _, err = _start_vllm(model, _model_maxlen_cache.get(model))
        if rc != 0:
            _set_dl(_jid=jid, active=False, status="error", error=f"Load failed: {err}")
            return
        _watch_vllm_until_ready(model, jid=jid)
    else:
        _set_dl(_jid=jid, active=False, status="downloaded")


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


def _load_worker(model: str, jid=None):
    """Load a cached model with rollback: if it fails, restore the previous one."""
    previous = currently_serving()
    write_env_value(CLUSTER_ENV, "MODEL", model)
    _set_dl(_jid=jid, active=True, model=model, status="loading", percent=0,
            detail=f"Starting {model}...", speed="", error=None, started=time.time())
    log_line(f"[model] loading {model}" + (f" (was {previous})" if previous else ""))
    rc, _, err = _start_vllm(model, _model_maxlen_cache.get(model))
    if rc != 0:
        if not previous:
            write_env_value(CLUSTER_ENV, "MODEL", "")
        _set_dl(_jid=jid, active=False, status="error", error=f"Could not start: {err}")
        return
    ok = _watch_vllm_until_ready(model, jid=jid)
    if ok:
        return
    # cancelled/superseded: the canceller owns the final state; don't roll back
    if jid is not None and not _job_is_current(jid):
        return
    # failed: roll back to the previous working model if there was one
    if previous and previous != model:
        log_line(f"[model] rolling back to {previous}")
        cur = download_status()
        _set_dl(_jid=jid, active=True, model=previous, status="loading",
                detail=f"Load failed. Rolling back to {previous}...",
                error=cur.get("error"))
        write_env_value(CLUSTER_ENV, "MODEL", previous)
        rc2, _, _ = _start_vllm(previous, _model_maxlen_cache.get(previous))
        if rc2 == 0 and _watch_vllm_until_ready(previous, jid=jid):
            _set_dl(_jid=jid, active=False, status="error",
                    error=f"{model} failed to load. Rolled back to {previous}.",
                    detail=f"{model} failed. {previous} restored and serving.")
            log_line(f"[model] rolled back to {previous} after {model} failed")
        else:
            _set_dl(_jid=jid, active=False, status="error",
                    error=f"{model} failed to load, and rolling back to {previous} also failed.",
                    detail="Both the new model and the rollback failed. Check the vLLM logs.")
            log_line(f"[model] rollback to {previous} FAILED after {model} failed")
    else:
        # nothing was serving before: clear the failed model from config so the next
        # cluster start doesn't auto-load a model we already know is broken
        write_env_value(CLUSTER_ENV, "MODEL", "")
        _set_dl(_jid=jid, active=False, status="error",
                error=f"{model} failed to load.",
                detail=f"{model} failed to load. Cluster is idle. See the vLLM logs.")


def start_download(model: str, auto_load: bool = True):
    jid, busy = _claim_job(model, status="downloading", detail=f"Preparing to download {model}...")
    if jid is None:
        return {"ok": False, "detail": busy}
    threading.Thread(target=_download_worker, args=(model, auto_load, jid), daemon=True).start()
    return {"ok": True, "detail": f"Download started for {model}."}


def swap_model(model: str):
    """Cached -> load + watch (with rollback). Not cached -> download + auto-load."""
    jid, busy = _claim_job(model, status="loading", detail=f"Loading {model}...")
    if jid is None:
        return {"ok": False, "detail": busy}
    cached = {m["id"] for m in hf_cache_models()}
    if model in cached:
        threading.Thread(target=_load_worker, args=(model, jid), daemon=True).start()
        return {"ok": True, "detail": f"Loading cached {model}..."}
    threading.Thread(target=_download_worker, args=(model, True, jid), daemon=True).start()
    return {"ok": True, "detail": f"Downloading then loading {model}..."}


def preload_check(model: str) -> dict:
    """Tell the user what will happen before they load a model."""
    c = cfg()
    cached_ids = {m["id"] for m in hf_cache_models()}
    is_cached = model in cached_ids
    serving = currently_serving()
    # which engine will handle this
    files = hf_repo_files(model)
    engine_decision = detect_engine(model, files)
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
    # engine routing note
    engine = engine_decision["engine"]
    notes.append(engine_decision["reason"])
    # known-risky formats, only a concern on vLLM
    low = model.lower()
    if engine == "vllm" and ("nvfp4" in low or "fp4" in low):
        warnings.append("NVFP4 format isn't supported by vLLM yet. Will likely fail. If a GGUF version exists, it runs on llama.cpp instead.")
    if engine == "llamacpp" and not is_cached:
        notes.append("GGUF will download on first load.")
    return {
        "model": model, "cached": is_cached, "currently_serving": serving,
        "fit": fit["fit"], "footprint_gb": fit["footprint_gb"],
        "engine": engine, "engine_reason": engine_decision["reason"],
        "warnings": warnings, "notes": notes,
        "ok_to_load": fit["fit"] != "toobig",
    }


def unload_model():
    """Stop vLLM serving (frees memory). Cluster/Ray stays up."""
    with _download_lock:
        if _download["active"]:
            return {"ok": False, "detail": "A job is running; use Cancel to stop it first."}
    rc, _, _ = run(["docker", "exec", CONTAINER, "pkill", "-f", "vllm serve"], timeout=10)
    # also tear down the single-node vLLM+DFlash container if it's the one serving
    run(["docker", "rm", "-f", DFLASH_CONTAINER], timeout=20)
    clear_serving_state()   # explicit stop: don't let the watchdog resurrect it
    _set_dl(active=False, status="idle", model=None, percent=0, detail="", error=None)
    invalidate_cache("library", "engine_active", "status", "metrics")
    log_line("[model] unloaded (vLLM serve stopped)")
    return {"ok": True, "detail": "Model unloaded. Memory freed. Load one to resume serving."}


def cancel_load():
    """Force-stop any in-progress load/compile/download and return to idle.
    Works even when a job is active (unlike unload), so a stuck or failing
    load can always be aborted."""
    global _job_id, _download_proc
    # bump the epoch so any in-flight worker stops writing state, and grab the proc
    with _download_lock:
        _job_id += 1
        proc = _download_proc
        _download_proc = None
        _download.update(active=False, status="cancelled", detail="Cancelling...", error=None)
    # terminate the tracked download subprocess (the docker-exec wrapper) directly
    if proc is not None:
        try:
            proc.terminate()
        except Exception:
            pass
    # kill vLLM serve (stops a compiling/initializing load)
    run(["docker", "exec", CONTAINER, "pkill", "-f", "vllm serve"], timeout=10)
    # kill any in-flight hf download — it runs INSIDE the container, not on the host,
    # so the pkill must run via docker exec (the old host-side pkill matched nothing)
    run(["docker", "exec", CONTAINER, "pkill", "-f", "hf download"], timeout=8)
    run(["docker", "exec", CONTAINER, "pkill", "-f", "huggingface-cli download"], timeout=8)
    time.sleep(1)
    with _download_lock:
        _download.update(active=False, status="idle", model=None, percent=0, detail="", error=None)
    invalidate_cache("library", "engine_active", "status", "metrics")
    log_line("[model] load cancelled by user; cluster idle")
    return {"ok": True, "detail": "Load cancelled. Cluster is idle. Pick a model when ready."}


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


def gguf_cache_models() -> list:
    """List GGUF files downloaded for llama.cpp."""
    c = cfg()
    gguf_dir = Path(c.get("HF_CACHE_DIR", "")) / "gguf"
    out = []
    if gguf_dir.is_dir():
        for f in gguf_dir.rglob("*.gguf"):
            try:
                out.append({"id": f.name, "path": str(f),
                            "size_gb": round(f.stat().st_size / 1e9, 1), "engine": "llamacpp"})
            except Exception:
                pass
    return out


def storage_usage() -> dict:
    """Cache path, total used by models, and disk free/total."""
    c = cfg()
    cache_root = c.get("HF_CACHE_DIR", "")
    used_models = 0.0
    for m in hf_cache_models():
        used_models += (m.get("size_gb") or 0)
    for m in gguf_cache_models():
        used_models += (m.get("size_gb") or 0)
    disk = {"free_gb": None, "total_gb": None, "used_pct": None}
    try:
        import shutil as _sh
        du = _sh.disk_usage(cache_root if os.path.isdir(cache_root) else "/")
        disk = {"free_gb": round(du.free / 1e9, 1), "total_gb": round(du.total / 1e9, 1),
                "used_pct": round((du.used / du.total) * 100, 1)}
    except Exception:
        pass
    return {"cache_path": cache_root, "models_gb": round(used_models, 1), "disk": disk}


# ------------------------------------------------------------
# NAS archive: bulk model storage, copy to local when needed
# ------------------------------------------------------------
def nas_path() -> str:
    return cfg().get("NAS_PATH", os.environ.get("GX10_NAS_PATH", "")).strip()


def nas_status() -> dict:
    """Is the NAS configured and mounted/reachable?"""
    p = nas_path()
    if not p:
        return {"configured": False, "mounted": False, "path": "",
                "detail": "No NAS path set. Configure it to enable the archive."}
    pp = Path(p)
    mounted = pp.is_dir() and os.access(p, os.R_OK)
    disk = {}
    if mounted:
        try:
            import shutil as _sh
            du = _sh.disk_usage(p)
            disk = {"free_gb": round(du.free / 1e9, 1), "total_gb": round(du.total / 1e9, 1)}
        except Exception:
            pass
    return {"configured": True, "mounted": mounted, "path": p, "disk": disk,
            "detail": "" if mounted else f"NAS path {p} not reachable. Is it mounted?"}


def nas_models() -> list:
    """List models archived on the NAS. Looks for HF-style model dirs and GGUF files."""
    p = nas_path()
    if not p or not Path(p).is_dir():
        return []
    out = []
    root = Path(p)
    # HF hub-style: models--org--name dirs (either at root or under a hub/ subdir)
    for base in (root, root / "hub"):
        if base.is_dir():
            for d in base.glob("models--*"):
                name = d.name.replace("models--", "").replace("--", "/", 1).replace("--", "-")
                size_gb = None
                try:
                    size_gb = round(sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e9, 1)
                except Exception:
                    pass
                out.append({"id": name, "kind": "safetensors", "size_gb": size_gb, "path": str(d)})
    # plain org/name directory layout (some people store that way)
    for org in root.iterdir() if root.is_dir() else []:
        if org.is_dir() and not org.name.startswith("models--") and org.name != "hub":
            for sub in org.iterdir():
                if sub.is_dir() and any(sub.glob("*.safetensors")):
                    size_gb = None
                    try:
                        size_gb = round(sum(f.stat().st_size for f in sub.rglob("*") if f.is_file()) / 1e9, 1)
                    except Exception:
                        pass
                    out.append({"id": f"{org.name}/{sub.name}", "kind": "safetensors",
                                "size_gb": size_gb, "path": str(sub)})
    # GGUF files anywhere under the NAS
    for f in root.rglob("*.gguf"):
        try:
            out.append({"id": f.name, "kind": "gguf",
                        "size_gb": round(f.stat().st_size / 1e9, 1), "path": str(f)})
        except Exception:
            pass
    # which are already local?
    local_ids = {m["id"] for m in hf_cache_models()} | {m["id"] for m in gguf_cache_models()}
    for m in out:
        m["local"] = m["id"] in local_ids
    return out


def _nas_copy_worker(model_id: str, src_path: str, kind: str, jid=None):
    """Copy a model from the NAS to the local HF cache, with progress.
    Copies to a temp path and renames on success, so a cancelled or failed copy
    never leaves a partial model in the cache that hf_cache_models() would list."""
    import shutil as _sh
    c = cfg()
    cache = Path(c.get("HF_CACHE_DIR", ""))
    try:
        total = sum(f.stat().st_size for f in Path(src_path).rglob("*") if f.is_file()) \
            if Path(src_path).is_dir() else Path(src_path).stat().st_size
    except Exception:
        total = 0
    _set_dl(_jid=jid, active=True, model=model_id, status="copying", percent=0,
            detail=f"Copying {model_id} from NAS to local...", started=time.time(), error=None)
    tmp = None
    try:
        if kind == "gguf":
            dest_dir = cache / "gguf"
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / Path(src_path).name
            tmp = dest.with_name(dest.name + ".partial")
            if tmp.exists():
                tmp.unlink()
            _copy_with_progress(Path(src_path), tmp, total, model_id, jid)
            if jid is not None and not _job_is_current(jid):
                tmp.unlink(missing_ok=True)
                return
            os.replace(tmp, dest)
        else:
            dirname = "models--" + model_id.replace("/", "--")
            dest = cache / "hub" / dirname
            tmp = cache / "hub" / (dirname + ".partial")
            _sh.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True, exist_ok=True)
            # copy tree file by file for progress
            copied = 0
            for f in Path(src_path).rglob("*"):
                if jid is not None and not _job_is_current(jid):
                    _sh.rmtree(tmp, ignore_errors=True)
                    return
                if f.is_file():
                    rel = f.relative_to(src_path)
                    target = tmp / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    _sh.copy2(f, target)
                    copied += f.stat().st_size
                    if total:
                        pct = int(copied / total * 100)
                        _set_dl(_jid=jid, active=True, model=model_id, status="copying", percent=pct,
                                detail=f"Copying from NAS... {pct}%")
            _sh.rmtree(dest, ignore_errors=True)
            os.replace(tmp, dest)
        _set_dl(_jid=jid, active=False, status="ready", percent=100,
                detail=f"{model_id} copied to local. Ready to load.", error=None)
        invalidate_cache("library", "nas")
        log_line(f"[nas] copied {model_id} from NAS to local cache")
    except Exception as e:
        if tmp is not None:
            try:
                if tmp.is_dir():
                    _sh.rmtree(tmp, ignore_errors=True)
                elif tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
        _set_dl(_jid=jid, active=False, status="error", error=f"NAS copy failed: {e}")
        log_line(f"[nas] copy failed for {model_id}: {e}")


def _copy_with_progress(src: Path, dest: Path, total: int, model_id: str, jid=None):
    import shutil as _sh
    if total > 5e9:  # large single file: chunked copy with progress
        copied = 0
        with open(src, "rb") as fi, open(dest, "wb") as fo:
            while True:
                if jid is not None and not _job_is_current(jid):
                    return
                chunk = fi.read(16 * 1024 * 1024)
                if not chunk:
                    break
                fo.write(chunk)
                copied += len(chunk)
                pct = int(copied / total * 100) if total else 0
                _set_dl(_jid=jid, active=True, model=model_id, status="copying", percent=pct,
                        detail=f"Copying from NAS... {pct}%")
    else:
        _sh.copy2(src, dest)


def bring_from_nas(model_id: str) -> dict:
    """Stage a NAS-archived model onto local NVMe (background copy)."""
    archived = {m["id"]: m for m in nas_models()}
    if model_id not in archived:
        return {"ok": False, "detail": "That model isn't in the NAS archive."}
    m = archived[model_id]
    jid, busy = _claim_job(model_id, status="copying",
                           detail=f"Copying {model_id} from NAS to local...")
    if jid is None:
        return {"ok": False, "detail": busy}
    threading.Thread(target=_nas_copy_worker,
                     args=(model_id, m["path"], m["kind"], jid), daemon=True).start()
    return {"ok": True, "detail": f"Copying {model_id} from NAS to local. Watch progress in the banner."}


def delete_cached_model(model_id: str) -> dict:
    """Delete a cached model (safetensors dir or GGUF file).
    Refuses only if the model is ACTUALLY serving right now (not merely the
    configured MODEL). A model that failed to load can be deleted."""
    c = cfg()
    if model_id == currently_serving() or model_id == llama_serving():
        return {"ok": False, "detail": "That model is actively serving. Unload it first."}
    # if it's the configured-but-not-serving model, clear it from cluster.env so
    # the cluster doesn't try to auto-load a model we're about to delete
    if model_id == c.get("MODEL", ""):
        write_env_value(CLUSTER_ENV, "MODEL", "")
        log_line(f"[library] cleared {model_id} from config (failed/idle) before delete")
    cache = Path(c.get("HF_CACHE_DIR", ""))
    # GGUF file?
    if model_id.lower().endswith(".gguf"):
        f = cache / "gguf" / model_id
        if f.is_file():
            try:
                freed = round(f.stat().st_size / 1e9, 1)
                f.unlink()
                log_line(f"[library] deleted GGUF {model_id} ({freed}GB)")
                return {"ok": True, "detail": f"Deleted {model_id}, freed ~{freed}GB."}
            except Exception as e:
                return {"ok": False, "detail": f"Delete failed: {e}"}
        return {"ok": False, "detail": "GGUF file not found."}
    # safetensors model dir: org/name -> models--org--name
    dirname = "models--" + model_id.replace("/", "--")
    d = cache / "hub" / dirname
    if d.is_dir():
        try:
            total = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            import shutil as _sh
            _sh.rmtree(d)
            freed = round(total / 1e9, 1)
            log_line(f"[library] deleted {model_id} ({freed}GB)")
            return {"ok": True, "detail": f"Deleted {model_id}, freed ~{freed}GB."}
        except Exception as e:
            return {"ok": False, "detail": f"Delete failed: {e}"}
    return {"ok": False, "detail": "Model not found in cache."}


def set_cache_path(new_path: str) -> dict:
    """Change the cache location in cluster.env (applies on next start)."""
    if not new_path.strip():
        return {"ok": False, "detail": "Empty path."}
    p = Path(new_path)
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return {"ok": False, "detail": f"Could not create {new_path}: {e}"}
    write_env_value(CLUSTER_ENV, "HF_CACHE_DIR", new_path)
    log_line(f"[library] cache path set to {new_path}")
    return {"ok": True, "detail": f"Cache path set to {new_path}. Applies on next cluster start.",
            "needs_relaunch": True}


def library_all() -> dict:
    """Unified library: safetensors + GGUF, with sizes, engine, active flag.
    'active' means actually serving right now (vLLM or llama.cpp), NOT merely the
    configured MODEL. A model that failed to load is not active and can be deleted."""
    serving = currently_serving()      # vLLM API actually responding with this id
    llama = llama_serving()            # llama.cpp actually serving
    configured = cfg().get("MODEL", "")
    st_models = hf_cache_models()
    cached_ids = {m["id"] for m in st_models}
    draft_ids = set(DFLASH_PAIRS.values())
    def mk(m, eng):
        is_serving = (m["id"] == serving) or (m["id"] == llama)
        out = {**m, "engine": eng, "active": is_serving,
               "configured": m["id"] == configured and not is_serving}
        if m["id"] in draft_ids:
            out["is_draft"] = True            # speculative draft, not loadable on its own
        if m["id"] in DFLASH_PAIRS:
            out["dflash"] = DFLASH_PAIRS[m["id"]] in cached_ids  # base with draft available
        return out
    st = [mk(m, "vllm") for m in st_models]
    gg = [mk(m, "llamacpp") for m in gguf_cache_models()]
    items = sorted(st + gg, key=lambda x: -(x.get("size_gb") or 0))
    return {"models": items, "storage": storage_usage()}


FAVORITES_FILE = Path(os.environ.get("GX10_STATE_DIR", str(Path.home()))) / ".gx10-favorites.json"


def get_favorites() -> list:
    try:
        if FAVORITES_FILE.exists():
            return json.loads(FAVORITES_FILE.read_text())
    except Exception:
        pass
    return []


def toggle_favorite(model: str) -> dict:
    favs = get_favorites()
    if model in favs:
        favs = [f for f in favs if f != model]
    else:
        favs.append(model)
    try:
        FAVORITES_FILE.write_text(json.dumps(favs))
    except Exception as e:
        return {"ok": False, "detail": f"Could not save favorites: {e}", "favorites": favs}
    return {"ok": True, "favorites": favs}


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
# Engine abstraction: vLLM (multi-node cluster) vs llama.cpp (single-node GGUF)
# ------------------------------------------------------------
LLAMA_CONTAINER = "llama-node"
LLAMA_PORT = os.environ.get("GX10_LLAMA_PORT", "8001")


def detect_engine(model: str, files: list = None) -> dict:
    """Decide which engine should serve a model, by format.
    GGUF -> llama.cpp (single-node). safetensors -> vLLM (cluster-capable)."""
    low = model.lower()
    # explicit signals in the repo id
    if "gguf" in low:
        return {"engine": "llamacpp", "reason": "GGUF model, runs on llama.cpp (single node)."}
    # if we have the file list, look for .gguf
    if files:
        if any(f.lower().endswith(".gguf") for f in files):
            return {"engine": "llamacpp", "reason": "Repo ships GGUF files, runs on llama.cpp (single node)."}
        if any(f.lower().endswith(".safetensors") for f in files):
            return {"engine": "vllm", "reason": "safetensors model, runs on vLLM (cluster-capable)."}
    # default: vLLM, which is the cluster path
    return {"engine": "vllm", "reason": "Defaulting to vLLM (cluster). Use llama.cpp for GGUF models."}


def llama_serving() -> str:
    """What llama.cpp is serving right now, or '' if down."""
    rc, out, _ = run(["curl", "-s", "--max-time", "3", f"http://localhost:{LLAMA_PORT}/v1/models"], timeout=5)
    if rc == 0 and out:
        try:
            return json.loads(out)["data"][0]["id"]
        except Exception:
            return ""
    return ""


def active_engine() -> dict:
    """Which engine is currently serving, if any."""
    vllm_model = currently_serving()  # checks vLLM API on 8000
    llama_model = llama_serving()      # checks llama.cpp API on 8001
    if vllm_model:
        return {"engine": "vllm", "model": vllm_model, "port": cfg().get("API_PORT", "8000")}
    if llama_model:
        return {"engine": "llamacpp", "model": llama_model, "port": LLAMA_PORT}
    return {"engine": None, "model": None, "port": None}


def _llama_start_worker(repo: str, gguf_file: str, ctx: int, jid=None):
    """Download (if needed) + start llama.cpp in the background, reporting progress."""
    script = str(KIT_DIR / "llama-serve.sh")
    _set_dl(_jid=jid, active=True, model=gguf_file, status="loading", percent=0,
            detail=f"Starting llama.cpp with {gguf_file} (downloads first if needed)...",
            started=time.time(), error=None)
    log_line(f"[llama] starting {repo} / {gguf_file}")
    # no short timeout: a GGUF download can take a long time
    rc, out, err = run(["bash", script, repo, gguf_file, str(ctx)], timeout=3600)
    if jid is not None and not _job_is_current(jid):
        return
    if rc != 0:
        msg = (err or out or "unknown error")[:240]
        _set_dl(_jid=jid, active=False, status="error", error=f"llama.cpp failed: {msg}")
        log_line(f"[llama] start failed: {msg}")
        invalidate_cache("engine_active", "library")
        return
    # wait for the server to actually answer before calling it ready
    for _ in range(60):
        if jid is not None and not _job_is_current(jid):
            return
        if llama_serving():
            _set_dl(_jid=jid, active=False, status="ready", percent=100,
                    detail=f"llama.cpp serving {gguf_file} on :{LLAMA_PORT}.", error=None)
            log_line(f"[llama] {gguf_file} is serving")
            save_serving_state("llamacpp", gguf_file, repo=repo)
            invalidate_cache("engine_active", "library")
            return
        time.sleep(2)
    # started but not answering yet; leave a soft note
    _set_dl(_jid=jid, active=False, status="ready", detail=f"llama.cpp started {gguf_file}; warming up.")
    invalidate_cache("engine_active", "library")


def start_llama(repo: str, gguf_file: str, ctx: int = 32768):
    """Launch llama.cpp serving a GGUF model (async: download + start in background)."""
    script = str(KIT_DIR / "llama-serve.sh")
    if not Path(script).exists():
        return {"ok": False, "detail": "llama-serve.sh not found in cluster kit."}
    jid, busy = _claim_job(gguf_file, status="loading",
                           detail=f"Starting llama.cpp with {gguf_file}...")
    if jid is None:
        return {"ok": False, "detail": busy}
    threading.Thread(target=_llama_start_worker, args=(repo, gguf_file, ctx, jid), daemon=True).start()
    return {"ok": True, "detail": f"Starting llama.cpp with {gguf_file}. Watch progress in the banner."}


def stop_llama():
    script = str(KIT_DIR / "llama-serve.sh")
    run(["bash", script, "stop"], timeout=20)
    clear_serving_state()
    log_line("[llama] stopped")
    return {"ok": True, "detail": "llama.cpp server stopped."}


def dflash_draft_for(model: str):
    """Return the DFlash draft repo for `model` if it's paired AND the draft is cached
    locally, else None. Used to route a base-model load to single-node vLLM+DFlash."""
    draft = DFLASH_PAIRS.get(model)
    if not draft:
        return None
    cached = {m["id"] for m in hf_cache_models()}
    return draft if draft in cached else None


def _dflash_start_worker(model: str, draft: str, jid=None):
    """Start single-node vLLM+DFlash in the background, reporting progress. The BF16
    base is large (~54GB), so first load takes several minutes before it answers."""
    script = str(KIT_DIR / "vllm-dflash-serve.sh")
    _set_dl(_jid=jid, active=True, model=model, status="loading", percent=0,
            detail=f"Starting vLLM+DFlash for {model} (loading ~54GB base; first start is slow)...",
            started=time.time(), error=None)
    log_line(f"[dflash] starting {model} (draft {draft})")
    cmd = (f"DFLASH_MODEL={shlex.quote(model)} DFLASH_DRAFT={shlex.quote(draft)} "
           f"bash {shlex.quote(script)} start")
    rc, out, err = run(["bash", "-c", cmd], timeout=600)
    if jid is not None and not _job_is_current(jid):
        return
    if rc != 0:
        msg = (err or out or "unknown error")[:240]
        _set_dl(_jid=jid, active=False, status="error", error=f"vLLM+DFlash failed: {msg}")
        log_line(f"[dflash] start failed: {msg}")
        invalidate_cache("engine_active", "library")
        return
    # wait (up to ~9 min) for the API to actually answer before calling it ready
    for _ in range(180):
        if jid is not None and not _job_is_current(jid):
            return
        if currently_serving():
            _set_dl(_jid=jid, active=False, status="ready", percent=100,
                    detail=f"vLLM+DFlash serving {model}.", error=None)
            log_line(f"[dflash] {model} is serving")
            save_serving_state("dflash", model, draft=draft)
            invalidate_cache("engine_active", "library")
            return
        time.sleep(3)
    _set_dl(_jid=jid, active=False, status="ready", detail=f"vLLM+DFlash started {model}; warming up.")
    invalidate_cache("engine_active", "library")


def start_dflash(model: str, draft: str):
    """Launch single-node vLLM + DFlash (async: load + start in background)."""
    script = str(KIT_DIR / "vllm-dflash-serve.sh")
    if not Path(script).exists():
        return {"ok": False, "detail": "vllm-dflash-serve.sh not found in cluster kit."}
    jid, busy = _claim_job(model, status="loading", detail=f"Starting vLLM+DFlash for {model}...")
    if jid is None:
        return {"ok": False, "detail": busy}
    threading.Thread(target=_dflash_start_worker, args=(model, draft, jid), daemon=True).start()
    return {"ok": True, "detail": f"Starting vLLM+DFlash for {model}. First load is slow (~54GB); watch the banner."}


def stop_dflash():
    script = str(KIT_DIR / "vllm-dflash-serve.sh")
    run(["bash", script, "stop"], timeout=30)
    clear_serving_state()
    log_line("[dflash] stopped")
    return {"ok": True, "detail": "vLLM+DFlash stopped."}



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
    return JSONResponse(cached("status", 3.0, gather_status))


@app.get("/api/metrics")
def api_metrics():
    return JSONResponse(cached("metrics", 1.5, gather_metrics))


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


def llama_logs(lines=120):
    # is the container even there?
    rcps, names, _ = run(["docker", "ps", "-a", "--format", "{{.Names}}"], timeout=8)
    if rcps == 0 and LLAMA_CONTAINER not in (names or "").split():
        return "llama.cpp container not started yet. Load a GGUF model to start it."
    # docker logs writes to stdout+stderr; capture combined. llama.cpp logs to stderr.
    rc, out, err = run(["docker", "logs", "--tail", str(lines), LLAMA_CONTAINER], timeout=10)
    if rc != 0:
        return f"Could not read llama.cpp logs: {(err or 'unknown')[:160]}"
    combined = "\n".join(p for p in (out, err) if p).strip()
    if not combined:
        # running but quiet: confirm it's up and serving
        if llama_serving():
            return "llama.cpp is serving (no recent log lines). Server is up on the API port."
        return "llama.cpp container is up but has produced no log output yet. It may still be loading the model."
    return combined


@app.get("/api/logs")
def api_logs(engine: str = "vllm"):
    if engine == "llamacpp":
        return JSONResponse({"engine": "llamacpp", "log": llama_logs()})
    return JSONResponse({"engine": "vllm", "log": vllm_logs()})


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


def hf_repo_files(repo: str) -> list:
    """List filenames in an HF repo (for engine detection and GGUF picking)."""
    rc, out, _ = run(["curl", "-s", "--max-time", "8",
                      f"https://huggingface.co/api/models/{repo}"], timeout=12)
    if rc == 0 and out:
        try:
            data = json.loads(out)
            return [s["rfilename"] for s in data.get("siblings", [])]
        except Exception:
            pass
    return []


def pick_gguf_file(files: list, prefer: str = None) -> str:
    """Choose a sensible GGUF file from a repo's file list."""
    ggufs = [f for f in files if f.lower().endswith(".gguf")]
    if not ggufs:
        return ""
    if prefer:
        for f in ggufs:
            if prefer.lower() in f.lower():
                return f
    # prefer a balanced quant; order of preference
    for tag in ("q4_k_m", "q4_k_xl", "ud-q4", "q5_k_m", "q4_0", "q8_0"):
        for f in ggufs:
            if tag in f.lower():
                return f
    return ggufs[0]


def route_model(model: str, gguf_file: str = None, ctx: int = 32768) -> dict:
    """Engine-agnostic load: detect format, route to vLLM or llama.cpp.
    Runs the actual work in a background thread so the request returns immediately."""
    jid, busy = _claim_job(model, status="loading", detail=f"Loading {model}...")
    if jid is None:
        return {"ok": False, "detail": busy}

    def _route_worker():
        # Case 1: model is already a local GGUF file (from the Library).
        # Serve it directly on llama.cpp; do not treat the filename as a repo.
        local_gguf = {m["id"]: m for m in gguf_cache_models()}
        if model.lower().endswith(".gguf") or model in local_gguf:
            chosen_file = model if model.lower().endswith(".gguf") else local_gguf[model]["id"]
            if currently_serving():
                log_line("[engine] stopping vLLM to free the box for llama.cpp")
                _set_dl(_jid=jid, active=True, model=model, status="loading", detail="Stopping vLLM...")
                run(["docker", "exec", CONTAINER, "pkill", "-f", "vllm serve"], timeout=10)
            # repo arg is empty: the file is already local, llama-serve.sh serves it directly
            _llama_start_worker("", chosen_file, ctx, jid)
            return

        files = hf_repo_files(model)
        decision = detect_engine(model, files)
        if decision["engine"] == "llamacpp":
            chosen = gguf_file or pick_gguf_file(files)
            if not chosen:
                _set_dl(_jid=jid, active=False, status="error",
                        error=f"No GGUF file found in {model}. Pick a GGUF repo or specify a file.")
                return
            if currently_serving():
                log_line("[engine] stopping vLLM to free the box for llama.cpp")
                _set_dl(_jid=jid, active=True, model=model, status="loading", detail="Stopping vLLM...")
                run(["docker", "exec", CONTAINER, "pkill", "-f", "vllm serve"], timeout=10)
            # run the llama start+download inline (we're already in a worker thread)
            _llama_start_worker(model, chosen, ctx, jid)
            return
        # vLLM path. If this base model has a cached DFlash draft, serve it single-node
        # via vLLM + DFlash speculative decoding (faster on code/structured output) rather
        # than the multi-node Ray cluster path.
        draft = dflash_draft_for(model)
        if draft:
            if llama_serving():
                log_line("[engine] stopping llama.cpp to free the box for vLLM+DFlash")
                _set_dl(_jid=jid, active=True, model=model, status="loading", detail="Stopping llama.cpp...")
                stop_llama()
            _dflash_start_worker(model, draft, jid)
            return
        if llama_serving():
            log_line("[engine] stopping llama.cpp to free the box for vLLM")
            _set_dl(_jid=jid, active=True, model=model, status="loading", detail="Stopping llama.cpp...")
            stop_llama()
        # we're already in a worker thread; run the load/download logic inline
        cached = {m["id"] for m in hf_cache_models()}
        if model in cached:
            _load_worker(model, jid)
        else:
            _download_worker(model, True, jid)

    threading.Thread(target=_route_worker, daemon=True).start()
    return {"ok": True, "detail": f"Loading {model}. Watch progress in the activity log.", "async": True}


@app.post("/api/model")
def api_model(payload: dict = Body(...)):
    model = (payload.get("model") or "").strip()
    if not model:
        return JSONResponse({"ok": False, "detail": "No model provided"}, status_code=400)
    # engine can be forced; otherwise auto-route by format
    forced = payload.get("engine")
    if forced == "vllm":
        return JSONResponse({**swap_model(model), "engine": "vllm"})
    if forced == "llamacpp":
        files = hf_repo_files(model)
        chosen = payload.get("gguf_file") or pick_gguf_file(files)
        if not chosen:
            return JSONResponse({"ok": False, "detail": f"No GGUF file in {model}."}, status_code=400)
        res = start_llama(model, chosen, _as_int(payload.get("ctx"), 32768))
        # only stop vLLM once the llama job is actually claimed, so a "busy" rejection
        # can't leave the box with vLLM killed and nothing started
        if res.get("ok") and currently_serving():
            run(["docker", "exec", CONTAINER, "pkill", "-f", "vllm serve"], timeout=10)
        return JSONResponse({**res, "engine": "llamacpp", "gguf_file": chosen})
    return JSONResponse(route_model(model, payload.get("gguf_file"), _as_int(payload.get("ctx"), 32768)))


@app.get("/api/models/curated")
def api_models_curated():
    return JSONResponse({"models": curated_models(), "budget": {
        "pooled_gb": POOLED_GB, "usable_gb": USABLE_GB, "single_node_gb": SINGLE_NODE_GB}})


@app.get("/api/models/cached")
def api_models_cached():
    return JSONResponse({"models": hf_cache_models()})


@app.get("/api/library")
def api_library():
    data = cached("library", 5.0, library_all)
    return JSONResponse({**data, "favorites": get_favorites()})


@app.post("/api/favorites/toggle")
def api_favorites_toggle(payload: dict = Body(...)):
    model = (payload.get("model") or "").strip()
    if not model:
        return JSONResponse({"ok": False, "detail": "No model"}, status_code=400)
    return JSONResponse(toggle_favorite(model))


@app.post("/api/library/delete")
def api_library_delete(payload: dict = Body(...)):
    model = (payload.get("model") or "").strip()
    if not model:
        return JSONResponse({"ok": False, "detail": "No model provided"}, status_code=400)
    result = delete_cached_model(model)
    invalidate_cache("library", "engine_active")
    return JSONResponse(result)


@app.get("/api/storage")
def api_storage():
    return JSONResponse(storage_usage())


@app.post("/api/storage/path")
def api_storage_path(payload: dict = Body(...)):
    path = (payload.get("path") or "").strip()
    return JSONResponse(set_cache_path(path))


@app.get("/api/nas/status")
def api_nas_status():
    return JSONResponse(nas_status())


@app.get("/api/nas/models")
def api_nas_models():
    return JSONResponse({"models": cached("nas", 30.0, nas_models), "status": nas_status()})


@app.post("/api/nas/path")
def api_nas_path(payload: dict = Body(...)):
    path = (payload.get("path") or "").strip()
    write_env_value(CLUSTER_ENV, "NAS_PATH", path)
    invalidate_cache("nas")
    s = nas_status()
    return JSONResponse({"ok": True, "detail": f"NAS path set to {path}." if path else "NAS path cleared.",
                         "status": s})


@app.post("/api/nas/bring")
def api_nas_bring(payload: dict = Body(...)):
    model = (payload.get("model") or "").strip()
    if not model:
        return JSONResponse({"ok": False, "detail": "No model provided"}, status_code=400)
    return JSONResponse(bring_from_nas(model))


@app.get("/api/models/search")
def api_models_search(q: str = ""):
    return JSONResponse({"results": hf_search(q)})


@app.post("/api/models/test")
def api_models_test(payload: dict = Body(...)):
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        return JSONResponse({"ok": False, "detail": "No prompt provided"}, status_code=400)
    return JSONResponse(test_prompt(prompt, _as_int(payload.get("max_tokens"), 128)))


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


@app.post("/api/model/cancel")
def api_model_cancel():
    return JSONResponse(cancel_load())


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


@app.get("/api/telemetry")
def api_telemetry():
    return JSONResponse(cached("telemetry", 1.5, gather_telemetry))


@app.get("/api/telemetry/history")
def api_telemetry_history(window: str = "hour"):
    secs = {"hour": 3600, "day": 86400, "week": 604800}.get(window, 3600)
    return JSONResponse({"window": window, "samples": db_history(secs)})


@app.post("/api/benchmark")
def api_benchmark(payload: dict = Body(...)):
    concurrency = max(1, min(64, _as_int(payload.get("concurrency"), 4)))
    prompt = (payload.get("prompt") or
              "Write a short paragraph about the future of computing.").strip()
    max_tokens = max(1, min(2048, _as_int(payload.get("max_tokens"), 128)))
    return JSONResponse(start_benchmark(concurrency, prompt, max_tokens))


@app.get("/api/benchmark/status")
def api_benchmark_status():
    with _benchmark_lock:
        return JSONResponse(dict(_benchmark))


@app.get("/api/benchmark/history")
def api_benchmark_history():
    return JSONResponse({"benchmarks": benchmark_history()})


@app.post("/api/chat")
def api_chat(payload: dict = Body(...)):
    """Streaming chat proxy to whichever engine is serving. Keeps the browser
    single-origin; the frontend times TTFT (first token) and tok/s off this stream."""
    eng = active_engine()
    port = eng.get("port") or cfg().get("API_PORT", "8000")
    model = eng.get("model")
    if not model:
        return JSONResponse({"ok": False, "detail": "No model is serving. Load one first."},
                            status_code=409)
    messages = payload.get("messages") or []
    if not isinstance(messages, list) or not messages:
        return JSONResponse({"ok": False, "detail": "No messages provided."}, status_code=400)
    max_tokens = max(1, min(8192, _as_int(payload.get("max_tokens"), 512)))
    req_body = {"model": model, "messages": messages, "max_tokens": max_tokens,
                "stream": True,
                # ask the engine to emit a final usage chunk so the client can
                # report exact completion tokens (and thus accurate tok/s).
                "stream_options": {"include_usage": True}}
    # optional sampling control; only forward when the client sets it.
    if payload.get("temperature") is not None:
        try:
            req_body["temperature"] = max(0.0, min(2.0, float(payload["temperature"])))
        except (TypeError, ValueError):
            pass
    body = json.dumps(req_body).encode()

    def gen():
        import urllib.request
        req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
                                     data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                for raw in resp:
                    yield raw
        except Exception as e:
            yield ("data: " + json.dumps({"error": str(e)[:200]}) + "\n\n").encode()
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/engine/active")
def api_engine_active():
    return JSONResponse(cached("engine_active", 3.0, active_engine))


@app.post("/api/engine/detect")
def api_engine_detect(payload: dict = Body(...)):
    model = (payload.get("model") or "").strip()
    files = payload.get("files")
    if not model:
        return JSONResponse({"ok": False, "detail": "No model provided"}, status_code=400)
    return JSONResponse(detect_engine(model, files))


@app.post("/api/llama/start")
def api_llama_start(payload: dict = Body(...)):
    repo = (payload.get("repo") or "").strip()
    gguf = (payload.get("gguf_file") or "").strip()
    ctx = _as_int(payload.get("ctx"), 32768)
    if not repo or not gguf:
        return JSONResponse({"ok": False, "detail": "Need repo and gguf_file"}, status_code=400)
    return JSONResponse(start_llama(repo, gguf, ctx))


@app.post("/api/llama/stop")
def api_llama_stop():
    return JSONResponse(stop_llama())


@app.get("/api/image/overview")
def api_image_overview():
    return JSONResponse(cached("image_overview", 120.0, image_overview))


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


def _pick_port(preferred: int, host: str) -> int:
    """Bind the preferred port, or the next free fallback if it's taken (e.g. the
    systemd unit hands us 8080 but Open WebUI already has it). Keeps the panel from
    crash-looping on a port conflict and logs where it actually landed."""
    import socket
    bind_host = "" if host in ("0.0.0.0", "::", "") else host
    for p in dict.fromkeys([preferred, 8090, 8091, 8092, 8190]):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind((bind_host, p))
            s.close()
            return p
        except OSError:
            s.close()
    return preferred


if __name__ == "__main__":
    import uvicorn
    log_line(f"Panel starting. Kit dir: {KIT_DIR}")
    db_init()
    threading.Thread(target=_collector_loop, daemon=True).start()
    log_line(f"Metrics collector started (db: {DB_PATH.name}, cost rate ${COST_PER_MTOK}/Mtok)")
    port = _pick_port(PANEL_PORT, PANEL_HOST)
    if port != PANEL_PORT:
        log_line(f"Port {PANEL_PORT} is in use; serving the panel on {port} instead.")
    uvicorn.run(app, host=PANEL_HOST, port=port, log_level="warning")
