import asyncio
import ctypes
import json
import logging
import os
import secrets
import socket
import subprocess
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

API_HOST = "0.0.0.0"
API_PORT = 8050
IDLE_TIMEOUT_MINUTES = 60
LOW_PORT = 8100
HIGH_PORT = 8900
CREATE_NO_WINDOW = 0x08000000
VALID_KV_TYPES = {"f16", "f32", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "i8", "i16", "i32"}

BIN_CPU = r"D:\llms\llama-b10809-bin-win-cpu-x64\llama-server.exe"
BIN_VULKAN = r"D:\llms\llama-b10809-bin-win-vulkan-x64\llama-server.exe"
BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "ornith": {
        "path": r"D:\llms\models\testing\ornith-ai_Ornith-1.5-9B-GGUF\Ornith-1.5-9B-Q4_K_M.gguf",
        "description": "Ornith-1.5-9B (qwen35), 33 blocks, native 262k ctx, strong tool calling",
        "ctx": 4096,
        "jinja": True,
        "quant": "Q4_K_M",
        "quant_pct": "~30",
        "layers": 33,
        "native_ctx": 262144,
        "kv_bytes_per_token": {"f16": 36, "q8_0": 18, "q4_0": 9},
    },
    "gpt-oss": {
        "path": r"D:\llms\models\testing\unsloth_gpt-oss-20b-GGUF\gpt-oss-20b-Q4_K_M.gguf",
        "description": "OpenAI gpt-oss-20b MoE (20B total / 3.6B active per token), 24 blocks, native 131k ctx",
        "ctx": 2048,
        "jinja": True,
        "quant": "Q4_K_M",
        "quant_pct": "~28",
        "layers": 24,
        "native_ctx": 131072,
        "kv_bytes_per_token": {"f16": 48, "q8_0": 24, "q4_0": 12},
    },
    "qwen3-0.6b": {
        "path": r"D:\llms\models\testing\Qwen_Qwen3-0.6B-GGUF\Qwen3-0.6B-Q8_0.gguf",
        "description": "Tiny Qwen3-0.6B (28 blocks), quick sanity checks",
        "ctx": 4096,
        "jinja": True,
        "quant": "Q8_0",
        "quant_pct": "~50",
        "layers": 28,
        "native_ctx": 40960,
        "kv_bytes_per_token": {"f16": 112, "q8_0": 56, "q4_0": 28},
    },
}

FIELD_NOTES = {
    "quant": "weight quant format (baked into file)",
    "quant_pct": "file size as % of the fp16 original",
    "layers": "total blocks = max value for gpu_layers",
    "ctx": "default context window in tokens (input+output share it)",
    "native_ctx": "model's absolute max context window",
    "kv_bytes_per_token": "KV-cache RAM needed per token (KB), per f16/q8_0/q4_0",
    "size_gb": "file size on disk",
    "gpu_layers": "blocks offloaded to GPU (0 = CPU only)",
    "ctk": "KV-cache dtype for K half (None → f16)",
    "ctv": "KV-cache dtype for V half (None → f16)",
}

# Static measured speeds, RAM-only (CPU) mode: llama-bench pp128/tg64,
# 4 threads, i5-3470 (see llm-theory/hardware-and-measurements.md).
RAM_ONLY_SPEEDS = {
    "ornith": {"prompt_tps": 6.1, "generation_tps": 3.3},
    "gpt-oss": {"prompt_tps": 7.8, "generation_tps": 4.8},
    "qwen3-0.6b": {"prompt_tps": 231.2, "generation_tps": 23.9},
}
RAM_ONLY_BASIS = "static measured CPU speeds (llama-bench pp128/tg64, 4 threads) — not live"

# Static measured GPU-offload samples (llama-bench pp128/tg64, 4 threads, RX 580 Vulkan):
# ngl = blocks offloaded to GPU; 0 = the RAM-only numbers above.
# gpt-oss past 12 layers OOMs on the 8 GB card; ornith was measured at 0/16/25/33.
GPU_SPEED_SAMPLES = {
    "ornith": {
        0: {"prompt_tps": 6.1, "generation_tps": 3.3},
        16: {"prompt_tps": 70.4, "generation_tps": 4.2},
        25: {"prompt_tps": 85.9, "generation_tps": 7.1},
        33: {"prompt_tps": 69.2, "generation_tps": 15.2},
    },
    "gpt-oss": {
        0: {"prompt_tps": 7.8, "generation_tps": 4.8},
        6: {"prompt_tps": 25.3, "generation_tps": 6.8},
        12: {"prompt_tps": 45.2, "generation_tps": 9.6},
    },
    "qwen3-0.6b": {
        0: {"prompt_tps": 231.2, "generation_tps": 23.9},
        28: {"prompt_tps": 661.1, "generation_tps": 41.7},
    },
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("model_manager")


def _model_info(name: str) -> dict:
    """Registry entry + live file size, so /models exposes quant/layers/KV facts."""
    info = dict(MODELS[name])
    path = Path(info["path"])
    info["size_gb"] = round(os.path.getsize(path) / (1024**3), 2) if path.exists() else None
    return info


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_uint32),
        ("dwMemoryLoad", ctypes.c_uint32),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]


def ram_info():
    """Available RAM via GlobalMemoryStatusEx (pure stdlib ctypes)."""
    ms = _MEMORYSTATUSEX()
    ms.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
        return None
    return {
        "load_percent": ms.dwMemoryLoad,
        "total_bytes": ms.ullTotalPhys,
        "free_bytes": ms.ullAvailPhys,
        "used_bytes": ms.ullTotalPhys - ms.ullAvailPhys,
    }


_VK_BINDIR = str(Path(BIN_VULKAN).parent)
_vk_loaded = False
_vk_lib = None
_vk_dir_handle = None


def _load_vk_lib():
    """Load ggml-vulkan.dll from the Vulkan bin folder (lazy, pure stdlib)."""
    global _vk_loaded, _vk_lib, _vk_dir_handle
    if _vk_loaded:
        return _vk_lib
    _vk_loaded = True
    dll_path = os.path.join(_VK_BINDIR, "ggml-vulkan.dll")
    if not os.path.exists(dll_path):
        logger.warning("no ggml-vulkan.dll in %s", _VK_BINDIR)
        return None
    try:
        _vk_dir_handle = os.add_dll_directory(_VK_BINDIR)
        ctypes.CDLL(os.path.join(_VK_BINDIR, "ggml-base.dll"))
        ctypes.CDLL(os.path.join(_VK_BINDIR, "ggml.dll"))
        lib = ctypes.CDLL(dll_path)
        lib.ggml_backend_vk_get_device_count.restype = ctypes.c_int
        lib.ggml_backend_vk_get_device_count.argtypes = []
        lib.ggml_backend_vk_get_device_memory.restype = None
        lib.ggml_backend_vk_get_device_memory.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        _vk_lib = lib
    except Exception as ex:
        logger.warning("vulkan VRAM probe load failed: %s", ex)
    return _vk_lib


def vram_devices():
    """Free/total VRAM per Vulkan device via ggml_backend_vk_get_device_memory."""
    lib = _load_vk_lib()
    if lib is None:
        return None
    try:
        devices = []
        for i in range(lib.ggml_backend_vk_get_device_count()):
            free = ctypes.c_size_t(0)
            total = ctypes.c_size_t(0)
            lib.ggml_backend_vk_get_device_memory(i, ctypes.byref(free), ctypes.byref(total))
            devices.append({
                "device": i,
                "total_bytes": total.value,
                "free_bytes": free.value,
                "used_bytes": total.value - free.value,
            })
        return devices
    except Exception as ex:
        logger.warning("vulkan VRAM probe failed: %s", ex)
        return None


def _max_ctx(free_bytes, weights_bytes, kv_kb):
    """Max context (tokens) so weights + KV fit in the given free memory."""
    if not free_bytes or not kv_kb or kv_kb <= 0:
        return 0
    avail = free_bytes - weights_bytes
    if avail <= 0:
        return 0
    return int(avail / (kv_kb * 1024))


def fits_map(free_bytes):
    """Per-model max ctx (tokens) for each KV dtype, for a given free budget."""
    fits = {}
    for name in MODELS:
        info = _model_info(name)
        kv = info.get("kv_bytes_per_token") or {}
        weights_bytes = int((info.get("size_gb") or 0) * (1024**3))
        fits[name] = {
            dtype: _max_ctx(free_bytes, weights_bytes, kb)
            for dtype, kb in kv.items()
            if isinstance(kb, (int, float)) and kb > 0
        }
    return fits


def _est_presets(speed: dict) -> dict:
    """Precomputed RAM-only time presets (minutes) for one model."""
    pp, gen = speed["prompt_tps"], speed["generation_tps"]
    return {
        "gen_1000_min": round(1000 / gen / 60, 1),
        "pp_30000_min": round(30000 / pp / 60, 1),
        "pp_60000_min": round(60000 / pp / 60, 1),
        "turn_30000_1000_min": round(30000 / pp / 60 + 1000 / gen / 60, 1),
        "turn_60000_1000_min": round(60000 / pp / 60 + 1000 / gen / 60, 1),
    }


def _est_calc(speed: dict, prompt_tokens: int, output_tokens: int) -> dict:
    pp, gen = speed["prompt_tps"], speed["generation_tps"]
    prompt_seconds = prompt_tokens / pp
    gen_seconds = output_tokens / gen
    total_seconds = prompt_seconds + gen_seconds
    return {
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "prompt_seconds": round(prompt_seconds, 1),
        "generation_seconds": round(gen_seconds, 1),
        "total_seconds": round(total_seconds, 1),
        "total_minutes": round(total_seconds / 60, 1),
    }


def _ctx_max_split(model: str, ngl: int, free_vram: int, free_ram: int) -> dict:
    """Max ctx at ngl for each KV dtype, given current free VRAM/RAM.

    KV follows its layer: layers on GPU carry their KV to VRAM, the rest to RAM.
    ctx = min(GPU-side budget, RAM-side budget), capped by the model native limit.
    """
    info = _model_info(model)
    layers = info["layers"]
    native = info["native_ctx"]
    w_block = (info["size_gb"] or 0) * (1024**3) / layers
    out = {}
    for dtype, kb in info["kv_bytes_per_token"].items():
        kv_block = kb * 1024 / layers
        gpu_weights = ngl * w_block
        cpu_weights = (layers - ngl) * w_block
        vram_avail = max(0, free_vram - gpu_weights) if free_vram else 0
        ram_avail = max(0, free_ram - cpu_weights) if free_ram else 0
        gpu_kv = ngl * kv_block
        cpu_kv = (layers - ngl) * kv_block
        vram_ctx = max(0, vram_avail) / gpu_kv if gpu_kv > 0 else float("inf")
        ram_ctx = max(0, ram_avail) / cpu_kv if cpu_kv > 0 else float("inf")
        out[dtype] = int(min(vram_ctx, ram_ctx, native))
    return out


def _interp_speeds(model: str, ngl: int) -> tuple[dict, bool]:
    """Speeds for an arbitrary ngl: exact at measured points, linear between them,
    clamped past the last measured point. Returns (speeds, interpolated)."""
    pts = sorted(GPU_SPEED_SAMPLES[model].items())
    if ngl in GPU_SPEED_SAMPLES[model]:
        return dict(GPU_SPEED_SAMPLES[model][ngl]), False
    if ngl <= pts[0][0]:
        return dict(pts[0][1]), True
    if ngl >= pts[-1][0]:
        return dict(pts[-1][1]), True
    for (n0, s0), (n1, s1) in zip(pts, pts[1:]):
        if n0 <= ngl <= n1:
            f = (ngl - n0) / (n1 - n0)
            return {
                "prompt_tps": round(s0["prompt_tps"] + (s1["prompt_tps"] - s0["prompt_tps"]) * f, 2),
                "generation_tps": round(s0["generation_tps"] + (s1["generation_tps"] - s0["generation_tps"]) * f, 2),
            }, True
    raise AssertionError(f"ngl {ngl} not bracketed in {model} samples")


def _best_ngl(model: str, target_ctx: int, free_vram: int, free_ram: int, kv_dtype: str = "q8_0") -> dict:
    """Largest ngl whose ctx still fits target_ctx (fastest config that keeps the
    needed context). If none fits, falls back to the ngl maximizing ctx."""
    info = _model_info(model)
    fits = None
    for ngl in range(info["layers"], -1, -1):
        ctx = _ctx_max_split(model, ngl, free_vram, free_ram).get(kv_dtype, 0)
        if ctx >= target_ctx:
            fits = (ngl, ctx)
            break
    if fits:
        ngl, ctx = fits
        return {"ngl": ngl, "ctx_max": ctx, "fits": True}
    best = max(range(info["layers"] + 1),
               key=lambda n: _ctx_max_split(model, n, free_vram, free_ram).get(kv_dtype, 0))
    return {"ngl": best, "ctx_max": _ctx_max_split(model, best, free_vram, free_ram).get(kv_dtype, 0),
            "fits": False}


class LaunchRequest(BaseModel):
    model: str
    gpu_layers: int = 0
    ctx: Optional[int] = None
    threads: int = 4
    ctk: Optional[str] = None
    ctv: Optional[str] = None


class StopRequest(BaseModel):
    model: str
    secret: str


class Instance:
    def __init__(self, model, port, secret, proc, stdout_log, stderr_log, gpu_layers, ctx, threads, ctk=None, ctv=None):
        self.model = model
        self.port = port
        self.secret = secret
        self.proc = proc
        self.stdout_log = stdout_log
        self.stderr_log = stderr_log
        self.gpu_layers = gpu_layers
        self.ctx = ctx
        self.threads = threads
        self.ctk = ctk
        self.ctv = ctv
        self.status = "starting"
        self.started = time.time()
        self.last_active = time.time()

    def info(self):
        return {
            "model": self.model,
            "status": self.status,
            "port": self.port,
            "gpu_layers": self.gpu_layers,
            "ctx": self.ctx,
            "threads": self.threads,
            "cache_type_k": self.ctk,
            "cache_type_v": self.ctv,
            "started": datetime.fromtimestamp(self.started, tz=timezone.utc).isoformat(),
            "last_active": datetime.fromtimestamp(self.last_active, tz=timezone.utc).isoformat(),
            "idle_seconds_left": max(0, int(IDLE_TIMEOUT_MINUTES * 60 - (time.time() - self.last_active))),
            "stdout_log": str(self.stdout_log),
            "stderr_log": str(self.stderr_log),
            "pid": self.proc.pid,
        }


RUNNING: dict[str, Instance] = {}
_client = None


def get_lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def free_port():
    for port in range(LOW_PORT, HIGH_PORT + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("no free port in range")


def detect_ip():
    return get_lan_ip()


def build_cmd(name, port, gpu_layers, ctx, threads, ctk=None, ctv=None):
    spec = MODELS[name]
    binary = BIN_VULKAN if gpu_layers > 0 else BIN_CPU
    cmd = [
        binary,
        "-m", spec["path"],
        "-c", str(ctx),
        "-t", str(threads),
        "--host", "127.0.0.1",
        "--port", str(port),
    ]
    if spec.get("jinja"):
        cmd.append("--jinja")
    if gpu_layers > 0:
        cmd += ["--device", "Vulkan0", "-ngl", str(gpu_layers)]
    if ctk:
        cmd += ["-ctk", ctk]
    if ctv:
        cmd += ["-ctv", ctv]
    return cmd


async def wait_healthy(port, timeout=1800):
    url = f"http://127.0.0.1:{port}/health"
    start = time.time()
    async with httpx.AsyncClient(timeout=3) as c:
        while time.time() - start < timeout:
            try:
                r = await c.get(url)
                if r.status_code == 200 and r.json().get("status") == "ok":
                    return True
            except Exception:
                pass
            await asyncio.sleep(2)
    return False


async def stop_instance(inst: Instance):
    if inst.status != "running":
        return
    inst.status = "stopping"
    logger.info("stopping model %s (pid %s)", inst.model, inst.proc.pid)
    try:
        inst.proc.terminate()
    except Exception:
        pass
    try:
        await asyncio.wait_for(asyncio.to_thread(inst.proc.wait), timeout=15)
    except Exception:
        try:
            subprocess.run(["taskkill", "/PID", str(inst.proc.pid), "/T", "/F"],
                           capture_output=True, timeout=15)
        except Exception:
            pass
    inst.status = "stopped"


async def idle_monitor():
    while True:
        await asyncio.sleep(20)
        now = time.time()
        for name in list(RUNNING.keys()):
            inst = RUNNING[name]
            if inst.proc.poll() is not None and inst.status == "running":
                inst.status = "exited (crash)"
                logger.warning("model %s crashed", name)
            elif inst.status == "running" and (now - inst.last_active) > IDLE_TIMEOUT_MINUTES * 60:
                logger.info("model %s idle for %s min, stopping", name, IDLE_TIMEOUT_MINUTES)
                await stop_instance(inst)
                RUNNING.pop(name, None)


@asynccontextmanager
async def lifespan(app):
    global _client
    _client = httpx.AsyncClient(timeout=httpx.Timeout(1800.0, connect=10.0))
    monitor = asyncio.create_task(idle_monitor())
    logger.info("model manager started on port %s", API_PORT)
    yield
    monitor.cancel()
    for name in list(RUNNING.keys()):
        await stop_instance(RUNNING[name])
        RUNNING.pop(name, None)
    await _client.aclose()


app = FastAPI(title="Local Model Manager", lifespan=lifespan)


@app.get("/")
def root():
    return {
        "ip": detect_ip(),
        "api_port": API_PORT,
        "api_base": f"http://{detect_ip()}:{API_PORT}/v1",
        "idle_timeout_minutes": IDLE_TIMEOUT_MINUTES,
        "models": {name: _model_info(name) for name in MODELS},
        "launch": f"POST http://{detect_ip()}:{API_PORT}/launch  {{'model': 'ornith'}}",
    }


@app.get("/models")
def list_models():
    return {
        "running": {k: v.info() for k, v in RUNNING.items()},
        "available": {name: _model_info(name) for name in MODELS},
        "field_notes": FIELD_NOTES,
    }


@app.get("/resources")
def resources():
    ram = ram_info()
    vram = vram_devices()
    out = {"ram": ram, "vram": vram}
    if vram:
        best = max(vram, key=lambda d: d["total_bytes"])
        out["fits_full_gpu"] = fits_map(best["free_bytes"])
        out["field_notes"] = {
            "fits_full_gpu": "max context (tokens) so weights + KV of that dtype still fit current free VRAM",
        }
    if ram:
        out["fits_ram"] = fits_map(ram["free_bytes"])
        out.setdefault("field_notes", {})["fits_ram"] = (
            "max context (tokens) so weights + KV of that dtype still fit current free RAM (CPU-only load)"
        )
    return out


@app.get("/estimate")
def estimate(model: Optional[str] = None, prompt_tokens: Optional[int] = None,
             output_tokens: int = 1000, gpu_layers: Optional[int] = None,
             kv_dtype: str = "q8_0"):
    """RAM-only time estimates + static GPU-offload profiles (measured samples,
    linear interpolation, KV-split ctx math on live free memory). No live tests."""
    if model is not None and model not in MODELS:
        raise HTTPException(404, f"unknown model '{model}', available: {', '.join(MODELS)}")
    if prompt_tokens is not None and prompt_tokens < 0:
        raise HTTPException(422, "prompt_tokens must be >= 0")
    if output_tokens < 0:
        raise HTTPException(422, "output_tokens must be >= 0")
    valid_dtypes = sorted({d for info in MODELS.values() for d in info["kv_bytes_per_token"]})
    if kv_dtype not in valid_dtypes:
        raise HTTPException(422, f"invalid kv_dtype '{kv_dtype}', valid: {', '.join(valid_dtypes)}")

    base = {"mode": "ram_only", "basis": RAM_ONLY_BASIS}
    if model is None:
        base["speeds"] = RAM_ONLY_SPEEDS
        base["presets"] = {name: _est_presets(s) for name, s in RAM_ONLY_SPEEDS.items()}
        base["gpu_samples"] = {name: {str(n): dict(s) for n, s in sorted(pts.items())}
                               for name, pts in GPU_SPEED_SAMPLES.items()}
        base["field_notes"] = {
            "gen_1000_min": "time to generate 1000 output tokens",
            "pp_30000_min": "time to process a 30k-token prompt",
            "pp_60000_min": "time to process a 60k-token prompt",
            "turn_30000_1000_min": "process 30k prompt then generate 1k tokens",
            "turn_60000_1000_min": "process 60k prompt then generate 1k tokens",
            "gpu_samples": "benchmarked ngl (gpu_layers) points per model — speeds for other ngl values are linearly interpolated when you query that model",
        }
        return base

    speed = RAM_ONLY_SPEEDS.get(model) or GPU_SPEED_SAMPLES[model][0]
    base["model"] = model
    base["speeds"] = speed
    base["presets"] = _est_presets(speed)
    if prompt_tokens is not None:
        base["estimate"] = _est_calc(speed, prompt_tokens, output_tokens)

    info = _model_info(model)
    ram = ram_info()
    devs = vram_devices()
    best_vram = max(devs, key=lambda d: d["total_bytes"]) if devs else None
    free_ram = ram["free_bytes"] if ram else 0
    free_vram = best_vram["free_bytes"] if best_vram else 0

    partial = {
        "basis": ("measured ngl samples (llama-bench pp128/tg64, 4 threads, RX 580 Vulkan) + linear "
                  "interpolation; `ctx_max` splits each device's weight+KV budget by ngl "
                  "(a layer's KV lives on the same device as its layer) using live free VRAM/RAM"),
        "measured_points": {str(n): dict(s) for n, s in sorted(GPU_SPEED_SAMPLES[model].items())},
    }
    if best_vram:
        partial["ctx_current"] = {
            "ram_free_bytes": free_ram,
            "vram_free_bytes": free_vram,
            "ngl_blocks_limit": info["layers"],
        }
        partial["ctx_max_by_ngl"] = {
            str(n): _ctx_max_split(model, n, free_vram, free_ram)
            for n in sorted(GPU_SPEED_SAMPLES[model])
        }
    else:
        partial["basis"] += " — Vulkan device unavailable, so ctx_max/ctx limits are omitted"

    if gpu_layers is not None:
        if gpu_layers < 0 or gpu_layers > info["layers"]:
            raise HTTPException(422, f"gpu_layers must be 0..{info['layers']} for '{model}'")
        chosen = gpu_layers
        chosen_speeds, interp = _interp_speeds(model, chosen)
    else:
        target = (prompt_tokens or 0) + output_tokens if prompt_tokens is not None else info["native_ctx"]
        rec = _best_ngl(model, target, free_vram, free_ram, kv_dtype)
        chosen = rec["ngl"]
        chosen_speeds, interp = _interp_speeds(model, chosen)
        partial["recommended"] = {
            "ngl": rec["ngl"],
            "kv_dtype": kv_dtype,
            "target_ctx": target,
            "fits": rec["fits"],
            "ctx_max": rec["ctx_max"],
            "speeds": chosen_speeds,
            "interpolated": interp,
        }

    profile = {"ngl": chosen, "interpolated": interp, "speeds": chosen_speeds}
    if best_vram:
        profile["ctx_max"] = _ctx_max_split(model, chosen, free_vram, free_ram)
    if prompt_tokens is not None:
        profile["tokens"] = {"prompt_tokens": prompt_tokens, "output_tokens": output_tokens}
        profile["estimate"] = _est_calc(chosen_speeds, prompt_tokens, output_tokens)
    partial["profile"] = profile

    partial["params"] = {
        "gpu_layers": f"blocks offloaded to GPU, 0..{info['layers']}. Each GPU layer's KV must live in VRAM, so more offload = faster generation but a smaller max context. Pass it to pin an exact split; omit it to get the recommended largest fit.",
        "kv_dtype": f"KV-cache dtype used to size the context budget (this endpoint's default q8_0; valid: {', '.join(valid_dtypes)}). q8_0 is the llama.cpp sweet spot — ~2x smaller than f16, near-lossless.",
        "prompt_tokens": "input tokens to process. Batched, so this is where the GPU helps most; it mostly shapes the `recommended` ngl.",
        "output_tokens": "tokens to generate serially — each one passes through every layer on both devices, so this mostly sets your wait time.",
        "interpolated": "true when ngl isn't a benchmarked point — speeds are a linear guess between the two neighboring measured points (clamped beyond the last one).",
        "ctx_max": "max context tokens that fit both devices at that ngl for each KV dtype, from current free VRAM/RAM; capped by the model's native limit.",
        "ctx_max_by_ngl": "the same ctx_max evaluated at every benchmarked ngl, so you can see the offload-vs-context trade-off curve.",
        "recommended": "largest ngl whose ctx_max(kv_dtype) >= target_ctx (= prompt_tokens+output_tokens, or the model's full native ctx when you pass no tokens): the most GPU offload that still keeps the needed context. fits=false means even the best ngl can't hold it.",
        "estimate": "seconds/minutes for the requested tokens at that profile, using its (possibly interpolated) speeds.",
    }
    base["partial_gpu"] = partial
    return base


@app.post("/launch")
async def launch(req: LaunchRequest):
    if req.model not in MODELS:
        raise HTTPException(404, f"unknown model '{req.model}', available: {', '.join(MODELS)}")
    if req.ctk and req.ctk not in VALID_KV_TYPES:
        raise HTTPException(422, f"invalid ctk '{req.ctk}', valid: {', '.join(sorted(VALID_KV_TYPES))}")
    if req.ctv and req.ctv not in VALID_KV_TYPES:
        raise HTTPException(422, f"invalid ctv '{req.ctv}', valid: {', '.join(sorted(VALID_KV_TYPES))}")
    spec = MODELS[req.model]
    if req.ctx is not None:
        if req.ctx < 1:
            raise HTTPException(422, "ctx must be >= 1")
        if req.ctx > spec["native_ctx"]:
            raise HTTPException(422, f"ctx {req.ctx} exceeds '{req.model}' native limit {spec['native_ctx']}")
    if req.model in RUNNING:
        inst = RUNNING[req.model]
        return _launch_response(inst, already=True)
    port = free_port()
    ctx = req.ctx or spec["ctx"]
    secret = secrets.token_urlsafe(16)
    stdout_log = LOG_DIR / f"{req.model}-{port}.out.log"
    stderr_log = LOG_DIR / f"{req.model}-{port}.err.log"
    cmd = build_cmd(req.model, port, req.gpu_layers, ctx, req.threads, req.ctk, req.ctv)
    logger.info("launching %s on port %s (gpu_layers=%s, ctk=%s, ctv=%s) cmd=%s",
                req.model, port, req.gpu_layers, req.ctk, req.ctv, " ".join(cmd))
    stdout_f = open(stdout_log, "w", encoding="utf-8", errors="replace")
    stderr_f = open(stderr_log, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(cmd, stdout=stdout_f, stderr=stderr_f,
                            creationflags=CREATE_NO_WINDOW)
    inst = Instance(req.model, port, secret, proc, stdout_log, stderr_log,
                    req.gpu_layers, ctx, req.threads, req.ctk, req.ctv)
    RUNNING[req.model] = inst
    if await wait_healthy(port):
        inst.status = "running"
        logger.info("model %s ready on 127.0.0.1:%s, secret=%s", req.model, port, secret)
    else:
        inst.status = "failed-to-start"
    return _launch_response(inst)


def _launch_response(inst: Instance, already: bool = False):
    ip = detect_ip()
    return {
        "status": inst.status,
        "already_running": already,
        "model": inst.model,
        "port": inst.port,
        "ip": ip,
        "secret": inst.secret,
        "api_base": f"http://{ip}:{API_PORT}/v1",
        "openai_usage": {
            "base_url": f"http://{ip}:{API_PORT}/v1",
            "api_key": inst.secret,
            "model": inst.model,
        },
        "idle_timeout_minutes": IDLE_TIMEOUT_MINUTES,
        "cache_type_k": inst.ctk,
        "cache_type_v": inst.ctv,
        "stdout_log": str(inst.stdout_log),
        "stderr_log": str(inst.stderr_log),
    }


@app.post("/stop")
async def stop(req: StopRequest):
    inst = RUNNING.get(req.model)
    if not inst:
        raise HTTPException(404, f"model '{req.model}' is not running")
    if not secrets.compare_digest(inst.secret, req.secret):
        raise HTTPException(401, "secret mismatch")
    await stop_instance(inst)
    RUNNING.pop(req.model, None)
    return {"status": "stopped", "model": req.model, "port": inst.port}


async def _resolve_instance(model: str) -> Instance:
    inst = RUNNING.get(model)
    if not inst:
        raise HTTPException(404, f"model '{model}' is not running, launch it first: POST /launch")
    if inst.status != "running":
        raise HTTPException(503, f"model '{model}' status is {inst.status}")
    return inst


def _check_auth(inst: Instance, authorization: Optional[str]):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing 'Authorization: Bearer <secret>' header")
    token = authorization[len("Bearer "):].strip()
    if not secrets.compare_digest(inst.secret, token):
        raise HTTPException(401, "invalid secret")


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def proxy(path: str, request: Request, authorization: Optional[str] = Header(default=None)):
    model = request.query_params.get("model")
    if request.method == "POST":
        raw = await request.body()
        try:
            body = json.loads(raw) if raw else {}
        except Exception:
            body = {}
        model = model or body.get("model")
    if not model:
        raise HTTPException(400, "specify model via JSON body 'model' field or ?model= query param")

    if path == "models":
        inst = await _resolve_instance(model)
        _check_auth(inst, authorization)
        return {"object": "list", "data": [{"id": model, "object": "model", "owned_by": "local"}]}

    inst = await _resolve_instance(model)
    _check_auth(inst, authorization)
    inst.last_active = time.time()

    url = f"http://127.0.0.1:{inst.port}/v1/{path}"
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length", "connection", "accept-encoding")}
    stream = False
    if request.method == "POST" and body:
        stream = bool(body.get("stream", False))
    stream = stream or request.query_params.get("stream") in ("true", "1")
    stream = stream or "text/event-stream" in request.headers.get("accept", "")

    async def gen():
        async with _client.stream(request.method, url, content=raw, headers=headers, params=dict(request.query_params)) as resp:
            async for chunk in resp.aiter_raw():
                yield chunk

    if stream:
        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    resp = await _client.request(request.method, url, content=raw, headers=headers, params=dict(request.query_params))
    return Response(content=resp.content, status_code=resp.status_code,
                    media_type=resp.headers.get("content-type"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=API_HOST, port=API_PORT, log_level="info")