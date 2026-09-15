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
    stream = body.get("stream", False) if request.method == "POST" and body else False

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