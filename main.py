import asyncio
import json
import logging
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

BIN_CPU = r"D:\llms\llama-b10809-bin-win-cpu-x64\llama-server.exe"
BIN_VULKAN = r"D:\llms\llama-b10809-bin-win-vulkan-x64\llama-server.exe"
BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "ornith": {
        "path": r"D:\llms\models\testing\ornith-ai_Ornith-1.5-9B-GGUF\Ornith-1.5-9B-Q4_K_M.gguf",
        "description": "Ornith-1.5-9B (qwen35), strong tool calling",
        "ctx": 4096,
        "jinja": True,
    },
    "gpt-oss": {
        "path": r"D:\llms\models\testing\unsloth_gpt-oss-20b-GGUF\gpt-oss-20b-Q4_K_M.gguf",
        "description": "OpenAI gpt-oss-20b MoE (20B total / 3.6B active)",
        "ctx": 2048,
        "jinja": True,
    },
    "qwen3-0.6b": {
        "path": r"D:\llms\models\testing\Qwen_Qwen3-0.6B-GGUF\Qwen3-0.6B-Q8_0.gguf",
        "description": "Tiny Qwen3-0.6B for quick sanity checks",
        "ctx": 4096,
        "jinja": True,
    },
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("model_manager")


class LaunchRequest(BaseModel):
    model: str
    gpu_layers: int = 0
    ctx: Optional[int] = None
    threads: int = 4


class StopRequest(BaseModel):
    model: str
    secret: str


class Instance:
    def __init__(self, model, port, secret, proc, stdout_log, stderr_log, gpu_layers, ctx, threads):
        self.model = model
        self.port = port
        self.secret = secret
        self.proc = proc
        self.stdout_log = stdout_log
        self.stderr_log = stderr_log
        self.gpu_layers = gpu_layers
        self.ctx = ctx
        self.threads = threads
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


def build_cmd(name, port, gpu_layers, ctx, threads, stdout, stderr):
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
        "models": MODELS,
        "launch": f"POST http://{detect_ip()}:{API_PORT}/launch  {{'model': 'ornith'}}",
    }


@app.get("/models")
def list_models():
    return {"running": {k: v.info() for k, v in RUNNING.items()}, "available": MODELS}


@app.post("/launch")
async def launch(req: LaunchRequest):
    if req.model not in MODELS:
        raise HTTPException(404, f"unknown model '{req.model}', available: {', '.join(MODELS)}")
    if req.model in RUNNING:
        inst = RUNNING[req.model]
        return _launch_response(inst, already=True)
    port = free_port()
    ctx = req.ctx or MODELS[req.model]["ctx"]
    secret = secrets.token_urlsafe(16)
    stdout_log = LOG_DIR / f"{req.model}-{port}.out.log"
    stderr_log = LOG_DIR / f"{req.model}-{port}.err.log"
    cmd = build_cmd(req.model, port, req.gpu_layers, ctx, req.threads, stdout_log, stderr_log)
    logger.info("launching %s on port %s (gpu_layers=%s) cmd=%s", req.model, port, req.gpu_layers, " ".join(cmd))
    stdout_f = open(stdout_log, "w", encoding="utf-8", errors="replace")
    stderr_f = open(stderr_log, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(cmd, stdout=stdout_f, stderr=stderr_f,
                            creationflags=CREATE_NO_WINDOW)
    inst = Instance(req.model, port, secret, proc, stdout_log, stderr_log,
                    req.gpu_layers, ctx, req.threads)
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