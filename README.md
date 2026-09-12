# Local Model Manager

FastAPI wrapper around [llama.cpp's `llama-server`](https://github.com/ggml-org/llama.cpp) that launches, proxies, and auto-stops local GGUF models on demand. Exposes an OpenAI-compatible HTTP API with tool-calling support.

Run CPU or GPU (Vulkan) instances with a single POST, never leave a model running longer than a configurable idle window, and talk to every model through the same API key and endpoint.

## Features

- **Launch on demand** — `POST /launch` boots a model and prints a per-instance secret token.
- **Idempotent launches** — re-POSTing `/launch` for an already-running model returns the same instance and secret.
- **OpenAI-compatible proxy** — `/v1/*` routes to the model; works with any OpenAI client, including `chat/completions` with streaming (`stream: true`, SSE) and `tool_calls`.
- **Auto-stop** — models idle for 60 minutes are shut down automatically; all models stop when the manager shuts down.
- **Secret auth** — every proxied request needs `Authorization: Bearer <secret>`. Non-secret endpoints (`/models`, `/launch`) reveal no secrets.
- **CPU or GPU** — `gpu_layers > 0` selects the Vulkan build (`--device Vulkan0 -ngl N`); `0` uses the CPU build.

## Requirements

- Python 3.10+ with FastAPI, uvicorn, httpx, pydantic
- A `llama-server` binary for CPU and, optionally, one for Vulkan
- GGUF model files registered in `MODELS`

## Setup

1. Edit `main.py`:
   - `BIN_CPU` / `BIN_VULKAN` (optional) — paths to `llama-server.exe` for each backend.
   - `MODELS` — register your GGUF files, e.g.:

```python
MODELS = {
    "ornith": {
        "path": r"D:\models\Ornith-1.5-9B-Q4_K_M.gguf",
        "description": "Ornith-1.5-9B (qwen35), strong tool calling",
        "ctx": 4096,
        "jinja": True,
    },
}
```

2. Install dependencies:

```bash
pip install fastapi uvicorn httpx pydantic
```

3. Run:

```bash
uvicorn main:app --host 0.0.0.0 --port 8050
```

## Endpoints

| Endpoint                    | Auth | Description                                                        |
| --------------------------- | ---- | ------------------------------------------------------------------ |
| `GET /`                     | no   | Server info, LAN IP, registered models.                            |
| `GET /models`               | no   | Running instances (port, pid, idle countdown, no secrets) + registry. |
| `POST /launch`              | no   | Start a model. Body: `{"model": "ornith", "gpu_layers": 0, "ctx": 4096, "threads": 4}`. Idempotent. |
| `POST /stop`                | yes  | `{"model": "ornith", "secret": "<secret>"}`.                       |
| `GET|POST /v1/{path}`       | yes  | Proxy to the model. Pass the model name via `?model=` query param or JSON body. Supports streaming. |

## Usage

```bash
# launch (returns port, LAN IP, and secret)
curl -X POST http://127.0.0.1:8050/launch \
  -H "Content-Type: application/json" \
  -d '{"model":"ornith","gpu_layers":99}'

# chat completion with tool calling
curl http://127.0.0.1:8050/v1/chat/completions \
  -H "Authorization: Bearer <secret>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ornith",
    "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }]
  }'

# stop
curl -X POST http://127.0.0.1:8050/stop \
  -H "Content-Type: application/json" \
  -d '{"model":"ornith","secret":"<secret>"}'
```

With an OpenAI client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://10.8.1.4:8050/v1",
    api_key="<secret>",
)
resp = client.chat.completions.create(
    model="ornith",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

## Notes

- Model ports are taken from a free port in `8100–8900`.
- Logs go to `logs/<model>-<port>.{out,err}.log`.
- The idle monitor checks every 20 seconds (see `idle_monitor`).
- Secrets live in memory only and die with the manager process.
- Tested with an AMD RX 580 (Vulkan) on Windows; the Vulkan build is picked automatically when `gpu_layers > 0`.