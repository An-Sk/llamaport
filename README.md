# Local Model Manager

A tiny `llama.cpp` launcher for people who run LLMs on one machine and do the actual work on another.

You point it at a GGUF model, it boots `llama-server.exe` on demand, hands you an **OpenAI-compatible endpoint**, and then — when you go quiet — it shuts the model back down. No LLM sitting in your RAM/VRAM 24/7. No juggling terminals.

## The core idea

The whole point of this thing is **offloading**:

- Models are packed away the moment they're not needed.
- The moment the server gets pinged, the model is loaded back.
- One hour of silence and the server stops it on its own.

That's the primary use case. Everything else is a bonus you get for free.

## Who is this for?

Mainly **coding agents**.

The typical setup is two machines:

1. The **server machine** — whose whole job is to host LLMs. It's headless, always on, runs this script. Everything here is built for that machine.
2. Your **work machine** — where the agent (or you) actually does the coding, talks to the server over the network, and treats it as a normal OpenAI endpoint.

## The network side

You need the two machines to reach each other. How you set that up is up to you:

- **Local subnet / port forwarding** — the classic way. I'm not going to comment on configuring port forwarding for a local sub-network; plenty of guides exist and it depends on your router.
- **Remote access** — the author's simplest setup is [Tailscale](https://tailscale.com): install it on *both* machines and log in to the same account. Both machines end up "in the same network", so reaching each other is relatively easy — no port forwarding, no public IP.

Either way, once the machines can reach each other, everything else is identical.

## How it works

Three small pieces:

### 1. The LLM launcher (the main part)

- The server machine has the `llama.cpp` binaries on it (the **CPU build** for simplicity — no GPU drivers, no Vulkan, just an `.exe`).
- That's all. The FastAPI server doesn't run inference itself. When you touch an endpoint, it launches `llama-server.exe` with the right flags via a subprocess and proxies your requests to it.
- Launches are **idempotent**: ask for a model that's already up and you get the same instance back — same port, same secret.
- Each launch gets a **secret token**; without `Authorization: Bearer <secret>` the proxy refuses you.
- A monitor thread watches for **1 hour of inactivity** and stops the model, and everything is shut down when the server itself goes down.
- If you happen to have a Vulkan build and want to offload to a GPU, `gpu_layers > 0` picks it up automatically. Otherwise: CPU.

### 2. The Python side

Boring on purpose. Just FastAPI + a few standard dependencies. No model code, no ML stack, nothing to break.

### 3. Models downloading

Handled outside this server. Hugging Face covers both bits — **model selection** and **model downloading** — they even ship scripts for it (e.g. `huggingface-cli download` or `snapshot_download`). Grab a GGUF you like, drop it in a folder, register the path in `MODELS`.

Or just ask a coding agent to help you set this part up. That's literally a two-liner.

## Usage

Practically, the flow looks like this:

1. **Ask your coding agent to create a tiny "help script"** that talks to this server — it POSTs `/launch`, waits for "running", and hands you back an OpenAI-compatible endpoint: `base_url`, `api_key`, `model`.
2. **Use that endpoint anywhere you'd use OpenAI.** The point of the proxy is that it's drop-in: `ChatOpenAI` in LangChain, the `openai` Python package, whatever your agent uses.

```bash
# launch a model (returns port, LAN IP, and secret)
curl -X POST http://127.0.0.1:8050/launch \
  -H "Content-Type: application/json" \
  -d '{"model":"ornith","gpu_layers":0}'
```

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://10.8.1.4:8050/v1",   # whatever the server told you
    api_key="<secret>",                    # from the /launch response
)
resp = client.chat.completions.create(
    model="ornith",
    messages=[{"role": "user", "content": "Hello, can you help me debug this function?"}],
)
```

## Endpoints

| Endpoint               | Auth | What it does |
| ---------------------- | ---- | ------------ |
| `GET /`                | no   | Server info: LAN IP, port, registered models. |
| `GET /models`          | no   | Running instances (port, pid, idle countdown — no secrets) + model registry. |
| `POST /launch`         | no   | Start a model. Body: `{"model": "ornith", "gpu_layers": 0, "ctx": 4096, "threads": 4}`. Idempotent. |
| `POST /stop`           | yes  | `{"model": "ornith", "secret": "<secret>"}`. |
| `GET\|POST /v1/{path}` | yes  | OpenAI-compatible proxy. Pass the model via `?model=` query param or JSON body. Streaming supported. |

## Setup

1. Register your models in `MODELS` in `main.py`:

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

2. Install the dependencies (Python side is just):

```bash
pip install fastapi uvicorn httpx pydantic
```

3. Run it:

```bash
uvicorn main:app --host 0.0.0.0 --port 8050
```

## Notes

- Model ports are picked automatically from `8100–8900`.
- Logs go to `logs/<model>-<port>.{out,err}.log`.
- Secrets live in memory only and die with the manager process.
- Tested on Windows with an AMD RX 580 + CPU builds of llama.cpp b10809.