# Local CUDA build of llama.cpp — for use with pi

This document describes the local, custom-built `llama.cpp` setup on this machine
for running the **gemma-4-12b-it-qat-q4_0** GGUF model on the NVIDIA GPU and
serving it to the [`pi`](../..) coding agent over an OpenAI-compatible HTTP API.

The upstream `README.md` covers the project itself. This file covers *this specific
build*: the why, the how, and the things you need to know to keep it working.

---

## TL;DR

```bash
# Manual start
~/bin/llama-cuda-start

# Manual stop
~/bin/llama-cuda-stop

# systemd-managed (auto-starts on graphical login)
systemctl --user status llama-cuda.service
systemctl --user start llama-cuda.service
systemctl --user stop llama-cuda.service

# Quick reference
~/bin/llama-cuda-help
```

The server listens on `http://127.0.0.1:8080` and is wired into `pi` as the
`llama.cpp` provider.

---

## Why a custom build?

The original `~/local/llama.cpp/` build was compiled **without GPU support**
(only `libggml-cpu.so`). The `--n-gpu-layers 999` flag was silently ignored,
so the model ran on CPU at ~4.7 tok/s, which made even simple `pi` prompts
feel sluggish.

We tried three backends for the local 12B model on the GTX 1660 SUPER (6 GB):

| Backend | Result | Notes |
|---|---|---|
| CPU only (original) | 4.7 tok/s | Baseline. The 12B dense model fits in RAM, just slow. |
| Vulkan | **Failed** | NVIDIA driver on this card cannot allocate a single 1 GB tensor even with 5.6 GB free VRAM. |
| CUDA 12.4 | **7.5 tok/s** | Works, but only 25/48 layers fit on the 6 GB card. |

The CUDA build lives in `/tmp/llama-cuda-build/` and is wired into a
systemd user service so it starts automatically on graphical login.

---

## Hardware constraints

| Component | Spec |
|---|---|
| CPU | Intel i5-11400F (6C/12T, AVX-512) |
| RAM | 32 GB |
| GPU | NVIDIA GeForce GTX 1660 SUPER, **6 GB GDDR6** |
| Driver | 550.163.01 (CUDA 12.4) |
| Vulkan ICD | 1.4.309.0 (working but useless for this model) |
| Storage | NVMe SSD, plenty free |

The 12B model is **6.5 GB on disk** in Q4_0 quant. It cannot fully fit on the
6 GB GPU, so we offload as many layers as will fit plus the embeddings.

### Why 25 layers, not 35?

| Config | VRAM used | Works? | Notes |
|---|---|---|---|
| 35 layers, `-c 1024` | ~5.74 GB | yes | 11 tk/s, but context too small for pi's 4681-token system prompt |
| 30 layers, `-c 4096` | ~5.82 GB | yes | 8 tk/s, still too small for pi |
| 30 layers, `-c 6144` | OOM | no | KV cache growth exceeds budget |
| **25 layers, `-c 6144`** | **~5.20 GB** | **yes** | **7.5 tk/s, fits pi's prompts** |
| 25 layers, `-c 8192` | OOM | no | Same issue |

The 23 CPU layers plus the 6144-token KV cache use the remaining headroom
on a 6 GB card. Push the context higher and the KV cache overflows VRAM.

---

## Build details

### Source

- Repo: `/home/joachim/lab/prj/llama.cpp` (commit `0ef6f06`, the same upstream
  commit the stock `~/local/llama.cpp/` binary was built from).

### Build directory

- Out-of-source build: `/tmp/llama-cuda-build/`
- Why `/tmp`? The 1.4 GB build fits comfortably there, and keeping the
  build separate from the upstream source tree means a `git pull` won't
  clobber it. It survives reboots because `/tmp` is tmpfs on this system
  with 32 GB RAM. **If `/tmp` gets cleared, you'll need to rebuild.**

### CMake invocation

```bash
cd /tmp/llama-cuda-build
cmake /home/joachim/lab/prj/llama.cpp \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_VULKAN=OFF \
  -DGGML_OPENMP=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DGGML_NATIVE=OFF \
  -DGGML_CPU_ALL_VARIANTS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES="75" \
  -DCMAKE_INSTALL_PREFIX=/tmp/llama-cuda-install

cmake --build . --config Release -j$(nproc) --target llama-server
```

Key flags:

- `-DGGML_CUDA=ON` — enables CUDA backend
- `-DCMAKE_CUDA_ARCHITECTURES="75"` — Turing (GTX 16-series, RTX 20-series).
  Wider lists (`"75;86;89"`) work but compile more code.
- `-DGGML_CPU_ALL_VARIANTS=OFF` — disable every x86 micro-arch variant
  of the CPU backend. We only need `-msse4.2` for this CPU, the
  defaults are fine.
- `-DGGML_NATIVE=OFF` — don't compile for this exact CPU. The
  resulting binary is portable across AVX2-capable CPUs.
- `-DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF` — smaller, faster build.

Build time: ~8 minutes on the i5-11400F, 12 threads.

### Runtime dependency

The build links against the **CUDA 12.4 toolkit** installed via
`sudo apt install nvidia-cuda-toolkit` (provides `libcudart`, `libcublas`,
`libcublasLt`, `nvcc`, and headers). Without this, the build fails at
configure time.

---

## Running it

### Manual: `~/bin/llama-cuda-start`

Wrapper that:

1. Verifies the binary and model exist.
2. **Clears `LD_LIBRARY_PATH` and re-exports it** to point at
   `/tmp/llama-cuda-build/bin`. The user's login shell has
   `LD_LIBRARY_PATH=/home/joachim/local/llama.cpp/lib` set globally, which
   makes the new binary find the old (CPU-only) `libggml.so` and break.
3. Launches `llama-server` with `nohup setsid` so it survives the parent shell.
4. Waits for the health endpoint to respond (max 30 s).
5. Writes a PID file to `/tmp/llama-cuda-server.pid`.

### Manual: `~/bin/llama-cuda-stop`

Reads the PID file, sends SIGTERM, waits 10 s, escalates to SIGKILL.
Falls back to `pkill -f /tmp/llama-cuda-build/bin/llama-server` if the
PID file is gone. **Does not touch the systemd service** — only kills
the manual launch.

### systemd user service: `~/.config/systemd/user/llama-cuda.service`

Auto-starts on graphical login. Linger is enabled via
`loginctl enable-linger $USER` so the service survives logout.

Key bits:

```ini
[Service]
Environment=LD_LIBRARY_PATH=/tmp/llama-cuda-build/bin
Environment=HOME=/home/joachim
Environment=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin
WorkingDirectory=/tmp/llama-cuda-build/bin
ExecStart=/tmp/llama-cuda-build/bin/llama-server \
  -m /home/joachim/models/gguf/gemma-4-12b-it-qat-q4_0.gguf \
  -c 6144 -t 8 \
  --host 127.0.0.1 --port 8080 \
  --n-gpu-layers 25
Restart=on-failure
RestartSec=5
```

Managing it:

```bash
systemctl --user daemon-reload         # after editing the unit file
systemctl --user enable --now llama-cuda.service
systemctl --user status llama-cuda.service
journalctl --user -u llama-cuda.service -f
```

The service runs as the user (no root), listens only on `127.0.0.1`
(no network exposure), and uses `Restart=on-failure` so a crash brings
it back automatically.

### Important: the `LD_LIBRARY_PATH` gotcha

`systemd` starts the service in a clean environment, so setting
`Environment=LD_LIBRARY_PATH=/tmp/llama-cuda-build/bin` in the unit
file is sufficient. The manual `~/bin/llama-cuda-start` script, however,
runs in the user's login shell which has `LD_LIBRARY_PATH` already set
to the **old** CPU-only build. The script does `unset` then re-exports
to work around this. **Don't** try to run the binary directly without
this dance — it will appear to start but then report
`ggml_backend_cuda_buffer_type_alloc_buffer: cudaMalloc failed: out of memory`
because the wrong `libggml.so` is being loaded.

---

## pi integration

Provider entry in `~/.pi/agent/models.json`:

```json
"llama.cpp": {
  "baseUrl": "http://127.0.0.1:8080/v1",
  "api": "openai-completions",
  "apiKey": "none",
  "compat": {
    "supportsDeveloperRole": false,
    "supportsReasoningEffort": false
  },
  "models": [
    {
      "id": "gemma-4-12b-it-qat-q4_0",
      "name": "Gemma 4 12B IT QAT Q4_0 (llama.cpp, CUDA, 25/48 GPU layers)",
      "reasoning": false,
      "input": ["text"],
      "contextWindow": 6144,
      "maxTokens": 2048,
      "compat": {
        "thinkingFormat": "qwen-chat-template"
      }
    }
  ]
}
```

### The "forever" fix (two parts)

**1. `reasoning: false`**
Tells pi not to render or request reasoning deltas. Without this, the
server emits `reasoning_content` for every token and pi waits for the
full 16k-token thinking trace before showing anything.

**2. `compat.thinkingFormat: "qwen-chat-template"`**
Tells pi to send `chat_template_kwargs: {"enable_thinking": false}` on
every request. The gemma-4 chat template embedded in the GGUF defaults
to inserting an empty `<|channel>thought\n<channel|>` block at the
start of the model's turn, which forces it into thinking mode. This
flag bypasses that.

Without part 2, the model goes straight into a long thinking loop and
emits `reasoning_content` for thousands of tokens — at 7.5 tok/s that's
many minutes of "working forever" before any answer appears.

### Known gotcha: pi's prompt size

`pi` sends a ~4681-token system prompt (instructions + tool definitions).
The model needs `contextWindow >= 4681` to accept the request. Setting
`contextWindow: 6144` gives ~1400 tokens of headroom for user content
and the model's response. If pi grows its toolset and exceeds 6144
tokens, you'll see `400 request (N tokens) exceeds the available context
size (6144 tokens)` errors.

---

## Performance

Measured on this hardware, prompt "Count from 1 to 30, one number per line.", 60 output tokens:

| Configuration | tk/s | Notes |
|---|---|---|
| `~/local/llama.cpp` CPU only | 4.68 | Baseline. Same 12B model, no GPU. |
| This build, 25 GPU layers | **7.5** | 1.6× speedup. |
| Ollama, 12B dense, full GPU | 11.22 | Ollama has more aggressive GPU offload + optimizations. |
| Ollama, 26B A4B QAT, 8k ctx | 10.5 | The preferred default for pi; higher quality, similar speed to 12B. |
| Ollama, 26B MoE, 4k ctx (early test) | 16.11 | Only ~3-4B params active per token (8/128 experts). |

The gap between us (7.5) and Ollama's 12B (11.2) is mostly because we
have 23 layers on CPU. The model architecture bottleneck is
feed-forward compute on the CPU layers.

For non-interactive pi sessions (file edits, code review, etc.) 7.5
tok/s is plenty — pi does tool calls, waits for a chunky response, and
the wall-clock time is dominated by the model's reasoning, not its
generation rate.

The **26B A4B QAT** model in Ollama (`gemma4-26b-a4b-8k`) is the
preferred default for pi despite being a "bigger" model: it has 25.8B
total params but only ~3-4B active per token (8 of 128 MoE experts),
so per-token cost is comparable to 12B-dense, while quality is higher.
It also gets full GPU offload via Ollama, hitting ~10.5 tk/s.

---

## Coexistence with Ollama

Both stacks want the same 6 GB GPU. They use different ports
(llama.cpp → 8080, Ollama → 11434) but share VRAM.

When llama.cpp is running: **5.2 GB used, ~0.8 GB free**. Ollama needs
~2.5–3 GB to load any model, so it can't load anything while we're up.

Workflow:

```bash
# Switch to Ollama (26B A4B QAT, or any other ollama model)
systemctl --user stop llama-cuda.service
pi --provider ollama --model gemma4-26b-a4b-8k

# Switch back to llama.cpp (12B dense)
systemctl --user start llama-cuda.service
pi --provider llama.cpp --model gemma-4-12b-it-qat-q4_0
```

If `pi` tries to talk to Ollama while llama-cuda is up, you'll get
Ollama errors like `model requires more VRAM` or it'll just be slow.
Stop llama-cuda first.

### Other Ollama models on the box

In addition to the 26B A4B QAT, the box has these Ollama models wired
into `~/.pi/agent/models.json`:

- `gemma4-12b-local` — same 12B Q4_0 model as the llama.cpp server,
  but served by Ollama with full GPU offload (faster: 11.2 vs 7.5
  tk/s). Useful for a quick A/B against the llama.cpp build.
- `gemma4-26b-a4b-8k` — the 26B A4B QAT Q4_0, 8k context. The
  preferred default.
- `gemma4:e4b`, `phi4:latest`, `qwen2.5-coder:14b`, etc. — pulled
  earlier, available for comparison.

Each Ollama model can use its own per-model `num_ctx` (set in its
Modelfile). The 26B A4B's default 4096 context was too small for pi's
system prompt, so we created the `-8k` variant with `PARAMETER num_ctx
8192` in its Modelfile.

---

## File inventory

| Path | Purpose |
|---|---|
| `/home/joachim/lab/prj/llama.cpp/` | Upstream source (commit `0ef6f06`) |
| `/home/joachim/lab/prj/llama.cpp/PI_CUDA_BUILD.md` | This file. Branch: `local/cuda-pi-setup` |
| `/tmp/llama-cuda-build/` | Out-of-source build directory (1.4 GB) |
| `/tmp/llama-cuda-build/bin/llama-server` | The compiled binary |
| `/tmp/llama-cuda-build/bin/libggml-cuda.so` | CUDA backend (loaded at runtime) |
| `/home/joachim/models/gguf/gemma-4-12b-it-qat-q4_0.gguf` | The 12B model (6.5 GB) |
| `~/bin/llama-cuda-start` | Manual launch script |
| `~/bin/llama-cuda-stop` | Manual stop script |
| `~/bin/llama-cuda-help` | Quick reference card |
| `~/.config/systemd/user/llama-cuda.service` | Systemd unit file |
| `/tmp/llama-cuda-server.log` | Manual launch log |
| `/tmp/llama-cuda-server.pid` | Manual launch PID file |
| `~/.pi/agent/models.json` | pi provider configuration |
| `~/.pi/agent/settings.json` | pi default model + enabled model list |
| `/tmp/gemma4-12b-local.Modelfile` | Modelfile that imports the 12B GGUF into Ollama as `gemma4-12b-local` |
| `/tmp/gemma4-26b-8k.Modelfile` | Modelfile that derives `gemma4-26b-a4b-8k` from the 4096-ctx ollama model with `PARAMETER num_ctx 8192` |

---

## Rebuilding

If you change the source, update llama.cpp, or want to try different
CMake flags:

```bash
# Stop the service
systemctl --user stop llama-cuda.service
~/bin/llama-cuda-stop

# Wipe and rebuild
rm -rf /tmp/llama-cuda-build
mkdir -p /tmp/llama-cuda-build
cd /tmp/llama-cuda-build
cmake /home/joachim/lab/prj/llama.cpp <flags>     # see Build details above
cmake --build . --config Release -j$(nproc) --target llama-server

# Restart the service
systemctl --user daemon-reload
systemctl --user start llama-cuda.service
```

Common reasons to rebuild:

- `git pull` in `/home/joachim/lab/prj/llama.cpp/` brings a new commit.
- A new GGUF model needs different kernels (e.g., a MoE).
- You upgrade the CUDA toolkit and want to match.
- `/tmp` was cleared and the build directory is gone.

---

## Troubleshooting

### Server says "model loaded" but every request returns empty content

The model is going into thinking mode. Verify `reasoning: false` and
`thinkingFormat: "qwen-chat-template"` are in `~/.pi/agent/models.json`.

### `cudaMalloc failed: out of memory` at startup

- Check `nvidia-smi` — something else is using VRAM (X11 + cinnamon
  always take ~400 MB; ollama is the usual culprit).
- Try fewer layers: edit `~/.config/systemd/user/llama-cuda.service`
  and lower `--n-gpu-layers` from 25.
- Try smaller context: lower `-c 6144` to `-c 4096`.

### `n-gpu-layers option will be ignored, llama.cpp was compiled without GPU support`

You're running the wrong binary. The new build is in
`/tmp/llama-cuda-build/bin/llama-server`, not
`/home/joachim/local/llama.cpp/bin/llama-server`. The
`LD_LIBRARY_PATH` issue described above is the most common cause of
this — the new binary loads the old `libggml.so` and reports the
limitation of *that* library.

### pi fails with `400 request (N tokens) exceeds the available context size (M tokens)`

pi's prompt doesn't fit in the model's context. Either:

- Increase `-c` in the service file (if VRAM allows).
- Reduce the number of tools or system-prompt length in pi.
- Accept that this model can only do short-form tasks for pi.

### Vulkan OOM even though there's plenty of free VRAM

Known limitation of NVIDIA's Vulkan driver on the 1660 SUPER with this
GGUF layout. Use CUDA instead. The math:

- `vulkaninfo` reports 5.26 GiB free on the device-local heap.
- The model needs to allocate a single 1.07 GB tensor for the
  embedding.
- Vulkan tries to allocate this from the device-local heap. It fails
  with `ErrorOutOfDeviceMemory` even though 5.26 GiB is theoretically
  available — the failure is consistent and reproducible.

The likely cause is that NVIDIA's driver accounts for some additional
reservation (display surfaces, command buffers, etc.) that Vulkan
itself doesn't see. CUDA's allocator handles this fine; Vulkan's
doesn't.

### Build fails: "Could NOT find Vulkan" or "SPIRV-Headers missing"

You have the Vulkan backend enabled. Disable it:

```bash
cmake ... -DGGML_VULKAN=OFF ...
```

(That's already what the recommended cmake invocation above does.)

### Build fails: "Could not find a package configuration file provided by CUDA"

You don't have the CUDA toolkit installed:

```bash
sudo apt install nvidia-cuda-toolkit
```

This is ~2 GB. nvidia-cuda-dev and libcudart12 are also needed.

---

## References

- llama.cpp upstream: https://github.com/ggml-org/llama.cpp
- CUDA toolkit: https://developer.nvidia.com/cuda-toolkit
- This build's commit: `0ef6f06`
- pi (the consumer): https://github.com/earendil-works/pi
- GGUF spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
- Gemma chat template docs: embedded in the model's
  `tokenizer.chat_template` metadata (visible via
  `curl -s http://127.0.0.1:8080/props | jq .chat_template`)
