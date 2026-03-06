# Tinfer — Tiny Inference Engine

Run LLMs locally with GPU acceleration. Built on [llama.cpp](https://github.com/ggml-org/llama.cpp). No cloud, no API keys, no C++ build tools.

## Install

```bash
pip install tinfer-ai
```

## Setup — Download the Inference Engine

After installing, run the setup command to download the optimal engine for your system:

```bash
tinfer-setup
```

This will automatically detect your **OS**, **CPU architecture**, and **GPU** (NVIDIA CUDA / Apple Metal / CPU), then download the correct pre-compiled engine.

```
[Tinfer] Analyzing your system...
[Tinfer] OS:   Windows (win)
[Tinfer] Arch: AMD64 (x64)
[Tinfer] GPU:  NVIDIA GeForce RTX 3060

[Tinfer] Selected Engine: win-x64-cuda
[Tinfer] Downloading: tinfer-v0.2.0-win-x64-cuda.zip
[████████████████████████████████████████] 100% (70.2/70.2 MB)

[Tinfer] ✓ Setup complete!
```

The engine is stored at `~/.tinfer/bin/` and will be automatically used by all tinfer commands.

> **Manual Download:** You can also download the engine directly from [GitHub Releases](https://github.com/Sunilk240/tinfer-ai/releases) and set the `TINFER_ENGINE_PATH` environment variable to its location.

---

## Quick Start

```bash
# 1. Download a model (GGUF format)
pip install huggingface-hub
huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF Llama-3.2-3B-Instruct-Q4_K_M.gguf --local-dir ./models

# 2. Run
tinfer -m ./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf -p "Hello, what is AI?" -n 100
```

---

## Three Ways to Use

### 1. CLI — Direct Chat

```bash
tinfer -m model.gguf -p "Explain quantum computing" -n 200
```

Key flags: `-m` model path, `-p` prompt, `-n` max tokens, `-ngl` GPU layers (auto-detected), `-c` context size, `--temp` temperature.

### 2. Server — WebUI + API

```bash
tinfer-server -m model.gguf --port 8080
# Open http://localhost:8080 for the built-in chat interface
```

Key flags: `--port`, `--host`, `-ngl`, `-c`, `--api-key`, `--embedding`, `--slots`.

### 3. HTTP API — OpenAI-Compatible

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100}'
```

Works with any OpenAI SDK — just change `base_url` to `http://localhost:8080/v1`.

---

## Supported Platforms

| Platform | GPU Acceleration | Engine |
|----------|-----------------|--------|
| Windows x64 | NVIDIA CUDA (All SM 50-90) | `tinfer-setup` auto-downloads |
| Windows x64 | CPU only | `tinfer-setup` auto-downloads |
| Linux x64 | NVIDIA CUDA (All SM 50-90) | `tinfer-setup` auto-downloads |
| Linux x64 | CPU only | `tinfer-setup` auto-downloads |
| Linux ARM64 | CPU only | `tinfer-setup` auto-downloads |
| macOS ARM64 | Apple Metal | `tinfer-setup` auto-downloads |
| macOS x64 | CPU only | `tinfer-setup` auto-downloads |

---

## Inference Types

| Type | Description |
|------|-------------|
| **Text Generation** | Standard LLM chat and completion |
| **Vision / Multimodal** | Image understanding, OCR, visual QA |
| **Embedding & Reranking** | Semantic search, text similarity |
| **LoRA Fine-Tuned** | Run fine-tuned adapters with `--lora` flag |

## Key Features

| Feature | Description |
|---------|-------------|
| **Layer Offloading** | Run models larger than VRAM — Disk → CPU → GPU |
| **PagedAttention** | Zero-fragmentation KV cache with Copy-on-Write |
| **KV Cache Eviction** | Infinite-length generation with smart eviction |
| **Model Conversion** | Convert HuggingFace models & LoRA adapters to GGUF |
| **Quantization** | 30+ types (Q4_K_M, Q5_K_M, Q8_0, IQ, etc.) |
| **Benchmarking** | Measure tokens/sec with `tinfer-bench` |

## Tools Included

| Command | Purpose |
|---------|---------|
| `tinfer` | CLI inference and chat |
| `tinfer-server` | HTTP server with WebUI |
| `tinfer-bench` | Performance benchmarking |
| `tinfer-quantize` | Model quantization |
| `tinfer-setup` | Download the inference engine for your system |

---

## Upgrade

```bash
pip install --upgrade tinfer-ai
tinfer-setup  # Re-download engine for the new version
```

## Uninstall

```bash
pip uninstall tinfer-ai
```

---

## Documentation

📖 **Full documentation:** [https://sunilk240.github.io/tinfer-documentation/](https://sunilk240.github.io/tinfer-documentation/)

---

## License

MIT
