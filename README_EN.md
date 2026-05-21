# Qwen3.6-27B-FP8 Docker Inference Deployment

[中文文档](README.md)

> Dual-service vLLM inference deployment | 2-4x GPU (24GB+, sm_86) | Pre-built images + source build
> Image: [`maricgeer/vllm-qwen36`](https://crpi-3jsqnspnt5spjb2h.ap-southeast-1.personal.cr.aliyuncs.com/maricgeer/vllm-qwen36) | Based on [vLLM](https://github.com/vllm-project/vllm) v0.21.0

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 2x RTX 3090 (24GB) | 4x RTX 3090 |
| GPU Arch | sm_86 (Ampere) | sm_86 |
| VRAM per GPU | ~22.7 GB | ~22.7 GB |
| Disk | 50 GB free | 100 GB free |
| CPU | 8+ cores | 24+ cores |
| Model files | ~27 GB | ~27 GB |

**GPU Architecture**: Pre-built images only support **sm_86** (RTX 3090 / A6000 / A40). Other GPUs require building from source.

## Quick Start

### Option 1: Pre-built Image (Recommended for sm_86, ~30 min)

```bash
# 1. Clone repo (no submodule needed, config files only)
git clone https://github.com/GeerMrc/docker-vllm-public.git && cd docker-vllm-public

# 2. Pull pre-built image (~6.3 GB compressed)
docker pull crpi-3jsqnspnt5spjb2h.ap-southeast-1.personal.cr.aliyuncs.com/maricgeer/vllm-qwen36:sm86-v0.21.0

# 3. Download model (~27GB)
huggingface-cli download Qwen/Qwen3.6-27B-FP8 --local-dir /path/to/Qwen3.6-27B-FP8

# 4. Configure
cp .env.example .env
# Edit .env - must change:
#   DOCKER_IMAGE=crpi-3jsqnspnt5spjb2h.ap-southeast-1.personal.cr.aliyuncs.com/maricgeer/vllm-qwen36:sm86-v0.21.0
#   VLLM_MODEL_PATH=/path/to/Qwen3.6-27B-FP8
#   VLLM_PRIMARY_GPU_IDS=0,1

# 5. Start
sudo ./manage.sh start
```

### Option 2: Build from Source (All GPU Architectures, ~60-90 min)

```bash
git clone https://github.com/GeerMrc/docker-vllm-public.git && cd docker-vllm-public
git submodule update --init              # Pull vLLM source (~201MB)
./manage.sh prepare-src                  # Build dependencies (~422MB)
./manage.sh build                        # Build (~45 min, auto-detects CPU cores)
# For other GPU archs: set VLLM_BUILD_CUDA_ARCH=8.9 in .env
# China users: set VLLM_USE_CHINA_MIRROR=true in .env
```

## Model Download

The Qwen3.6-27B-FP8 model (~27GB) must be downloaded separately:

```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3.6-27B-FP8 --local-dir /path/to/Qwen3.6-27B-FP8

# Faster with hf-transfer:
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download Qwen/Qwen3.6-27B-FP8 --local-dir /path/to/Qwen3.6-27B-FP8
```

Set `VLLM_MODEL_PATH` in `.env` to the download directory.

## Building for Other GPUs

```bash
# Check your GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Common architectures:
# RTX 3090 / A6000 / A40  →  8.6  (pre-built image available)
# RTX 4090 / L40S         →  8.9  (build from source)
# A100                    →  8.0  (build from source)
# H100                    →  9.0  (build from source)

# Build with custom architecture (set in .env)
VLLM_BUILD_CUDA_ARCH=8.9 ./manage.sh build
```

> CUDA kernels cannot be cross-compiled. Each architecture must be built on matching hardware.

## Architecture

Primary (GPU 0,1, port 8089) + Secondary (GPU 2,3, port 8099), sharing the same Docker image and model files (read-only mount). You can also run a single service with 2x GPU.

## Profile Reference

| Profile | Thinking | Temp | top_p | Context | Decode | Use Case |
|---------|----------|------|-------|---------|--------|----------|
| agent-fast | OFF | 0.7 | 0.80 | 140K | 37.6 tok/s | Low-latency tool calling |
| agent-thinking | ON | 0.6 | 0.95 | 140K | 37.6 tok/s | Multi-turn reasoning |
| code | ON | 0.6 | 0.95 | 140K | 37.6 tok/s | Code generation |
| instruct | OFF | 0.7 | 0.80 | 128K | ~48 tok/s | Quick Q&A |
| think | ON | 1.0 | 0.95 | 256K | ~48 tok/s | General thinking |

## Management Commands

```bash
./manage.sh start [primary|secondary]   # Start service(s)
./manage.sh stop [primary|secondary|all] # Stop service(s)
./manage.sh restart [primary|secondary|all] # Restart
./manage.sh set-profile primary <name>  # Switch profile
./manage.sh config                      # View configuration
./manage.sh status                      # View running status
./manage.sh logs [primary|secondary]    # View Docker logs
./manage.sh build                       # Build Docker image
```

## China Users

For faster downloads during build, set in `.env`:
```bash
VLLM_USE_CHINA_MIRROR=true
VLLM_PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
```

## Performance

| Metric | Per Service (2x GPU) |
|--------|---------------------|
| Decode (bf16+MTP) | 37.6 tok/s @c=1 |
| Throughput (c=4) | 203 tok/s |
| VRAM per GPU | ~22.7 GB |
| Max context | 140K (agent/code), 256K (think) |

## Documentation

| Document | Content |
|----------|---------|
| [BUILD_INFO](BUILD_INFO) | Image version tracking |
| [build_guide.md](docs/build_guide.md) | Build instructions |
| [migration_guide.md](docs/migration_guide.md) | Migration guide |
| [troubleshooting.md](docs/troubleshooting.md) | Troubleshooting |
| [benchmark-results.md](docs/benchmark-results.md) | Benchmark data |
| [api_reference.md](docs/api_reference.md) | API reference |

## License

[Apache License 2.0](LICENSE)

## Acknowledgments

- [vLLM](https://github.com/vllm-project/vllm) — High-performance LLM inference engine
- [Qwen](https://github.com/QwenLM/Qwen3) — Qwen3.6 model
- [Qwen-Fixed-Chat-Templates](https://github.com/froggeric/Qwen-Fixed-Chat-Templates) — Fixed chat template
