# Qwen3.6-27B-FP8 Docker 推理部署指南

[English](README_EN.md)

> 双服务并行 vLLM 推理部署 | 支持 2-4x GPU (24GB+, sm_86) | 预编译镜像 + 源码构建
> 镜像: [`maricgeer/vllm-qwen36`](https://crpi-3jsqnspnt5spjb2h.ap-southeast-1.personal.cr.aliyuncs.com/maricgeer/vllm-qwen36) | 基于 [vLLM](https://github.com/vllm-project/vllm) v0.21.0
> 版本溯源: [BUILD_INFO](BUILD_INFO)

## 硬件要求

| 配置 | 最低 | 推荐 |
|------|------|------|
| GPU | 2x RTX 3090 (24GB) | 4x RTX 3090 |
| GPU 架构 | sm_86 (Ampere) | sm_86 |
| 显存 (每卡) | ~22.7 GB | ~22.7 GB |
| 磁盘 | 50 GB 可用 | 100 GB 可用 |
| CPU | 8 核+ | 24 核+ |
| 模型文件 | ~27 GB | ~27 GB |

**GPU 架构说明**: 预编译镜像仅支持 **sm_86** (RTX 3090 / A6000 / A40)。其他架构 GPU (RTX 4090/A100/H100) 需从源码构建。

## 快速开始

### 方式一: 使用预编译镜像（推荐，sm_86 GPU，约 30 分钟）

> 适用于 RTX 3090 / A6000 / A40 等 sm_86 架构 GPU，无需编译。

```bash
# 1. 克隆仓库（不需要 submodule，仅获取配置文件和管理脚本）
git clone https://github.com/GeerMrc/docker-vllm-public.git && cd docker-vllm-public

# 2. 拉取预编译镜像（~6.3 GB 压缩，约 10-30 分钟）
docker pull crpi-3jsqnspnt5spjb2h.ap-southeast-1.personal.cr.aliyuncs.com/maricgeer/vllm-qwen36:sm86-v0.21.0

# 3. 下载模型（~27GB，首次需等待）
huggingface-cli download Qwen/Qwen3.6-27B-FP8 --local-dir /path/to/Qwen3.6-27B-FP8
# 或启用加速: HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download ...

# 4. 配置
cp .env.example .env
# 编辑 .env，必须修改以下内容:
#   VLLM_MODEL_PATH=/path/to/Qwen3.6-27B-FP8          ← 改为你的模型路径
#   VLLM_PRIMARY_GPU_IDS=0,1                           ← 改为你的 GPU 编号
#   VLLM_SECONDARY_GPU_IDS=2,3                         ← 如有 4 张 GPU
#   DOCKER_IMAGE 已预设为预编译镜像，无需修改

# 5. 启动
sudo ./manage.sh start
```

### 方式二: 从源码构建（所有 GPU 架构，约 60-90 分钟）

> 适用于任何 GPU 架构（RTX 4090/A100/H100 等），需编译。

```bash
# 1. 克隆仓库 + 子模块
git clone https://github.com/GeerMrc/docker-vllm-public.git && cd docker-vllm-public
git submodule update --init              # 拉取 vLLM 源码（~201MB）

# 2. 准备编译依赖
./manage.sh prepare-src                  # ~422MB，约 5-10 分钟

# 3. 构建（首次约 45 分钟，自动检测 CPU 核心数）
./manage.sh build
# 中国用户: 编辑 .env 添加以下变量加速:
#   VLLM_USE_CHINA_MIRROR=true
#   VLLM_PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
# 其他 GPU 架构: 设置 VLLM_BUILD_CUDA_ARCH=8.9（RTX 4090）等

# 4. 下载模型 + 配置 + 启动
# 模型下载和启动同方式一的步骤 3、5
# 配置时需修改 .env 中的 DOCKER_IMAGE:
#   DOCKER_IMAGE=vllm-qwen36:rtx3090-sm86              ← 源码构建使用本地标签
```

## 模型下载

Qwen3.6-27B-FP8 模型 (~27GB) 需要单独下载，不包含在 Docker 镜像中:

```bash
# 方式一: huggingface-cli
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3.6-27B-FP8 --local-dir /path/to/Qwen3.6-27B-FP8

# 方式二: hf-transfer（更快，需安装 hf-transfer）
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download Qwen/Qwen3.6-27B-FP8 --local-dir /path/to/Qwen3.6-27B-FP8
```

下载完成后，在 `.env` 中设置 `VLLM_MODEL_PATH` 指向模型路径。

## 为其他 GPU 架构构建

预编译镜像仅支持 sm_86。如果你的 GPU 是其他架构，需要从源码构建:

```bash
# 查看你的 GPU 计算能力
nvidia-smi --query-gpu=compute_cap --format=csv

# 常见架构对照:
# RTX 3090 / A6000 / A40  →  8.6  (预编译镜像可用)
# RTX 4090 / L40S         →  8.9  (需构建)
# A100                    →  8.0  (需构建)
# H100                    →  9.0  (需构建)

# 构建时指定架构（在 .env 中设置）
VLLM_BUILD_CUDA_ARCH=8.9    # RTX 4090
./manage.sh build
```

**注意**: CUDA 内核无法交叉编译，每种架构必须在对应 GPU 上构建。

## 目录结构

```
docker-vllm-public/
├── vllm/                           # vLLM 源码（git submodule，需 git submodule update --init）
│   └── external-src/               # 编译依赖（通过 ./manage.sh prepare-src 准备）
├── Dockerfile                      # 自定义 Dockerfile（官方 + 可选中国镜像加速）
├── docker-compose.yml              # 双服务 Docker Compose 编排
├── manage.sh                       # 管理脚本 (start/stop/restart/logs/...)
├── profiles/                       # vLLM 配置模板
│   ├── agent-fast.yaml             # Agent-Fast 非思考模式
│   ├── agent-thinking.yaml         # Agent-Thinking 思考模式
│   ├── code.yaml                   # 精确编码
│   ├── instruct.yaml               # 非思考指令
│   ├── test-spec.yaml              # 测试专用入口
│   └── think.yaml                  # 通用思考 256K
├── local-fixes/                    # vLLM 兼容性补丁（构建时自动应用）
├── templates/                      # Chat Template（修复官方模板 bug）
├── config/                         # 日志 + systemd 模板
├── scripts/                        # 性能测试脚本
├── docs/                           # 文档
├── .env.example                    # 环境变量模板
└── BUILD_INFO                      # 镜像版本溯源
```

## 双服务架构

Primary (GPU 0,1, 端口 8089) + Secondary (GPU 2,3, 端口 8099)，共享同一个 Docker 镜像和模型文件（只读挂载），通过 `CUDA_VISIBLE_DEVICES` 做 GPU 隔离。

```
┌────────────────────────────────────────────────────────────┐
│                    宿主机 (4x GPU)                          │
│                                                            │
│  ┌───────────────────┐    ┌───────────────────────┐       │
│  │   vllm-primary       │    │   vllm-secondary      │       │
│  │   GPU: 0,1        │    │   GPU: 2,3            │       │
│  │   Port: 8089      │    │   Port: 8099          │       │
│  │   Config: profile.yaml │   Config: profile.yaml│       │
│  │     (由 .env 控制)│    │   (由 .env 控制)      │       │
│  │   Log: logs/primary │   Log: logs/secondary │       │
│  └────────┬──────────┘    └────────┬───────────────┘       │
│           │                        │                        │
│     ┌─────┴────────────────────────┴─────┐                 │
│     │         共享资源                     │                 │
│     │  • Docker 镜像                      │                 │
│     │  • 模型文件 (只读挂载)               │                 │
│     │  • torch.compile 缓存 (named vol)   │                 │
│     │  • chat_template.jinja             │                 │
│     │  • logging.json                    │                 │
│     └────────────────────────────────────┘                 │
└────────────────────────────────────────────────────────────┘
```

> 也可仅运行单服务 (2x GPU)，参见 `./manage.sh start primary`。

## .env 环境变量

```bash
cp .env.example .env   # 从模板创建
```

**关键变量**:

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VLLM_MODEL_PATH` | (必须设置) | 模型文件路径 |
| `VLLM_PRIMARY_GPU_IDS` | `0,1` | Primary GPU 编号 |
| `VLLM_SECONDARY_GPU_IDS` | `2,3` | Secondary GPU 编号 |
| `VLLM_PRIMARY_PROFILE` | `agent-fast` | Primary profile |
| `VLLM_SECONDARY_PROFILE` | `agent-thinking` | Secondary profile |
| `VLLM_PRIMARY_API_KEY` | (空) | API Key (空=不鉴权) |
| `DOCKER_IMAGE` | 阿里云预编译地址 (见 .env.example) | 镜像名 (源码构建后为 `vllm-qwen36:rtx3090-sm86`) |
| `NCCL_P2P_DISABLE` | (空) | SYS 拓扑设 `1` |

**中国用户可选变量**:

| 变量 | 说明 |
|------|------|
| `VLLM_USE_CHINA_MIRROR=true` | 构建时使用阿里云 apt + PyPI 镜像 |
| `VLLM_PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/` | 自定义 PyPI 镜像 |

完整变量列表见 `.env.example`。

## 管理脚本用法

```bash
# 启动
./manage.sh start                   # 启动双服务
./manage.sh start primary           # 仅启动 Primary (2x GPU)
./manage.sh start secondary         # 仅启动 Secondary

# 停止/重启
./manage.sh stop [primary|secondary|all]
./manage.sh restart [primary|secondary|all]

# Profile 管理
./manage.sh set-profile primary <name>   # 切换 profile（写入 .env）
./manage.sh restart primary              # 重启生效

# 状态/日志
./manage.sh config                     # 查看配置（无需 Docker）
./manage.sh status                     # 运行状态
./manage.sh logs [primary|secondary]   # Docker 日志
./manage.sh logfiles [primary|secondary]  # 文件日志

# 编辑/构建
./manage.sh build                   # 构建镜像
./manage.sh prepare-src             # 准备编译依赖

# CPU 隔离（NUMA 拓扑）
./manage.sh detect-topology            # 检测 NUMA 拓扑
./manage.sh apply-cpuset               # 自动配置 cpuset

# 开机自启 + 关机保护
./manage.sh enable-boot                # 一键配置
./manage.sh disable-boot               # 移除
./manage.sh check-driver               # NVIDIA 驱动检查
```

## 服务 Profile 说明

| Profile | 思考 | 温度 | top_p | penalty | 上下文 | Decode | 说明 |
|---------|------|------|-------|---------|--------|--------|------|
| agent-fast | OFF | 0.7 | 0.80 | 1.5 | 140K | 37.6 tok/s | 低延迟工具调用 |
| agent-thinking | ON | 0.6 | 0.95 | 0.0 | 140K | 37.6 tok/s | 多轮推理 (preserve) |
| code | ON | 0.6 | 0.95 | 0.0 | 140K | 37.6 tok/s | 精确编码 |
| instruct | OFF | 0.7 | 0.80 | 1.5 | 128K | ~48 tok/s | 快速问答 |
| think | ON | 1.0 | 0.95 | 0.0 | 256K | ~48 tok/s | 通用思考，大上下文 |
| test-spec | ON | 0.6 | 0.95 | 0.0 | 152K | — | 测试专用 |

> agent-fast、agent-thinking、code 已启用 MTP Speculative Decoding (n=1) + bf16 KV cache，吞吐量在并发 c=4 时达 203 tok/s。

## API 使用

```bash
# 列出模型
curl http://localhost:8089/v1/models

# Chat Completions
curl http://localhost:8089/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.6-27B-FP8",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 200
  }'

# 带 API Key
curl -H "Authorization: Bearer YOUR_KEY" http://localhost:8089/v1/models
```

> 完整 API 文档见 [docs/api_reference.md](docs/api_reference.md)。

## 性能实测

| 指标 | 单服务 (2x GPU) | 双服务 (4x GPU) |
|------|-----------------|-----------------|
| Decode (bf16+MTP) | 37.6 tok/s @c=1 | 持平 |
| c=4 吞吐 | 203 tok/s | 持平 |
| GPU 显存 (每卡) | ~22.7 GB | ~22.7 GB |
| 最大上下文 | 140K (agent/code) | 每服务视 profile |

## GPU 拓扑要求

部署前**必须**检查 GPU 拓扑:

```bash
nvidia-smi topo -m
```

如果 GPU 对之间显示 **SYS**（跨 NUMA 节点），必须在 `.env` 设置 `NCCL_P2P_DISABLE=1`。

## local-fixes 说明

项目包含 3 个 vLLM 源码补丁，解决 Qwen3.6 模型的兼容性问题:

| 修复 | 文件 | 说明 |
|------|------|------|
| MTP 兼容性 | kv_cache_utils.py, kv_cache_interface.py, sampling_params.py | KV cache `page_size_padded` 缩放 + Speculative decoding `min_p`/`logit_bias` 兼容性 |

`manage.sh build` 自动检测并应用补丁。当 vLLM 官方包含修复后，补丁会自动跳过。

## 预编译镜像（阿里云）

```
docker pull crpi-3jsqnspnt5spjb2h.ap-southeast-1.personal.cr.aliyuncs.com/maricgeer/vllm-qwen36:sm86-v0.21.0
```

| Tag | GPU 架构 | vLLM 版本 | 压缩大小 |
|-----|---------|-----------|---------|
| `sm86-v0.21.0` | RTX 3090 / A6000 / A40 (sm_86) | 0.21.0 | ~6.3 GB |
| `sm86-latest` | 同上 | latest | ~6.3 GB |

**重要**: 镜像仅包含 vLLM 运行时，不包含模型文件。模型需单独下载。

## 文档索引

| 文档 | 内容 |
|------|------|
| [BUILD_INFO](BUILD_INFO) | 镜像版本溯源 |
| [build_guide.md](docs/build_guide.md) | 构建全流程 |
| [migration_guide.md](docs/migration_guide.md) | 镜像迁移部署指南 |
| [troubleshooting.md](docs/troubleshooting.md) | 编译错误 + 运行时问题排查 |
| [benchmark-results.md](docs/benchmark-results.md) | 基准测试数据 |
| [api_reference.md](docs/api_reference.md) | API 规范性参考 |

## 注意事项

1. **顺序启动**: `start all` 先启动 primary 再启动 secondary，确保命中 torch.compile 热缓存
2. **sudo**: manage.sh 使用 `sudo` 调用 docker（或将用户加入 docker 组）
3. **模型路径**: 模型文件通过只读 volume 挂载，不复制到镜像内
4. **显存配置**: `gpu_memory_utilization` 默认 0.97（极限），如遇 OOM 可降至 0.95
5. **Profile 切换**: 通过 `set-profile` + `restart` 两步操作

## 迁移到其他机器

详见 [migration_guide.md](docs/migration_guide.md)。

## 许可证

[Apache License 2.0](LICENSE)

## 致谢

- [vLLM](https://github.com/vllm-project/vllm) — 高性能 LLM 推理引擎
- [Qwen](https://github.com/QwenLM/Qwen3) — Qwen3.6 模型
- [Qwen-Fixed-Chat-Templates](https://github.com/froggeric/Qwen-Fixed-Chat-Templates) — 修复版 Chat 模板
