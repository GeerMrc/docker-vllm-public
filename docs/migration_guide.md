# Docker 镜像迁移部署指南

> 将本地构建的 `vllm-qwen36:rtx3090-sm86` 镜像迁移到另一台服务器进行推理部署。
> 镜像完全自包含，无需额外基础镜像或编译环境。

---

## 一、核心结论

**镜像是否可以直接使用？** 是。`docker save` 导出的 tar 包包含全部 26 层，无需任何外部依赖。

**是否需要额外基础镜像？** 不需要。镜像已内嵌完整的 CUDA 13.0.2 runtime、Python 3.12、PyTorch、vLLM 及全部 CUDA 编译产物。

**目标服务器是否需要编译环境？** 不需要。无需 CUDA Toolkit、无需 GCC、无需 Python、无需 git。

**目标服务器需要什么？** 仅需 3 项：Linux 系统 + NVIDIA 驱动 + Docker + NVIDIA Container Toolkit。

---

## 二、镜像技术档案

| 属性 | 值 |
|------|-----|
| 镜像名 | `vllm-qwen36:rtx3090-sm86` |
| 镜像 ID | `sha256:8a1589dc4ec860f7c81480d80ca177c164b23492f8d07fd55d13366196688907` |
| 虚拟大小 | 20.1 GB（26 层） |
| 压缩导出 | ~6.25 GB |
| 操作系统 | Ubuntu 22.04 |
| CUDA | 13.0.2 runtime（内嵌，无需主机安装 CUDA Toolkit） |
| Python | 3.12（内嵌于 `/opt/venv`） |
| PyTorch | 已编译 sm_86（RTX 3090 专用） |
| vLLM | 0.21.0（commit `ad7125a431`） |
| 父镜像 | 无（完全独立，无外部依赖） |
| 入口 | `vllm serve` |

---

## 三、目标服务器最低要求

### 3.1 硬件要求

| 组件 | 要求 | 说明 |
|------|------|------|
| **GPU** | RTX 3090 x4（或任意 compute_cap=8.6 的 GPU） | 双服务并行: Primary (GPU 0,1) + Secondary (GPU 2,3) |
| **显存** | 每卡 >= 24 GB | 4x RTX 3090 = 96 GB 总显存 |
| **系统内存** | >= 32 GB | 推荐 64 GB |
| **磁盘** | >= 150 GB 可用空间 | 镜像 20 GB + 模型 50 GB + 编译缓存 + 余量 |
| **PCIe** | PCIe 3.0/4.0 x16 | TP=2 需要足够的卡间带宽 |

### 3.2 软件要求

| 组件 | 最低版本 | 推荐版本 | 验证命令 |
|------|---------|---------|---------|
| **NVIDIA 驱动** | >= 565.x | >= 570.x | `nvidia-smi` |
| **Docker Engine** | >= 20.10 | >= 24.0 | `docker version` |
| **NVIDIA Container Toolkit** | >= 1.12.0 | >= 1.14.0 | `nvidia-ctk --version` |
| **操作系统** | Linux x86_64 | Ubuntu 22.04+ | `uname -m` |
| **Docker Compose** | >= 2.17（V2） | >= 2.24 | `docker compose version` |

### 3.3 NVIDIA 驱动与 CUDA 兼容性

```
镜像内 CUDA 版本: 13.0.2
NVIDIA 驱动 CUDA 下限要求: >= 565.x (支持 CUDA 13.0+)
当前构建主机驱动: 595.45.04 (支持 CUDA 13.2)

关键规则: 驱动支持的 CUDA 版本 >= 镜像内 CUDA 版本
         595.45.04 (13.2) >= 13.0.2 ✓
```

**驱动版本选择指南：**

| 驱动版本系列 | CUDA 上限 | 能否运行此镜像 |
|-------------|-----------|--------------|
| 535.x | 12.2 | **不能**（镜像 CUDA 13.0 > 驱动上限 12.2） |
| 545.x | 12.3 | **不能** |
| 550.x | 12.4 | **不能** |
| 555.x | 12.5 / 12.6 | **不能**（视具体版本） |
| 565.x | 13.0 | **边界**（13.0 >= 13.0 勉强通过，建议更高） |
| 570.x | 13.0+ | **可以** |
| 575.x+ | 13.0+ | **可以**（推荐） |

> **重要**: 镜像内嵌 CUDA 13.0.2 runtime，驱动必须支持 CUDA 13.0+。
> 旧驱动（< 565）无法运行此镜像。如目标服务器驱动版本不够，需先升级驱动。

### 3.4 GPU 架构兼容性

此镜像 **仅编译了 sm_86（compute capability 8.6）**。

| GPU | Compute Cap | 能否运行 |
|-----|------------|---------|
| RTX 3090 | 8.6 | **可以**（目标架构） |
| RTX 3080 | 8.6 | **可以** |
| RTX 3080 Ti | 8.6 | **可以** |
| RTX 4090 | 8.9 | **不能**（需要 sm_89） |
| RTX 4080 | 8.9 | **不能** |
| A100 | 8.0 | **不能**（需要 sm_80） |
| H100 | 9.0 | **不能**（需要 sm_90） |

> 如需支持其他 GPU 架构，需要重新构建镜像并修改 `--build-arg torch_cuda_arch_list`。
> 例如 RTX 4090: `torch_cuda_arch_list="8.9"`，多架构: `"8.0;8.6;8.9"`。

---

## 四、迁移步骤

### 4.1 构建主机：导出镜像

```bash
# 导出镜像为 tar.gz（约 6.25 GB，耗时约 5-10 分钟）
sudo docker save vllm-qwen36:rtx3090-sm86 | gzip > vllm-qwen36-rtx3090-sm86.tar.gz

# 验证导出完整性
ls -lh vllm-qwen36-rtx3090-sm86.tar.gz
# 预期: ~6.25G

# 记录镜像 ID（用于目标验证）
sudo docker images vllm-qwen36:rtx3090-sm86 --format "{{.ID}}"
# 预期: 8a1589dc4ec8 (短 ID)
```

### 4.2 构建主机：打包部署目录

```bash
# 打包部署配置（不含 .env 敏感信息、不含模型文件、不含日志）
cd ..
tar czf docker-vllm-deploy.tar.gz \
    --exclude='docker-vllm-public/.env' \
    --exclude='docker-vllm-public/flashinfer-cache/' \
    --exclude='docker-vllm-public/logs/' \
    --exclude='docker-vllm-public/vllm/' \
    --exclude='docker-vllm-public/.git/' \
    docker-vllm-public/

ls -lh docker-vllm-deploy.tar.gz
# 预期: 约 50 KB（仅配置文件和文档）
```

**需要传输的文件清单：**

| 文件 | 大小 | 说明 |
|------|------|------|
| `vllm-qwen36-rtx3090-sm86.tar.gz` | ~6.25 GB | Docker 镜像 |
| `docker-vllm-deploy.tar.gz` | ~50 KB | 部署配置（profiles、模板、脚本） |
| 模型文件 | ~50 GB | 需单独传输（通常已有或用 hfd 下载） |

### 4.3 传输到目标服务器

```bash
# 方式一: scp（适合局域网）
scp vllm-qwen36-rtx3090-sm86.tar.gz target-user@target-ip:~/
scp docker-vllm-deploy.tar.gz target-user@target-ip:~/

# 方式二: rsync（断点续传，适合大文件）
rsync -avP vllm-qwen36-rtx3090-sm86.tar.gz target-user@target-ip:~/

# 方式三: 移动硬盘（适合离线环境）
cp vllm-qwen36-rtx3090-sm86.tar.gz /media/usb/
```

### 4.4 目标服务器：环境准备

```bash
# 1. 检查 NVIDIA 驱动
nvidia-smi
# 确认: Driver Version >= 565, CUDA Version >= 13.0

# 2. 检查 Docker
docker version
# 确认: >= 20.10

# 3. 检查 NVIDIA Container Toolkit
nvidia-ctk --version
# 确认: >= 1.12.0

# 4. 如未安装 NVIDIA Container Toolkit:
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 5. 验证 GPU 可被 Docker 访问
sudo docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu22.04 nvidia-smi

# 6. 检查 PCIe 拓扑（多卡必查，决定是否需要禁用 P2P）
nvidia-smi topo -m
# GPU 间显示 SYS = 跨 NUMA，.env 必须设置 NCCL_P2P_DISABLE=1
# GPU 间显示 PIX/NODE = 同 PCIe，不需要设置
```

### 4.5 目标服务器：导入镜像和部署

```bash
# 1. 导入 Docker 镜像（约 3-5 分钟）
sudo docker load < vllm-qwen36-rtx3090-sm86.tar.gz

# 验证导入
sudo docker images vllm-qwen36:rtx3090-sm86

# 2. 解压部署配置
cd ~
tar xzf docker-vllm-deploy.tar.gz

# 3. 创建 .env 文件（从模板复制后修改）
cd docker-vllm-public
cp .env.example .env

# 4. 修改关键配置（必改项）
# 编辑 .env，至少修改以下内容:
#   VLLM_MODEL_PATH           → 目标服务器的模型实际路径
#   VLLM_MODEL_DIR            → 模型目录名（如与默认值 Qwen3.6-27B-FP8 不同需修改）
#   NCCL_P2P_DISABLE          → SYS 拓扑设 1（nvidia-smi topo -m 查看）
#   VLLM_PRIMARY_API_KEY      → Primary 服务 API 密钥（留空则不鉴权）
#   VLLM_SECONDARY_API_KEY    → Secondary 服务 API 密钥（留空则不鉴权）
#   VLLM_PRIMARY_GPU_IDS      → Primary 使用的 GPU 编号（默认 0,1）
#   VLLM_SECONDARY_GPU_IDS    → Secondary 使用的 GPU 编号（默认 2,3）

# 5. 启动双服务（顺序启动: primary → secondary）
cd docker-vllm-public
./manage.sh start
```

### 4.6 目标服务器：验证部署

```bash
# 1. 检查容器状态
./manage.sh status

# 2. 测试 Primary 服务 API（端口 8089）
curl http://localhost:8089/v1/models

# 3. 测试 Secondary 服务 API（端口 8099）
curl http://localhost:8099/v1/models

# 4. 测试 API（有 API Key）
curl -H "Authorization: Bearer YOUR_PRIMARY_KEY" http://localhost:8089/v1/models
curl -H "Authorization: Bearer YOUR_SECONDARY_KEY" http://localhost:8099/v1/models

# 5. 测试推理
curl http://localhost:8089/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.6-27B-FP8",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 50
  }'

# 6. 检查 GPU 使用（应看到 GPU 0,1 被 vllm-primary 占用，GPU 2,3 被 vllm-secondary 占用）
nvidia-smi
```

---

## 五、注意事项

### 5.1 镜像架构限制

- 镜像编译参数 `torch_cuda_arch_list="8.6"`，**仅支持 compute capability 8.6 的 GPU**
- 如目标服务器使用不同 GPU（如 RTX 4090/A100/H100），需重新构建镜像
- 多 GPU 混合场景不支持（所有 GPU 必须是相同架构）

### 5.2 模型文件处理

- 模型文件不包含在镜像中，通过 volume 只读挂载
- 模型文件需单独传输到目标服务器（约 50 GB）
- 推荐使用 hfd (HuggingFace Downloader) 在目标服务器直接下载
- `.env` 中的 `VLLM_MODEL_PATH` 必须指向目标服务器的实际路径

### 5.3 GPU 数量与 TP

- 双服务架构：每个服务使用 TP=2（2 张 GPU），共需 4 张 GPU
- Primary 服务：GPU 0,1（通过 `VLLM_PRIMARY_GPU_IDS` 配置）
- Secondary 服务：GPU 2,3（通过 `VLLM_SECONDARY_GPU_IDS` 配置）
- TP 值由各服务的 profile YAML 文件中 `tensor_parallel_size` 控制，无需在 `.env` 中设置
- 单服务部署（仅 2 张 GPU）：`./manage.sh start primary` 仅启动 Primary

### 5.4 编译缓存

- 首次启动需要 torch.compile + CUDA Graph 编译（3-5 分钟）
- Docker named volume `vllm-compile-cache` 持久化 torch.compile 编译缓存
- Bind mount `./flashinfer-cache/primary` 和 `./flashinfer-cache/secondary` 分别持久化各服务的 FlashInfer JIT 缓存
- **torch.compile 缓存共享**: 两个服务使用相同模型和编译参数，第二个服务直接命中热缓存；FlashInfer 缓存各服务独立以避免并发写入
- **注意**: `docker compose down -v` 会删除 named volume 缓存，使用 `docker compose down` 保留
- 缓存与 GPU 架构绑定，迁移到不同架构 GPU 后需重新编译

### 5.5 网络与安全

- Primary 服务默认端口 `8089`，Secondary 服务默认端口 `8099`
- 两个服务各自独立 API Key（`VLLM_PRIMARY_API_KEY` / `VLLM_SECONDARY_API_KEY`）
- Key 为空时不鉴权，生产环境**强烈建议**设置
- 暴露到公网时，使用反向代理（Nginx/Caddy）+ TLS
- `.env` 文件包含敏感信息（API Key），不要提交到 git

### 5.6 Docker 版本兼容性

| 功能 | Docker 最低版本 |
|------|----------------|
| `docker compose` V2 插件 | 20.10+ |
| BuildKit（构建用，迁移不需要） | 20.10+ |
| `--gpus all` 参数 | 需 NVIDIA Container Toolkit |
| Docker named volume | 所有版本支持 |
| 健康检查 `healthcheck` | 20.10+ |

### 5.7 性能预期

目标服务器上的性能取决于：
- GPU 型号和数量（必须 sm_86）
- PCIe 带宽（TP=2 时卡间通信依赖 PCIe）
- 系统内存（影响 KV cache 分配）
- 磁盘 I/O（影响模型加载速度）
- PCIe 拓扑（SYS 跨 NUMA 需禁用 P2P，通信略慢）

预期性能与构建主机持平（RTX 3090 x4，每服务 2 张）：
- Decode (每服务): 37.6 tok/s @c=1 (bf16+MTP n=1)
- 首次启动: Primary ~5 min + Secondary ~30s（命中热缓存）
- 热启动: ~3 分钟

> SYS 拓扑服务器设置 `NCCL_P2P_DISABLE=1` 后，NCCL 使用 SHM 替代 PCIe P2P，
> TP 通信延迟略增但 decode 速度影响极小（<5%）。

---

## 六、故障排查

### F01: `CUDA driver version is insufficient`

```
症状: 容器启动失败，日志显示 CUDA driver version insufficient
原因: 主机 NVIDIA 驱动版本太低，不支持 CUDA 13.0
修复: 升级 NVIDIA 驱动到 565+ (推荐 575+)
验证: nvidia-smi 确认 Driver Version >= 565
```

### F02: `no kernel image is available for execution on the device`

```
症状: 运行时报错 no kernel image available
原因: GPU 架构不是 sm_86（如 RTX 4090 是 sm_89）
修复: 重新构建镜像，修改 --build-arg torch_cuda_arch_list="8.9"
```

### F03: `Could not load library libcudnn_ops_infer.so`

```
症状: 容器启动时报 cuDNN 库缺失
原因: 不太可能（镜像已内嵌），检查是否误用了其他基础镜像
修复: 确认使用的是导入的 vllm-qwen36:rtx3090-sm86 镜像
```

### F04: GPU 数量不匹配

```
症状: 启动失败，TP=2 但只检测到 1 张 GPU
原因: CUDA_VISIBLE_DEVICES 指定的 GPU 数量与 profile 中 tensor_parallel_size 不匹配
修复:
  - 检查 .env 中 VLLM_PRIMARY_GPU_IDS / VLLM_SECONDARY_GPU_IDS 是否正确
  - 确认指定的 GPU 编号数量等于 2（tensor_parallel_size=2）
  - 检查 nvidia-smi 是否能看到所有 GPU
  - 检查 docker-compose.yml 中 CUDA_VISIBLE_DEVICES 配置
```

### F05: 模型加载失败

```
症状: Model not found 或 tokenizer 加载失败
原因: VLLM_MODEL_PATH 指向的路径不存在或权限不足
修复:
  - 确认模型文件存在于目标路径: ls $VLLM_MODEL_PATH
  - 确认 .env 中 VLLM_MODEL_PATH 正确
  - 确认模型目录包含 config.json, tokenizer_config.json 等文件
```

### F06: NCCL 超时 / Worker 初始化卡死

```
症状: NCCL error 通信超时，或 Worker 进程 200% CPU 卡死在 NCCL init
原因: SYS PCIe 拓扑（GPU 跨 NUMA 节点）下 PCIe P2P 不可靠
排查:
  - nvidia-smi topo -m 查看拓扑（SYS = 跨 NUMA）
  - Worker 长时间 200% CPU + 低 GPU 显存 = NCCL hang
修复:
  - .env 设置 NCCL_P2P_DISABLE=1（禁用 P2P 改用 SHM）
  - 确认 docker-compose.yml 中 shm_size >= 4g
  - 确认 NCCL_IB_DISABLE=1 (无 InfiniBand，docker-compose.yml 已内置，无需手动设置)
  - 确认 VLLM_SKIP_P2P_CHECK=1 (跳过 vLLM P2P 检查，docker-compose.yml 已内置，无需手动设置)
```

---

## 七、快速迁移检查清单

### 构建主机

- [ ] 导出镜像: `sudo docker save vllm-qwen36:rtx3090-sm86 | gzip > vllm-qwen36-rtx3090-sm86.tar.gz`
- [ ] 打包配置: `tar czf docker-vllm-deploy.tar.gz docker-vllm-public/`
- [ ] 传输文件到目标服务器

### 目标服务器

- [ ] NVIDIA 驱动 >= 565（`nvidia-smi` 确认）
- [ ] Docker >= 20.10（`docker version` 确认）
- [ ] NVIDIA Container Toolkit >= 1.12（`nvidia-ctk --version` 确认）
- [ ] GPU 为 sm_86 架构（`nvidia-smi --query-gpu=compute_cap` 确认）
- [ ] 至少 4 张 GPU（双服务各需 2 张）
- [ ] **检查 PCIe 拓扑**: `nvidia-smi topo -m`（SYS 拓扑需设置 `NCCL_P2P_DISABLE=1`）
- [ ] 磁盘空间 >= 150 GB
- [ ] 导入镜像: `sudo docker load < vllm-qwen36-rtx3090-sm86.tar.gz`
- [ ] 解压配置: `tar xzf docker-vllm-deploy.tar.gz`
- [ ] 创建 .env（修改 `VLLM_MODEL_PATH`、`VLLM_MODEL_DIR`、`NCCL_P2P_DISABLE`、`VLLM_PRIMARY_API_KEY`、`VLLM_SECONDARY_API_KEY`）
- [ ] 确认模型文件在目标路径存在
- [ ] 配置预检: `./manage.sh config`（确认 GPU、端口、profile 配置无误，无需 Docker 运行）
- [ ] 启动双服务: `./manage.sh start`
- [ ] 验证: `curl http://localhost:8089/v1/models` 和 `curl http://localhost:8099/v1/models`
