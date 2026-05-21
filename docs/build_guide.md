# Docker 镜像构建指南

> 本文档记录了从零构建 vLLM Docker 镜像的完整实战经验。
> 所有内容来自实际构建过程，包含成功方案和失败教训。

## 版本溯源

镜像所基于的 vLLM 源码版本记录在 [BUILD_INFO](../BUILD_INFO) 中。重新编译或升级前请确认版本信息。

当前构建版本：
- **vLLM**: `0.21.0` (commit `ad7125a431`, vllm-project/vllm 官方 main, 含 local-fixes/ 兼容性补丁)
- **CUDA**: 13.0.2 | **Python**: 3.12 | **架构**: sm_86 (RTX 3090 only)
- **外部依赖版本**: 详见根目录 BUILD_INFO `[external-deps]` 段

## 快速构建

```bash
cd docker-vllm-public && ./manage.sh build
```

等价于:
```bash
sudo -E DOCKER_BUILDKIT=1 docker build \
    --network=host \
    --progress=plain \
    --file docker-vllm-public/Dockerfile \
    --target vllm-openai \
    --tag vllm-qwen36:rtx3090-sm86 \
    --build-arg torch_cuda_arch_list="8.6" \
    --build-arg max_jobs=$(nproc) \
    --build-arg nvcc_threads=$(($(nproc) > 8 ? 8 : $(nproc))) \
    --build-arg RUN_WHEEL_CHECK=false \
    --build-arg PIP_INDEX_URL="https://pypi.org/simple/" \
    --build-arg PIP_EXTRA_INDEX_URL="https://pypi.org/simple/" \
    --build-arg UV_INDEX_URL="https://pypi.org/simple/" \
    --build-arg UV_EXTRA_INDEX_URL="https://pypi.org/simple/" \
    ./vllm
# 中国用户: 添加 --build-arg USE_CHINA_MIRROR=true
```

## 构建参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `--target vllm-openai` | 构建目标 | 最终生成 OpenAI 兼容服务镜像 |
| `--network=host` | 网络模式 | 直连国内源（阿里云 apt/pip、nvidia.cn），无需代理 |
| `torch_cuda_arch_list` | `"8.6"` | 仅编译 RTX 3090，大幅缩短编译时间 |
| `max_jobs` | `24` | Ninja 并行任务数（96 核 CPU 适配，OOM 时降至 12） |
| `nvcc_threads` | `4` | NVCC 每编译单元线程数（有效并发 = 24÷4 = 6） |
| `RUN_WHEEL_CHECK` | `false` | 跳过 wheel 大小校验 |
| `PIP_INDEX_URL` | 阿里云 | pip 主索引 |
| `PIP_EXTRA_INDEX_URL` | 阿里云 | pip 备用索引（国内直连，无需 pypi.org） |
| `UV_INDEX_URL` | 阿里云 | uv 主索引 |
| `UV_EXTRA_INDEX_URL` | 阿里云 | uv 备用索引 |

## 多阶段构建流程

自定义 Dockerfile 基于 vLLM 官方 Dockerfile（`./vllm/docker/Dockerfile`），做了以下补丁：

```
base              → Ubuntu 22.04 + 阿里云 apt 源 + Python 3.12 + PyTorch
csrc-build        → 编译全部 C++/CUDA 扩展 (340 个编译目标)
                    - FA2 (SM80): flash_attn 内核
                    - FA3 (SM90): Hopper 内核（不用于 RTX 3090 但仍编译）
                    - MoE, quantization, triton_kernels 等
extensions-build  → 编译 DeepEP (与 csrc-build 并行)
build             → 打包 vLLM Python wheel
vllm-base         → 运行时基础 (FlashInfer 可选, GDRCopy)
vllm-openai       → 最终镜像, ENTRYPOINT ["vllm", "serve"]
```

### Dockerfile 与官方的差异点

| 位置 | 官方 | 自定义 |
|------|------|--------|
| base apt 源 | Ubuntu 官方 | 阿里云镜像 |
| vllm-base apt 源 | Ubuntu 官方 | 阿里云镜像 |
| pip/uv 索引 | PyPI 官方 | 阿里云为主，PyPI 为备 |
| Python 安装 | `ensurepip` | `--without-pip` + `get-pip.py`（绕过 distutils 缺失） |
| FlashInfer | 默认安装 | 可选（`INSTALL_FLASHINFER=false`） |
| 外部依赖 | cmake FetchContent (git clone) | 预克隆 + COPY + `*_SRC_DIR` 环境变量 |
| uv pip install | 直接执行 | 带重试循环（10 次重试） |
| cutlass | vLLM 专用 v4.4.2 | v4.4.2 + flash-attn 专用 v3.9 |

## 构建时间实测

| 场景 | 时间 | 说明 |
|------|------|------|
| 完整构建（无缓存，无代理优化） | ~45 分钟 | 无代理 + 全阿里云源 + max_jobs=24 |
| 完整构建（无缓存，旧参数） | ~90 分钟 | 代理环境 + max_jobs=8 + pypi.org 备用源 |
| 增量重建（Dockerfile 小改） | ~75 分钟 | csrc-build 缓存不命中需重新编译 |
| 仅改 Python 代码 | ~5 分钟 | csrc-build 缓存命中 |
| 切换 cutlass 版本 | ~70 分钟 | csrc-build 从 FA2 开始重编译 |

## 外部依赖管理策略

### 为什么需要预克隆

vLLM 的 cmake 构建通过 FetchContent 从 GitHub 拉取 7 个外部依赖。在代理环境下 git clone 经常超时失败。
解决方案：在宿主机预克隆，通过 Docker COPY 注入容器，利用 cmake 的 `*_SRC_DIR` 环境变量使用本地源码。

### 依赖清单

| 依赖 | 版本 | 用途 | `*_SRC_DIR` 环境变量 |
|------|------|------|---------------------|
| triton | v3.6.0 (tag) | Triton kernels | `TRITON_KERNELS_SRC_DIR` |
| deepgemm | 891d57b4db | DeepGEMM kernels | `DEEPGEMM_SRC_DIR` |
| flashmla | 9241ae3ef9 | FlashMLA | `FLASH_MLA_SRC_DIR` |
| qutlass | 830d2c4537 | qutlass | `QUTLASS_SRC_DIR` |
| flash-attn | f5bc33cfc0 | Flash Attention FA2/FA3 | `VLLM_FLASH_ATTN_SRC_DIR` |
| cutlass | v4.4.2 (tag) | CUTLASS (vLLM 用) | `VLLM_CUTLASS_SRC_DIR` |
| cutlass-fa | 62750a2b (v3.9) | CUTLASS (flash-attn 用) | — |

### 关键经验：两份 cutlass

**flash-attn 需要自己的 cutlass 副本**，原因：
- vLLM 使用 cutlass v4.4.2
- flash-attn 的 git submodule 指向 cutlass v3.9 (commit 62750a2b)
- FA3 代码中 `sm90_get_smem_store_op_for_accumulator` 函数签名在 v3.9 和 v4.4.2 之间不兼容
- 用 v4.4.2 编译 FA3 会报 C++ 模板推导错误

容器内的目录结构：
```
/opt/external-deps/
├── cutlass/          # v4.4.2 — vLLM CMakeLists.txt 通过 FetchContent 使用
├── cutlass-fa/       # v3.9  — flash-attn 的 csrc/cutlass 符号链接指向这里
├── flash-attn/
│   └── csrc/cutlass  # → /opt/external-deps/cutlass-fa (符号链接)
├── triton/
├── deepgemm/
├── flashmla/
└── qutlass/
```

Dockerfile 中在 COPY 后创建符号链接：
```dockerfile
RUN rm -rf /opt/external-deps/flash-attn/csrc/cutlass && \
    ln -s /opt/external-deps/cutlass-fa /opt/external-deps/flash-attn/csrc/cutlass
```

### 预克隆命令参考

> 推荐使用 `./manage.sh prepare-src` 自动克隆全部 7 个依赖。以下手动命令仅供参考/调试。

```bash
# 在 ./vllm/external-src/ 下克隆（git submodule 目录内）
cd ./vllm/external-src

git clone --depth 1 --branch v3.6.0 https://github.com/triton-lang/triton.git
git clone --depth 1 https://github.com/deepseek-ai/DeepGEMM.git deepgemm && cd deepgemm && git checkout 891d57b4db
git clone --depth 1 https://github.com/deepseek-ai/FlashMLA.git flashmla && cd flashmla && git checkout 9241ae3ef9
git clone --depth 1 https://github.com/vllm-project/flash-attention.git flash-attn && cd flash-attn && git checkout f5bc33cfc0
git clone --depth 1 https://github.com/IST-DASLab/qutlass.git qutlass && cd qutlass && git checkout 830d2c4537
git clone --depth 1 --branch v4.4.2 https://github.com/NVIDIA/cutlass.git

# flash-attn 需要独立的 cutlass v3.9 副本（与 vLLM 的 v4.4.2 不兼容）
git clone --depth 1 --branch v3.9.0 https://github.com/NVIDIA/cutlass.git cutlass-fa

# 清理 .git 目录减小体积
find . -maxdepth 2 -name ".git" -type d -exec rm -r {} +
```

## Dockerfile 补丁点详解

### 补丁点 1: base 阶段 — 阿里云 apt 源 + Python 3.12 安装

```dockerfile
FROM ${BUILD_BASE_IMAGE} AS base

# 注入阿里云 apt 源
RUN sed -i 's|http://archive.ubuntu.com|https://mirrors.aliyun.com|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com|https://mirrors.aliyun.com|g' /etc/apt/sources.list

# Python 3.12 安装（绕过 distutils 缺失）
RUN apt-get update -y && apt-get install -y ... \
    && python${PYTHON_VERSION} -m venv --without-pip --copies /opt/venv \
    && curl -sS ${GET_PIP_URL} | /opt/venv/bin/python3 - \
        --index-url ${PIP_INDEX_URL} --trusted-host mirrors.aliyun.com \
    && /opt/venv/bin/pip install --quiet uv
```

**为什么 `--without-pip`**: Python 3.12 移除了 `distutils`（PEP 632），`ensurepip` 依赖 distutils 会报错。
使用 `--without-pip --copies` 创建 venv，然后通过 `get-pip.py` 引导安装 pip。

**为什么 `--copies`**: 默认 venv 使用符号链接（`/opt/venv/bin/python3 → /usr/bin/python3.12`）。
当后续创建 `/usr/bin/python3 → /opt/venv/bin/python3` 时形成双重符号链接链，
`python3 -m pip` 无法检测到 venv 环境。`--copies` 复制二进制避免此问题。

### 补丁点 2: csrc-build 阶段 — 外部依赖注入

```dockerfile
# 注意：复制到 /opt/external-deps/ 而非 /workspace/.deps/
# 因为 setup.py 会执行 rm -rf .deps 清理 FetchContent 缓存
COPY external-src/triton /opt/external-deps/triton
COPY external-src/deepgemm /opt/external-deps/deepgemm
# ...
ENV VLLM_FLASH_ATTN_SRC_DIR=/opt/external-deps/flash-attn
ENV VLLM_CUTLASS_SRC_DIR=/opt/external-deps/cutlass
```

### 补丁点 3: uv pip install 重试循环

```dockerfile
RUN --mount=type=cache,target=/root/.cache/uv \
    _SUCCESS=0 && for _attempt in 1 2 3 4 5 6 7 8 9 10; do \
        uv pip install --system ... && _SUCCESS=1 && break; \
        echo "[Retry ${_attempt}/10] ..."; sleep 10; \
    done && [ "$_SUCCESS" = "1" ]
```

**为什么需要重试**: Clash 代理 (7890) 对大文件下载（PyTorch ~2GB, bitsandbytes ~800MB）
存在 SSL 连接中断问题。uv cache (`--mount=type=cache`) 会保留已下载的包，
每次重试只下载之前失败的部分，逐步积累直到全部成功。

## 构建注意事项

### 代理冲突

Docker 构建受两层代理影响，必须同时清除：

1. **Docker 客户端代理** — `~/.docker/config.json` 中的 `proxies.default` 配置会被自动注入所有构建容器。manage.sh 通过 `source ~/.off_proxy` 清除 shell 变量，但如果仍失败，需显式传空值覆盖：
   ```bash
   --build-arg http_proxy= --build-arg https_proxy= --build-arg no_proxy=
   ```

2. **Shell 环境代理** — `http_proxy`/`https_proxy` 环境变量会被 `sudo -E` 传递到构建进程。

**症状**: apt-get 报 `Error reading from server [IP: 127.0.0.1 7890]`，说明代理仍在干扰。

### 磁盘空间

镜像 ~20GB + 构建缓存 ~90GB，构建前检查：
```bash
df -h /var/lib/docker
```
可用空间不足 30GB 时，构建末尾镜像解包阶段会报 `ENOSPC: no space left on device`（前面的编译阶段都正常，仅最后写入层失败）。

清理命令：
```bash
sudo docker image prune -f       # 删除 dangling 镜像
sudo docker builder prune -f     # 清理构建缓存
sudo docker system prune -f      # 综合清理
```

### 编译资源调优

`max_jobs` 和 `nvcc_threads` 需根据服务器硬件调整：

| 服务器 | CPU 核数 | RAM | max_jobs | nvcc_threads | 有效并发 |
|--------|---------|-----|----------|-------------|---------|
| 开发测试 | 8 核 | 32GB | 8 | 4 | 2 |
| 生产部署 | 96 核 | 251GB | 24 | 4 | 6 |

OOM 时降低 `max_jobs`，不调整 `nvcc_threads`。

## 构建缓存

BuildKit 自动缓存以下内容（跨构建持久化）:
- `/root/.cache/uv` — pip/uv 包缓存（避免重新下载）
- `/root/.cache/ccache` — C/CUDA 编译缓存（避免重新编译未修改的文件）

清除缓存重新构建:
```bash
sudo docker buildx prune -a -f
```

## 重新构建场景

### 仅修改 Python 代码
csrc-build 层有缓存 → 5-10 分钟

### 修改 C++/CUDA 代码
csrc-build 层缓存不命中 → 20-40 分钟（取决于修改范围）

### 升级 vLLM 版本

```bash
# 1. 更新 docker-vllm submodule 到新官方版本
cd docker-vllm-public/vllm && git fetch origin && git checkout <新版本>
cd docker-vllm-public && git add vllm

# 2. 检查修复覆盖兼容性
#    manage.sh build 自动检测 local-fixes/CHECK 标记
#    如不兼容: 基于 vllm/ 新版源码更新 local-fixes/ 中的文件
#    如官方已修复: manage.sh 自动跳过覆盖

# 3. 重新准备外部依赖（清除旧缓存，重新克隆）
rm -r vllm/external-src/* 2>/dev/null; rmdir vllm/external-src/* 2>/dev/null
./manage.sh prepare-src

# 4. 更新 Dockerfile（如有变化）
#    - 复制新官方 Dockerfile
#    - 重新打补丁（阿里云源、Python 安装、外部依赖注入）
#    - 更新 cutlass-fa（如果 flash-attn submodule 变了）

# 5. 记录新版本到 BUILD_INFO
#    - 更新 [vllm-source] 段的 version/commit/commit_date/branch/describe
#    - 更新 [external-deps] 段的依赖版本
#    - 更新 [build-args] 段（如有变化）

# 6. 重新构建（manage.sh build 自动应用修复覆盖）
./manage.sh build
```

### Dockerfile 维护策略

自定义 Dockerfile 是官方 `./vllm/docker/Dockerfile` 的完整副本 + 补丁。
升级时重新复制官方 Dockerfile 并在以下位置注入补丁：

1. `base` 阶段: `FROM ${BUILD_BASE_IMAGE} AS base` 后 — 阿里云 apt 源
2. `vllm-base` 阶段: `FROM ${FINAL_BASE_IMAGE} AS vllm-base` 后 — 阿里云 apt 源
3. `csrc-build` 阶段: 外部依赖 COPY + ENV + 符号链接
4. `csrc-build` 阶段: uv pip install 重试循环

## 本地修复管理

### 架构

项目包含 `local-fixes/` 目录，存放 vLLM 源码的兼容性补丁。`manage.sh build` 自动检测并应用这些补丁。

```
docker-vllm-public/
├── vllm/ (submodule → vllm-project/vllm 官方)
├── local-fixes/               ← 修改后的 vllm 源码
│   ├── CHECK                  ← auto-detection 标记
│   └── vllm/vllm/v1/core/...  ← 修改文件
├── manage.sh（含 _apply_local_fixes）
└── Dockerfile
```

`manage.sh build` 自动检测 `local-fixes/` 存在性：存在则覆盖到 vllm/ 后构建，不存在则构建纯官方版本。

### auto-detection

`local-fixes/CHECK` 文件包含检测标记字符串。构建时，manage.sh grep 这些标记在 vllm/ 官方源码中是否存在。若存在 → 官方已包含修复 → 跳过覆盖。

### 当前修复

| 修改文件 | 说明 |
|---------|------|
| kv_cache_utils.py, kv_cache_interface.py | Qwen3.6 混合架构 KV cache page_size_padded 缩放修复 |
| sampling_params.py | Speculative decoding min_p/logit_bias 兼容性 (raise→warn+reset) |

### vLLM 版本升级

```bash
# 1. 更新 submodule
cd docker-vllm-public/vllm
git fetch origin
git checkout <新版本>
cd .. && git add vllm && git commit -m "update vllm to vX.Y.Z"

# 2. 验证兼容性（manage.sh build 自动检测 local-fixes/ 兼容性）
./manage.sh build
# - 全部跳过 → 官方已包含修复，可删除 local-fixes/
# - 部分覆盖 → 正常构建
# - 不兼容 → 手动更新 local-fixes/ 中对应文件

# 3. 重新构建
./manage.sh prepare-src --force && ./manage.sh build

# 4. 更新 BUILD_INFO 所有版本字段
```

## 代理与版本检测（自动处理）

`./manage.sh build` 已内置以下自动处理，无需手动干预：

1. **三层代理清除**: a) unset shell 环境变量 + source ~/.off_proxy；b) 显式 `--build-arg http_proxy= https_proxy= ftp_proxy=` 覆盖 Docker `config.json` 代理注入；c) `--network=host` 确保容器直连宿主机网络

2. **setuptools_scm 版本检测**: 子模块 `.git` 是 gitdir 引用（指向父仓库 `.git/modules/vllm`），在 Docker COPY 后无法使用。manage.sh 自动通过 `git describe` 获取版本字符串并传入 `--build-arg SETUPTOOLS_SCM_PRETEND_VERSION`

## 镜像信息

```
镜像名: vllm-qwen36:rtx3090-sm86
基础: nvidia/cuda:13.0.2-base-ubuntu22.04
Python: 3.12
CUDA: 13.0.2
架构: sm_86 (RTX 3090 only)
磁盘: 20.1 GB
压缩: 6.25 GB
入口: vllm serve
```

## 镜像迁移

```bash
# 导出
sudo docker save vllm-qwen36:rtx3090-sm86 | gzip > vllm-qwen36-rtx3090-sm86.tar.gz

# 导入
sudo docker load < vllm-qwen36-rtx3090-sm86.tar.gz
```

目标机器需: NVIDIA 驱动 (CUDA 13.0+) + Docker + NVIDIA Container Toolkit
