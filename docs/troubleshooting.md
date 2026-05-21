# Docker 构建与运行 常见问题排查

> 本文档按问题分类整理所有实际遇到的错误。
> 每个条目包含：现象 → 根因 → 修复方案 → 预防措施。
> 最新: R10 Speculative Decoding OOM + R11 Profile override 不生效 + R12 Prefill OOM 根因分析 + R13 expandable_segments:False 修复后完整重测 + E14/E15 v0.21.0 构建问题。

---

## 一、编译阶段错误（csrc-build）

### E01: `cutlass/numeric_types.h: No such file or directory`

**现象**:
```
/opt/external-deps/flash-attn/csrc/flash_attn/flash_api.cpp:12:10:
fatal error: cutlass/numeric_types.h: No such file or directory
```

**根因**: flash-attn 的 `csrc/cutlass` 是 git submodule，克隆时未初始化子模块，目录为空。
flash-attn 的 CMakeLists.txt 第 161-165 行引用 `csrc/cutlass/include` 作为 include 路径。

**修复**: 在 Dockerfile 中 COPY 后创建符号链接，指向预克隆的 cutlass-fa：
```dockerfile
RUN rm -rf /opt/external-deps/flash-attn/csrc/cutlass && \
    ln -s /opt/external-deps/cutlass-fa /opt/external-deps/flash-attn/csrc/cutlass
```

**预防**: 检查 flash-attn 子模块状态：`git ls-tree HEAD csrc/cutlass`，确保目录非空。

---

### E02: FA3 编译 `sm90_get_smem_store_op_for_accumulator` 模板推导失败

**现象**:
```
error: no instance of function template "cutlass::epilogue::collective::detail::sm90_get_smem_store_op_for_accumulator"
matches the argument list
```

**根因**: cutlass 版本不匹配。
- vLLM 使用 cutlass v4.4.2
- flash-attn 的 submodule 指向 cutlass v3.9 (commit 62750a2b)
- FA3 代码中 `sm90_get_smem_store_op_for_accumulator` 函数签名在两个版本间不兼容
- 最初将 flash-attn 的 csrc/cutlass 链接到了 vLLM 的 v4.4.2 版本

**修复**: 准备两份 cutlass：
```
/opt/external-deps/cutlass/      → v4.4.2 (vLLM 用)
/opt/external-deps/cutlass-fa/   → v3.9   (flash-attn 用)
```

flash-attn 的符号链接指向 cutlass-fa：
```dockerfile
RUN rm -rf /opt/external-deps/flash-attn/csrc/cutlass && \
    ln -s /opt/external-deps/cutlass-fa /opt/external-deps/flash-attn/csrc/cutlass
```

**教训**: 不同组件可能依赖同一个库的不同版本。flash-attn 有自己的 cutlass submodule 是有原因的。

---

### E03: Python 3.12 `ensurepip` 失败

**现象**:
```
ModuleNotFoundError: No module named 'distutils'
```
发生在 `python3.12 -m venv /opt/venv` 时。

**根因**: Python 3.12 完全移除了 `distutils`（PEP 632），而 `ensurepip` 内部依赖 distutils。

**修复**: 使用 `--without-pip --copies` 创建 venv，通过 get-pip.py 引导安装 pip：
```dockerfile
RUN python${PYTHON_VERSION} -m venv --without-pip --copies /opt/venv \
    && curl -sS ${GET_PIP_URL} | /opt/venv/bin/python3 - \
        --index-url ${PIP_INDEX_URL} --trusted-host mirrors.aliyun.com \
    && /opt/venv/bin/pip install --quiet uv
```

**预防**: Python 3.12+ 环境下，避免使用 `ensurepip`。

---

### E04: `python3 -m pip` 报 `No module named pip`（但 pip 已安装）

**现象**: 通过 `ln -s /opt/venv/bin/python3 /usr/bin/python3` 创建系统 python 后，
`python3 -m pip` 报错找不到 pip。

**根因**: 双重符号链接链 `/usr/bin/python3 → /opt/venv/bin/python3 → /usr/bin/python3.12`。
当 `python3 -m pip` 运行时，Python 检测到自己不在 venv 中（因为最终指向系统 python），
于是去系统路径找 pip 而非 venv 中的 pip。

**修复**:
1. venv 创建时加 `--copies`（复制二进制而非符号链接）
2. 验证时使用 `/opt/venv/bin/pip --version` 而非 `python3 -m pip --version`

**预防**: 不要用 `python3 -m pip` 验证，用绝对路径 `/opt/venv/bin/pip`。

---

### E05: FlashInfer wheel (2.1GB) 下载损坏

**现象**: `flashinfer-jit-cache` 安装时报 wheel 损坏或校验失败。

**根因**: FlashInfer 预编译 wheel 约 2.1GB，通过 Clash 代理下载时 SSL 连接中断。

**修复**: 将 FlashInfer 安装改为可选（RTX 3090 使用 FA2 后端，不需要 FlashInfer）：
```dockerfile
ARG INSTALL_FLASHINFER=false
RUN if [ "$INSTALL_FLASHINFER" = "true" ]; then \
        uv pip install --system flashinfer-jit-cache==${FLASHINFER_VERSION} ...; \
    else \
        echo "Skipping FlashInfer installation"; \
    fi
```

**预防**: 评估是否真的需要 FlashInfer。RTX 3090 (SM86) 使用 FA2 后端，无需 FlashInfer。

---

### E06: 预克隆的 cutlass 被 `rm -rf .deps` 删除

**现象**: 编译时找不到外部依赖源码。

**根因**: setup.py 执行 `rm -rf .deps` 清理 cmake FetchContent 缓存。
最初将预克隆源码放在 `/workspace/.deps/` 下，被一起删除。

**修复**: 将 COPY 目标改到 `/opt/external-deps/`（不受 setup.py 清理影响）。

---

### E07: `FileExistsError: './vllm/vllm_flash_attn/cute'`

**现象**: build 阶段 `setup.py bdist_wheel` 失败：
```
FileExistsError: [Errno 17] File exists: './vllm/vllm_flash_attn/cute'
```

**根因**: 裸金属构建时 cmake install 在 `vllm/vllm_flash_attn/cute` 创建了符号链接
（指向 FetchContent 缓存）。Docker `COPY . .` 将此损坏的符号链接复制到容器中。
当 setup.py 尝试 `os.makedirs` 时，发现路径已被符号链接占用。

**修复**:
1. 删除源码中的残留符号链接：`rm ./vllm/vllm/vllm_flash_attn/cute`
2. 添加到 `.dockerignore`：`vllm/vllm_flash_attn/cute`

**预防**: 构建前检查源码中是否有 cmake 安装残留的符号链接。

---

### E08: Docker COPY 不跟随符号链接

**现象**: external-src 中的符号链接（如 `flash-attn/csrc/cutlass → cutlass`）
在容器内变成损坏链接。

**根因**: Docker COPY 不跟随符号链接，只复制链接本身。

**修复**:
1. 在 Dockerfile 中用 RUN 创建符号链接（而非依赖宿主机的符号链接）
2. 或使用 `cp -r` 替代符号链接

---

### E09: bitsandbytes 下载失败

**现象**: `uv pip install bitsandbytes` 超时。

**根因**: bitsandbytes 包约 800MB，通过代理下载不稳定。

**修复**: 使用与 PyTorch 相同的重试循环 + uv cache 积累策略。

---

### E10: Docker config.json 代理注入导致 TLS 错误

**现象**:
```
apt-get 报 Error reading from server [IP: 127.0.0.1 7890]
或 pip 报 ConnectionError/SSLError
```

**根因**: `~/.docker/config.json` 中的 `proxies.default` 配置会被 Docker 自动注入到所有构建容器的环境变量中。即使 shell 层面已 unset 代理，Docker 客户端的自动注入仍会生效。

**修复**: manage.sh 现在显式传入空值覆盖: `--build-arg http_proxy= --build-arg https_proxy= --build-arg ftp_proxy=`

**预防**: 检查 `cat ~/.docker/config.json | jq '.proxies'`，确认 manage.sh build 已包含空值 build-arg。

---

### E11: flash-attn 源码 URL 错误 — Dao-AILab vs vllm-project fork

**现象**:
```
git clone Dao-AILab/flash-attention 成功，但 checkout f5bc33cfc0 失败:
error: pathspec 'f5bc33cfc0' did not match any file(s) known to git
```

**根因**: vLLM 使用自己的 flash-attention fork (vllm-project/flash-attention)，而非 Dao-AILab 上游。commit f5bc33cfc0 只存在于 vllm-project fork。

**修复**: 使用正确的仓库: `git clone https://github.com/vllm-project/flash-attention.git`

**预防**: 检查 vLLM 源码中的 CMakeLists.txt 或 .gitmodules 确认 flash-attn 的实际来源。

---

### E12: setuptools_scm 版本检测失败 — gitdir 引用

**现象**:
```
setuptools_scm 无法从 .git 获取版本信息，version 落为 "0.0.0"
```

**根因**: Docker 构建上下文是 `./vllm/`（git submodule）。子模块的 `.git` 文件是 gitdir 引用（指向 `../.git/modules/vllm`），在 Docker COPY 后此引用失效。

**修复**: manage.sh 自动通过 `git describe` 获取版本字符串，并通过 `--build-arg SETUPTOOLS_SCM_PRETEND_VERSION` 传入 Dockerfile。

**预防**: 升级 vLLM 版本后运行 `./manage.sh build`（自动处理），无需手动设置。

---

### E13: qutlass 仓库 URL 变更 — deepseek-ai/Qutlass 已删除

**现象**:
```
git clone deepseek-ai/Qutlass → fatal: repository not found
```

**根因**: deepseek-ai/Qutlass 仓库已被删除。社区 fork IST-DASLab/qutlass 保留了相同代码。

**修复**: 使用新 URL: `git clone https://github.com/IST-DASLab/qutlass.git`

**预防**: 仓库迁移在开源社区经常发生。prepare-src 失败时检查上游仓库是否仍存在。

### E14: FA3 链接失败: `No space left on device`

**现象**: csrc-build 编译全部完成 (339/339)，但链接 FA3 (Flash Attention 3 Hopper) 时失败:
```
[339/339] Linking CXX shared module vllm-flash-attn/_vllm_fa3_C.abi3.so
FAILED: [code=1] vllm-flash-attn/_vllm_fa3_C.abi3.so
/usr/bin/ld: final link failed: No space left on device
```

**根因**: v0.21.0 新增 FA3 (Hopper sm_90) 编译目标，编译产物 + 链接临时文件需要大量磁盘空间。Docker BuildKit 的构建缓存也会占用空间（每次构建约 70 GB 缓存）。

**修复**:
```bash
# 清理构建缓存（释放 ~50-70 GB）
sudo docker builder prune --all -f
# 删除旧版本镜像（释放 ~20 GB）
sudo docker rmi vllm-qwen36:rtx3090-sm86
# 确认可用空间 ≥50 GB
df -h /
```

**预防**: 构建前确保可用空间 ≥50 GB。manage.sh build 已添加自动磁盘空间检查（警告级别，不阻止构建）。

---

### E15: flashinfer-jit-cache 下载卡住

**现象**: vllm-base 阶段安装 `flashinfer-jit-cache` 时长时间无输出:
```
#54 [vllm-base 11/22] RUN ... uv pip install --system flashinfer-jit-cache==0.6.8.post1 --extra-index-url https://flashinfer.ai/whl/cu130
#54 ...
```

**根因**: flashinfer.ai 服务器从国内网络访问受限，TLS 连接可能超时或阻塞。

**修复**: 在 Dockerfile 中设置 `INSTALL_FLASHINFER=false`（默认值）。RTX 3090 (sm_86) 使用 FlashInfer JIT 运行时编译，不需要预下载 cubin 缓存:
```dockerfile
ARG INSTALL_FLASHINFER=false
```

**预防**: 如果需要预下载 cubin 缓存（减少首次启动时间），构建前测试 flashinfer.ai 可达性: `curl -I https://flashinfer.ai/whl/cu130/`。

---

## 二、网络与代理问题

### N01: Docker Hub 拉取镜像超时

**现象**: `failed to authorize: DeadlineExceeded: dial tcp ... i/o timeout`

**根因**: 国内直连 Docker Hub 不稳定。

**解决**: Docker daemon 配置代理 + BuildKit 内置重试：
```bash
# /etc/systemd/system/docker.service.d/proxy.conf
[Service]
Environment="HTTPS_PROXY=http://127.0.0.1:7890"

# 构建时
sudo -E DOCKER_BUILDKIT=1 docker build ...
```

**教训**: 不要用 crane/skopeo 预拉取大镜像，BuildKit 的重试机制更可靠。

---

### N02: 国内 Docker 镜像加速源不可用

**已测试 (2026-05)**:
| 源 | 结果 |
|----|------|
| dockerhub.icu | HTTP 200 但数据损坏 |
| docker.1ms.run | HTTP 401 |
| docker.m.daocloud.io | HTTP 401 |
| docker.nju.edu.cn | HTTP 403 |

**结论**: 直接用代理 + BuildKit，不浪费时间在加速源上。

---

### N03: 构建时代理 vs Daemon 代理

| 类型 | 配置位置 | 用途 |
|------|---------|------|
| Daemon 代理 | `/etc/systemd/system/docker.service.d/proxy.conf` | 拉取基础镜像 |
| 构建继承 | Docker daemon 配置自动继承 | RUN 内的 apt/pip/uv 下载 |

注意：本方案不在 `--build-arg` 中传代理，而是依赖 Docker daemon 配置自动继承。
因为 GitHub git clone 需要代理，而国内 apt/PyPI 镜像走代理也正常。

---

### N04: 构建必须 `--network=host`

不加 `--network=host` 时，容器内 `127.0.0.1` 指向容器自身而非宿主机，
代理不可达，所有网络操作失败。

---

## 三、运行时问题

### R01: 容器启动后 API 不可达

**排查**:
```bash
./manage.sh logs primary  # 查看 Primary 日志（或 secondary）
./manage.sh status    # 查看状态

# 常见原因:
# 1. 模型路径不存在 → 检查 .env 中 VLLM_MODEL_PATH
# 2. GPU 被占用 → nvidia-smi 检查，停掉裸金属实例
# 3. YAML 配置错误 → vllm --config 会校验参数
# 4. 首次启动慢 → torch.compile 编译中，等待 3-5 分钟
```

---

### R02: CUDA out of memory

```bash
# 检查 GPU 使用
nvidia-smi

# 降低 GPU 显存: 编辑 profile YAML 中 gpu_memory_utilization (当前默认 0.97)
#   ./manage.sh edit primary (或 secondary)
# 降低上下文: 编辑 profile YAML 中 max_model_len
```

---

### R03: NCCL Worker 初始化卡死（SYS 拓扑必遇） ⚠️ 高频

> **这是多 GPU 服务器部署最容易遇到且最难排查的问题。**
> 症状表现为服务永远无法就绪，但日志无明显报错，容易误判为"编译慢"而长时间等待。

**现象**:
- `./manage.sh start` 后日志显示 NCCL 版本信息后不再输出
- Worker 进程持续 **200% CPU**，GPU 显存仅 **~452MiB**（模型未加载）
- 等待数小时无变化，无报错、无超时
- 容器不会 crash，健康检查持续失败

**根因**:
服务器的 GPU PCIe 拓扑为 **SYS**（GPU 分布在不同 CPU socket / NUMA 节点上）。
NCCL 初始化时 `ncclCommInitRank` 尝试建立 PCIe P2P 直传通道，跨 NUMA 节点的
P2P 不可靠导致 NCCL 内部 busy-wait 死循环。RTX 3090 无 NVLink，完全依赖 PCIe。

**一分钟诊断**:
```bash
# 1. 检查 GPU 拓扑（部署前就应该做）
nvidia-smi topo -m

# 看到 SYS = 跨 NUMA，必须设置 NCCL_P2P_DISABLE=1
# 看到 PIX/NODE = 同 PCIe 总线，不需要设置

# 2. 确认 Worker 是否卡死
sudo docker compose ps                    # 容器 running 但 unhealthy
sudo docker compose exec vllm-primary ps aux # worker 进程 200% CPU
nvidia-smi                           # GPU 显存仅 ~452MiB
```

**修复**: 在 `.env` 中添加一行:
```bash
NCCL_P2P_DISABLE=1
```

然后重启: `./manage.sh restart all`

**拓扑类型速查**:

| `nvidia-smi topo -m` 显示 | 含义 | 是否需要禁用 P2P |
|---|---|---|
| `PIX` | 同一 PCIe 交换机 | 不需要 |
| `PXB` | 同 PCIe Host Bridge | 不需要 |
| `NODE` | 同 NUMA 节点 | 不需要 |
| **`SYS`** | **跨 NUMA 节点 / 跨 CPU socket** | **必须禁用** |

**为什么本机不需要但远程需要？**
同一份镜像和配置，本机拓扑为 `NODE`（两卡在同一 CPU socket 下），P2P 正常。
远程服务器拓扑为 `SYS`（多卡跨 NUMA），P2P 路径不可靠。`NCCL_P2P_DISABLE=1`
改用 SHM（CPU 共享内存中转），延迟略增但 decode 速度影响 <5%。

---

### R04: torch.compile 缓存丢失

**现象**: 每次启动都要 3-5 分钟编译。

**检查**: Docker named volume 是否存在：
```bash
sudo docker volume ls | grep vllm
```

**注意**: `docker compose down -v` 会删除 volume，使用 `docker compose down` 保留。

---

### R05: 权限问题 (`permission denied`)

**现象**: `manage.sh` 中 `docker compose` 命令报权限错误。

**修复**: `manage.sh` 中已使用 `sudo docker compose`。
或添加用户到 docker 组：`sudo usermod -aG docker $USER`

---

### R06: 上下文容量远低于预期 / enable_prefix_caching 配置 ⚠️ 高频

> **Qwen3.6 混合架构模型特有问题。** 配置不当会导致上下文容量从 256K 骤降至不足 100K。
> 所有 profile 默认已配置 `enable_prefix_caching: false` 以避免此问题。

**现象**:
- 日志显示 `GPU KV cache size: 95,648 tokens` 但 `max_model_len` 设为 200K+
- 以为 KV cache 不足，实际不影响推理（95,648 是跨组 slot 计数，非实际上限）
- 或设置 `enable_prefix_caching: true` 后 KV cache 容量大幅缩水

**根因分析**:

Qwen3.6-27B 使用混合架构：**48 层 DeltaNet (linear_attention) + 16 层 full_attention = 64 层**。

`enable_prefix_caching` 控制了 Mamba cache mode 的自动选择：

| 配置 | mamba_cache_mode | DeltaNet state 分配方式 | 效果 |
|------|-----------------|----------------------|------|
| `enable_prefix_caching: true` | `"align"` | DeltaNet state 使用 paged KV cache | per-token 开销暴增，上下文容量骤降 |
| `enable_prefix_caching: false` | `"none"` | DeltaNet state 单独分配 | 仅 16 层 full_attention 使用 paged KV cache，容量充足 |

vLLM 代码路径（`vllm/model_executor/models/config.py`）：
```python
# 简化逻辑（完整路径见上方文件）
# enable_prefix_caching=true  → mamba_cache_mode="align"
#   Qwen3.6 不支持 mamba prefix caching，走 align 路径
#   DeltaNet state 使用 paged KV cache, per-token 开销增大
# enable_prefix_caching=false → mamba_cache_mode="none"
#   DeltaNet state 单独分配, 仅 16 层 full_attention 使用 paged KV
```

**实测数据（2x RTX 3090, gpu_memory_utilization=0.97, FP8 KV, 多模态）**:

| 配置 | Available KV cache | 报告 tokens | 实测并发 | 实际可用上下文 |
|------|-------------------|------------|---------|-------------|
| prefix_caching=false, max_len=262144 | ~5.89 GiB | 95,648 | 1.43x | **256K (实测 261,947 tokens)** |
| prefix_caching=false, max_len=204800 | ~5.89 GiB | 95,648 | 1.83x | **200K (实测 204,407 tokens)** † |

> 注意: 日志中 `GPU KV cache size: 95,648 tokens` 是跨所有 KV cache 组的 slot 计数，
> **不代表实际上下文上限**。实际可用容量由 `Maximum concurrency` 指标反映。
> † max_len=204800 仅为测试对比数据，当前所有 profile 均未使用此值。实际 profile 使用 262144/143360/155648/131072。

**Trade-off**:
- `prefix_caching=false`: 失去 prefix caching 加速（重复前缀的 TTFT 优化），获得最大上下文容量
- `prefix_caching=true`: 获得 prefix caching 加速，但上下文容量大幅缩水

**推荐配置**:

| Profile | max_model_len | prefix_caching | 设计意图 |
|---------|--------------|----------------|---------|
| think | 262,144 (256K) | false | 通用思考，需要长上下文 |
| code | 143,360 (140K) | false | 精确编码 (bf16+MTP, ~132K 可用上下文) |
| agent-fast | 143,360 (140K) | false | Agent 非思考，低延迟工具调用 (Primary 默认) |
| agent-thinking | 143,360 (140K) | false | Agent 思考模式，多轮推理 (Secondary 默认) |
| instruct | 131,072 (128K) | false | 快速问答，非思考模式不需要极长上下文 |
| test-spec | 155,648 (152K) | false | 测试专用入口，参数可调（见头部注释速查） |

**验证上下文容量**:
```bash
# 1. 检查启动日志
sudo docker logs vllm-primary 2>&1 | grep -i "Maximum concurrency"
# 预期: 值取决于 profile (256K→1.43x, 140K→1.26x, 128K→更高)

# 2. 测试长上下文请求
curl http://localhost:8089/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3.6-27B-FP8","messages":[{"role":"user","content":"<长文本>"}],"max_tokens":50}'
```

---

### R07: gpu_memory_utilization 过高导致 OOM

**现象**: 启动时 KV cache 分配阶段 CUDA OOM 崩溃。

**实测极限值**（2x RTX 3090 24GB）:

| gpu_memory_utilization | 结果 |
|----------------------|------|
| 0.95 | 正常（默认安全值） |
| 0.97 | 正常（RTX 3090 极限，实测通过） |
| 0.985 | OOM 崩溃 |

**修复**: 如果 0.97 启动 OOM，降至 0.95：
```bash
# 编辑对应服务的 profile YAML 中 gpu_memory_utilization:
# ./manage.sh edit primary   # 编辑 Primary profile
# 或直接编辑 profiles/<name>.yaml 中 gpu_memory_utilization: 0.95
# 然后重启: ./manage.sh restart primary
```

当前 agent-fast/agent-thinking 默认 `gpu_memory_utilization: 0.96`（预置余量确保 n=2 兼容），其他 profile 为 0.97，均已通过实测验证。

---

### R08: 双服务启动问题 ⚠️ 新增

**现象**: `./manage.sh start all` 时两个服务冲突或启动失败。

**常见原因与修复**:

1. **GPU 编号冲突**:
   manage.sh 启动时自动校验 GPU ID 不重叠，如重叠会报错:
   ```
   错误: GPU ID 重叠: 1
     VLLM_PRIMARY_GPU_IDS=0,1
     VLLM_SECONDARY_GPU_IDS=1,2
     每张 GPU 只能分配给一个服务
   ```
   **修复**: 检查 `.env` 中 `VLLM_PRIMARY_GPU_IDS` 和 `VLLM_SECONDARY_GPU_IDS` 确保无重叠。使用 `nvidia-smi` 检查 GPU 当前占用情况。

2. **端口冲突**:
   ```
   错误: Bind for 0.0.0.0:8089 failed: port is already allocated
   ```
   **修复**: 检查端口占用 `ss -tlnp | grep -E '8089|8099'`，修改 `.env` 中的 `VLLM_PRIMARY_HOST_PORT` 或 `VLLM_SECONDARY_HOST_PORT`。

3. **torch.compile 缓存并发**:
   首次启动（冷缓存）时如果两个服务同时编译，可能出现缓存写入竞争。
   **修复**: `start all` 已实现顺序启动（先 primary → wait_ready → 再 secondary），确保第二个服务命中热缓存。**不要**直接使用 `docker compose up -d` 同时启动两个服务。

4. **FlashInfer JIT 并发**:
   各服务使用独立的 FlashInfer 缓存目录，无并发写入风险。

5. **容器名冲突**:
   ```
   错误: The container name "/vllm-primary" is already in use
   ```
   **修复**: 先执行 `./manage.sh stop all`，再启动。

6. **Docker daemon 重启自动恢复**:
   `restart: unless-stopped` 确保系统重启后自动恢复服务。Primary/Secondary 各自使用独立的 FlashInfer 缓存目录（`./flashinfer-cache/primary/`、`./flashinfer-cache/secondary/`），无并发写入风险，可安全并行启动。重启后建议执行 `./manage.sh restart all` 以恢复正确顺序。

---

### R09: 4 卡 GPU 拓扑检查 ⚠️ 新增（部署前必查）

**现象**: 4 卡服务器上 NCCL 初始化卡死或通信异常。

**检查步骤**:

```bash
nvidia-smi topo -m
```

**分析拓扑表**:

| GPU0 | GPU1 | GPU2 | GPU3 | 说明 |
|------|------|------|------|------|
| X | PIX | SYS | SYS | GPU 0,1 同 PCIe（理想） |
| SYS | X | SYS | SYS | GPU 0,1 跨 NUMA（需注意） |

**关键判断**:
- `PIX` / `NODE`: 同 PCIe 或同 NUMA，NCCL P2P 可用，性能最佳
- `SYS`: 跨 NUMA 节点，必须设置 `NCCL_P2P_DISABLE=1`

**双服务特殊说明**:
每个容器通过 `CUDA_VISIBLE_DEVICES` 只能看到分配给自己的 2 张 GPU。例如 vllm-primary 容器只能看到 GPU 0,1，NCCL 通信仅限这两张卡之间。因此：
- 只要 GPU 0,1 之间和 GPU 2,3 之间的拓扑**不是 SYS**，双服务可以正常工作
- 如果 GPU 0,1 之间显示 SYS（例如 0 和 1 在不同 NUMA 节点），则该对需要 `NCCL_P2P_DISABLE=1`
- 跨服务的 GPU（如 GPU 1 和 GPU 2 之间）不会互相通信，拓扑关系无关

**GPU 分配优化**: 如果默认分配 (0,1)(2,3) 的拓扑不理想，可以根据实际拓扑重新分配。例如如果实际拓扑是 (0,2) 和 (1,3) 分别在同一 PCIe 域，可以修改 `.env`:
```
VLLM_PRIMARY_GPU_IDS=0,2
VLLM_SECONDARY_GPU_IDS=1,3
```

---

### R10: Speculative Decoding 启用后 OOM ⚠️ 注意: R10 132K 并发数据在 expandable_segments:True 环境下收集（短 prompt，未测长上下文），修复后完整数据见 R13

**现象**:
```
ValueError: To serve at least one request with the models's max seq len (262144),
(8.66 GiB KV cache is needed, which is larger than the available KV cache memory (6.51 GiB).
Based on the available memory, the estimated maximum model length is 196000.
```

**根因**: Qwen3.6 模型内置 1 层 MTP 头（`mtp_num_hidden_layers=1`）。启用 `speculative-config` (method=mtp) 后，MTP draft model 加载额外权重到 GPU，大幅压缩 KV cache 预算。

**显存实测** (2x RTX 3090, gpu_memory_utilization=0.97):

| 场景 | KV cache 预算 | bf16 KV cache 上限 | fp8 KV cache 上限 |
|------|-------------|-------------------|------------------|
| 无 MTP | ~6.36 GiB/卡 | 256K+ | 256K+ |
| MTP 启用 | ~6.51 GiB/卡 | **~195K** | 256K |

**修复方案**:

根据需求选择组合:

| 目标 | kv_cache_dtype | max_model_len | 是否可行 | KV/GPU | 并发 |
|------|---------------|---------------|---------|--------|------|
| MTP + 高精度 KV | `auto` (bf16) | 155648 (152K) | ✓ | ~5.0 GiB | 1.25x |
| MTP + bf16 + 180K | `auto` (bf16) | 184320 (180K) | ✓ | ~5.6 GiB | 1.06x |
| MTP + bf16 + 195K | `auto` (bf16) | 195968 (195K) | ✓ (极限) | ~6.5 GiB | 1.00x |
| MTP + 极限上下文 | `fp8_e4m3` | 262144 (256K) | ✓ | ~4.3 GiB | 1.49x |
| MTP + bf16 + 200K | `auto` (bf16) | 204800 (200K) | ✗ OOM | 需 6.8 GiB | - |
| MTP + bf16 + 256K | `auto` (bf16) | 262144 (256K) | ✗ OOM | 需 8.7 GiB | - |

> num_speculative_tokens 实测、132K 并发压力测试、无 Thinking 完整测试数据已归档至 `docs/benchmark-results.md` R10 章节。
> 注意: R10 数据在 expandable_segments:True 环境下收集，修复后完整数据见 R13。
> 部署推荐: 1-2 并发选 bf16+n=2; 3-4 并发选 bf16+n=1。详见 `docs/benchmark-results.md` 第一节"推荐配置摘要"。

---

### R11: Profile 切换方式 (已更新)

**历史背景**: 早期版本支持 `./manage.sh restart primary <profile>` 临时切换 profile，但因 `sudo env_reset` 导致环境变量丢失，实际不生效（UI 显示新 profile 但容器使用旧 profile）。

**当前方案**: 已移除临时切换功能，Profile 切换统一通过 `set-profile` + `restart` 两步操作:
```bash
# 第 1 步: 写入 .env
./manage.sh set-profile primary new-profile

# 第 2 步: 重启生效
./manage.sh restart primary
```

**验证**: 检查容器内实际挂载的配置文件:
```bash
sudo docker exec vllm-primary cat /etc/vllm/config.yaml | tail -5
```

---

### R12: Prefill OOM — expandable_segments 与 CUDA Graph 不兼容 ⚠️ 关键发现

> **这是启用 CUDA Graph + torch.compile 时最容易踩的坑。**
> 症状表现为长上下文 prefill 阶段 OOM，但实际不是显存不足，而是内存分配器与 CUDA Graph 冲突。

**现象**:
- 132K + max_num_batched_tokens=8192: prefill 在 ~7.5K tokens 时 OOM
- 152K + max_num_batched_tokens=8192: prefill 在 ~60K tokens 时 OOM
- 错误信息: `Tried to allocate 136 MiB. GPU 0 has a total capacity of 23.57 GiB of which 129.31 MiB is free.`
- 看似显存不足，实际是地址空间损坏（expandable_segments 重映射导致 CUDA Graph 地址失效）

**根因**:

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 启用 PyTorch VMM（Virtual Memory Management）分配器。
VMM 分配器在运行时动态合并/拆分内存段（segment），会**重新映射虚拟内存页**。
当 `compilation_config.mode=VLLM_COMPILE` + `cudagraph_mode=FULL_DECODE_ONLY` 时，vLLM 在 decode 路径使用 CUDA Graph 捕获 GPU 操作序列。
CUDA Graph 捕获时记录的是**绝对内存地址**，VMM 的重新映射使这些地址失效，导致后续 prefill 分配时误报 OOM。

相关 GitHub Issues: vllm-project/vllm #29544（expandable_segments 导致初始化失败，已过期关闭，未修复）

**重要区分**:
- PR #40812（已包含在本地 vLLM v0.21.0+ 源码中）修复的是 **cumem sleep mode memory pool** 与 expandable_segments 的兼容性（`cumem.py` 中自动在 pool 上下文中临时禁用 expandable_segments），**不涉及 CUDA Graph**
- CUDA Graph + expandable_segments 的架构性不兼容（VMM 动态映射 vs CUDA Graph 静态地址锁定）在 PyTorch 和 vLLM 中**均未修复**
- vLLM 通过 `max_split_size_mb=20`（PR #41268）替代 expandable_segments 来缓解内存碎片，因此生产环境应保持 `PYTORCH_CUDA_ALLOC_CONF: expandable_segments:False`

**一分钟诊断**:
```bash
# 1. 检查当前 PYTORCH_CUDA_ALLOC_CONF 配置
sudo docker exec vllm-primary env | grep PYTORCH_CUDA_ALLOC_CONF
# 如果输出包含 expandable_segments:True → 这就是问题根源

# 2. 确认使用了 CUDA Graph
sudo docker logs vllm-primary 2>&1 | grep -i "cuda graph"
# 如果看到 "Capturing CUDA graphs" → 与 expandable_segments 冲突

# 3. 验证: 实际显存是否真的不足
nvidia-smi
# 如果有数 GiB 空闲 → 不是真正的显存不足
```

**修复**:

在 `docker-compose.yml` 的共享环境变量中显式禁用 expandable_segments:

```yaml
x-vllm-shared-env: &vllm-shared-env
  PYTORCH_CUDA_ALLOC_CONF: expandable_segments:False    # 显式禁用，避免与 CUDA Graph 冲突
```

当前部署已包含此修复。

**修复前后对比** (2×RTX 3090, bf16 KV + MTP n=1, max_num_batched_tokens=8192):

| 配置 | 修复前 prefill 上限 | 修复后 prefill 上限 |
|------|-------------------|-------------------|
| 132K (max_model_len=135168) | ~7.5K tokens (不可用) | ~130K tokens |
| 152K (max_model_len=155648) | ~60K tokens | ~150K tokens |

**Decode 性能影响**:

移除 expandable_segments 后 decode 吞吐有 ~20-27% 回退（VMM 的动态合并原本减少了内存碎片）:

| 并发 | 修复前 (tok/s) | 修复后 (tok/s) | 回退 |
|------|--------------|--------------|------|
| c=1 Decode | 37 | 27 | -27% |
| c=4 Wall | 219 | 170 | -22% |

> 此回退是可接受的代价: expandable_segments 在 CUDA Graph 环境下不安全，会随机 OOM。
> 对于 RTX 3090 24GB 显存极度紧张（0.97 utilization）的场景，VMM 重映射是致命的。

**关键概念区分**:

| 概念 | 含义 | 受什么影响 |
|------|------|-----------|
| KV cache 分配上限 | decode 阶段最多能存储多少 token 的 KV | kv_cache_dtype, max_model_len, gpu_memory_utilization |
| Prefill 输入上限 | 单次 prefill 能处理多少 token | max_num_batched_tokens, 显存碎片, CUDA Graph 兼容性 |

例如 bf16+MTP ~195K 是 KV cache 分配上限，不代表能一次性 prefill 195K tokens。
实际 prefill 能力需要通过长上下文请求实测验证。

**预防措施**:
- 任何使用 `compilation_config.mode=VLLM_COMPILE` + `cudagraph_mode` 的 vLLM 部署都应避免 `expandable_segments:True`
- 不要从社区博客或默认配置直接复制 `PYTORCH_CUDA_ALLOC_CONF` 设置
- `cudagraph_capture_sizes` 显式限制无额外收益（默认行为已足够），添加不当反而引入额外 decode 回退

---

### R13: expandable_segments:False 修复后完整重测

> 完整基准测试数据已迁移至 `docs/benchmark-results.md` R13 章节。
> 关键结论: bf16+n=1 132K c=4 达 167.7 tok/s, n=2 需 gpu_util=0.96, 上下文质量 100%。
> 部署建议见 `docs/benchmark-results.md` 第一节"推荐配置摘要"。

---

### R14: 152K 上下文完整验证

> 完整基准测试数据已迁移至 `docs/benchmark-results.md` R14 章节。
> 关键结论: prefill 上限 155,137 tok (n=1/n=2 一致), n=1 c=4 达 200 tok/s, n=2 c=1-3 吞吐优于 n=1, 并发 prefill 60K×c=4=240K 仍 0 OOM。
> 部署推荐见 `docs/benchmark-results.md` 第一节"推荐配置摘要"。

| 错误关键词 | 问题编号 | 一句话总结 |
|-----------|---------|-----------|
| `numeric_types.h` | E01 | flash-attn cutlass 子模块为空 |
| `sm90_get_smem_store_op` | E02 | cutlass v4.4.2 vs v3.9 不兼容 |
| `No module named 'distutils'` | E03 | Python 3.12 移除了 distutils |
| `No module named pip` | E04 | 双重符号链接破坏 venv 检测 |
| `flashinfer.*corrupted` | E05 | 2.1GB wheel 下载损坏 |
| `rm -rf .deps` 删除源码 | E06 | COPY 目标路径不当 |
| `FileExistsError: cute` | E07 | 源码残留损坏符号链接 |
| `broken symlink` | E08 | Docker COPY 不跟随符号链接 |
| `bitsandbytes timeout` | E09 | 大包下载超时 |
| `SSLError` / `ConnectionError` | E10 | Docker config.json 代理注入 |
| `pathspec 'f5bc33cfc0'` | E11 | flash-attn 应使用 vllm-project fork |
| `setuptools_scm "0.0.0"` | E12 | gitdir 引用在 Docker COPY 后失效 |
| `repository not found` Qutlass | E13 | deepseek-ai/Qutlass 已删除，用 IST-DASLab fork |
| `No space left on device` FA3 | E14 | v0.21.0 FA3 编译磁盘空间不足 |
| `flashinfer.ai` 超时 | E15 | 国内网络访问 flashinfer.ai 受限 |
| `i/o timeout` Docker Hub | N01 | 国内网络不稳定 |
| `exit code 100` apt-get | N03 | 容器内代理不可达 |
| Worker 200% CPU 卡死 | **R03** | **SYS 拓扑需禁用 P2P（NCCL_P2P_DISABLE=1）** |
| NCCL 通信超时 | **R03** | **SYS 拓扑需禁用 P2P（NCCL_P2P_DISABLE=1）** |
| 上下文容量不足/95K tokens | **R06** | **prefix_caching 触发 mamba align 模式（设为 false）** |
| KV cache 分配 OOM | **R07** | **gpu_memory_utilization 过高（RTX 3090 极限 0.97）** |
| 双服务启动冲突 | **R08** | **GPU/端口冲突或缓存并发（顺序启动）** |
| 4 卡拓扑问题 | **R09** | **nvidia-smi topo -m 检查 GPU 对间是否 SYS** |
| Spec Decoding OOM (MTP) | **R10** | **MTP draft model 压缩 KV cache 预算至 ~6.5 GiB，需调整 kv_cache_dtype 或 max_model_len** |
| Profile override 不生效 | **R11** | **sudo 不继承 export 环境变量，必须先 set-profile 再 restart** |
| Prefill OOM (长上下文) | **R12** | **expandable_segments:True 与 CUDA Graph 不兼容，显式设为 expandable_segments:False** |
| n=2 OOM (gpu_util=0.97) | **R13** | **MTP n=2 需 gpu_memory_utilization=0.96（0.97 下 OOM，释放 ~240 MiB 连续显存即可运行）** |

---

## 五、预防检查清单

### 部署前检查（目标服务器）

```bash
# 1. 检查 GPU PCIe 拓扑（最容易被忽略，但影响最大）
nvidia-smi topo -m
# GPU 间显示 SYS → .env 必须设置 NCCL_P2P_DISABLE=1
# GPU 间显示 PIX/NODE → 不需要设置

# 2. 检查 NVIDIA 驱动版本
nvidia-smi | grep "Driver Version"
# 确认 >= 565

# 2.5 检查 NVIDIA 驱动健康（start/restart 自动执行，也可手动运行）
./manage.sh check-driver
# 通过: 输出 "NVIDIA 驱动检查通过 (vXXX, Nx GPU)"
# 失败: 输出具体错误和 dmesg 排查命令

# 3. 检查 Docker GPU 访问
sudo docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu22.04 nvidia-smi

# 4. 检查 CUDA Graph 与 expandable_segments 兼容性
# CUDA Graph 启用时 PYTORCH_CUDA_ALLOC_CONF 必须为 expandable_segments:False
grep "PYTORCH_CUDA_ALLOC_CONF" docker-compose.yml
# 预期: expandable_segments:False
# 如果含 True → 必须修改（否则 prefill OOM，详见 R12）

# 5. MTP n>=2 需额外显存余量
# 使用 num_speculative_tokens>=2 的 profile 需 gpu_memory_utilization<=0.96
# 详见 docs/benchmark-results.md R13.4

# 6. 配置开机自启（可选，生产环境推荐）
sudo ./manage.sh enable-boot
# 包含: systemd 服务 + nvidia-persistenced + Docker shutdown-timeout
# 验证: sudo systemctl status vllm.service
```

### 构建前检查（构建主机）

```bash
# 1. 检查代理是否可用
curl -s --proxy http://127.0.0.1:7890 https://www.google.com > /dev/null && echo "OK" || echo "FAIL"

# 2. 检查外部依赖完整性
ls ./vllm/external-src/flash-attn/csrc/cutlass/include/cutlass/numeric_types.h
ls ./vllm/external-src/cutlass/include/cutlass/numeric_types.h
ls ./vllm/external-src/cutlass-fa/include/cutlass/numeric_types.h

# 3. 检查源码无 cmake 残留
ls -la ./vllm/vllm/vllm_flash_attn/cute 2>/dev/null && echo "WARN: stale symlink!" || echo "OK"

# 4. 检查 Docker daemon 代理
sudo systemctl show docker --property=Environment | grep PROXY

# 5. 检查磁盘空间（构建需 ~50GB 临时空间）
df -h /data
```

---

### R15: CPU 隔离 (cpuset) 配置问题

**现象**: 双服务并行时 decode 输出卡顿/掉字，单服务时正常。

**根因**: 双服务共 ~1562 个线程在全部 CPU 核心自由竞争，CPU 调度器频繁切换上下文导致 decode 延迟波动。

**诊断**:

```bash
# 1. 检查是否配置 cpuset
./manage.sh config | grep "CPU 隔离"

# 2. 查看 NUMA 拓扑
./manage.sh detect-topology

# 3. 查看当前 .env 中的 cpuset 值
grep CPUSET .env
```

**常见错误与修复**:

| 错误 | 症状 | 修复 |
|------|------|------|
| cpuset 格式错误 | `start`/`restart` 报错 "cpuset 格式无效" | 使用 `0-23,48-71` 格式（逗号分隔的编号或范围） |
| CPU 范围重叠 | `start`/`restart` 报错 "CPU 隔离范围重叠" | 每个核心只能分配给一个服务 |
| 仅一侧配置 | 警告 "仅 Primary/Secondary 配置了 cpuset" | 补全另一侧，或确认有意为之 |
| 单 NUMA 自动检测退出 | `apply-cpuset` 提示 "不自动配置" | 手动均分核心，参考提示中的示例值 |

**修复流程**:
1. 自动检测并应用: `./manage.sh apply-cpuset`
2. 多 NUMA: 基于各 GPU 的 CPU Affinity 自动对齐
3. 单 NUMA: 按核心数 50/50 手动均分
4. 生效: `./manage.sh restart all`

### R16: 重启后 vLLM 服务未自动启动

**现象**: 服务器重启后 vLLM 服务未运行。

**诊断**:

```bash
# 检查 systemd 服务状态
sudo systemctl status vllm.service

# 查看启动日志
journalctl -u vllm.service -b --no-pager

# 检查是否已启用
sudo systemctl is-enabled vllm.service

# 检查 NVIDIA 驱动是否正常
./manage.sh check-driver

# 如果驱动异常，查看内核日志
dmesg | grep -iE 'nvidia|NVRM|GPU'
```

**常见原因**:
- 未运行 `./manage.sh enable-boot` 配置开机自启
- NVIDIA 驱动异常（`check-driver` 会检测并输出排查指引）
- Docker 服务未就绪（`vllm.service` 依赖 `docker.service`）

**预防**: 运行 `sudo ./manage.sh enable-boot` 一键配置。

### R17: 关机/重启后 NVIDIA 驱动异常（掉驱动）

**现象**: 执行 `sudo shutdown` 或 `sudo reboot` 后，重启时 `nvidia-smi` 报错或 GPU 不可用。

**根因**: vLLM 容器未优雅停止，GPU 显存/CUDA 上下文未释放，驱动状态损坏。

**诊断**:

```bash
dmesg | grep -iE 'NVRM|GPU lost|GPU fallen off'
nvidia-smi
./manage.sh check-driver
```

**预防**: `sudo ./manage.sh enable-boot` 配置关机保护（systemd 在关机前自动执行 `manage.sh stop all`，优雅释放 GPU 资源）。

**修复**: 如果已掉驱动:

```bash
sudo modprobe -r nvidia_drm nvidia_modeset nvidia_uvm nvidia
sudo modprobe nvidia
./manage.sh check-driver
```

### R18: 开机后 vLLM 启动延迟 60 秒

**现象**: 开机后 vLLM 服务在 journal 中显示 "系统启动中，等待 60s..."。

**说明**: 这是预期行为。`boot-delay` 在系统 uptime < 3 分钟时自动延迟 60 秒，确保 Docker、NVIDIA 驱动等基础服务完全就绪后再启动 vLLM，避免启动风暴。系统稳定后手动 `systemctl start` 不延迟。
