#!/bin/bash
# ==============================================================================
# Qwen3.6-27B-FP8 Docker 部署管理脚本 — 双服务并行版 (Primary + Secondary)
#
# 用法:
#   ./manage.sh start [primary|secondary|all]          # 启动
#   ./manage.sh stop [primary|secondary|all]           # 停止
#   ./manage.sh restart [primary|secondary|all]        # 重启
#   ./manage.sh set-profile [primary|secondary] <name> # 切换 profile (写入 .env)
#   ./manage.sh config                                 # 查看完整配置预检
#   ./manage.sh status [primary|secondary]             # 查看运行状态
#   ./manage.sh logs [primary|secondary]               # Docker 日志
#   ./manage.sh logfiles [primary|secondary]           # 文件日志
#   ./manage.sh edit [primary|secondary]               # 编辑服务配置
#   ./manage.sh detect-topology                        # 检测 NUMA 拓扑，推荐 cpuset
#   ./manage.sh apply-cpuset [--with-ht]               # 自动配置 cpuset（默认仅物理核心）
#   ./manage.sh build                                  # 构建 Docker 镜像
#   ./manage.sh pull-base                              # 拉取 CUDA 基础镜像
#   ./manage.sh enable-boot                            # 一键配置开机自启 + 关机保护
#   ./manage.sh disable-boot                           # 移除开机自启 + 关机保护
#   ./manage.sh check-driver                           # 检查 NVIDIA 驱动健康状态
#
# 安全检查:
#   启动时自动检测 PYTORCH_CUDA_ALLOC_CONF expandable_segments 与 CUDA Graph 兼容性
#   CUDA Graph 启用时必须为 expandable_segments:False（详见 docs/troubleshooting.md R12）
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 加载 .env（如不存在则提示从模板创建）
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        echo "提示: .env 不存在，已从 .env.example 复制，请按需修改"
        cp .env.example .env
    else
        echo "错误: .env 和 .env.example 均不存在"
        exit 1
    fi
fi
set -a && source .env && set +a

# ---------- API Key 格式清理 (防御性: 去除引号和前后空格) ----------
for _key_var in VLLM_PRIMARY_API_KEY VLLM_SECONDARY_API_KEY; do
    _val="${!_key_var:-}"
    _val="${_val#\"}" ; _val="${_val%\"}"
    _val="${_val#\'}" ; _val="${_val%\'}"
    _val="${_val## }" ; _val="${_val%% }"
    printf -v "${_key_var}" '%s' "$_val"
done
unset _key_var _val

# ---------- 校验 .env 格式 ----------

if [ -z "${VLLM_PRIMARY_GPU_IDS:-}" ]; then
    echo "错误: .env 格式与当前版本不匹配"
    if [ -n "${VLLM_FAST_GPU_IDS:-}" ]; then
        echo "  检测到旧变量 VLLM_FAST_GPU_IDS (v2 格式，已重命名为 VLLM_PRIMARY_GPU_IDS)"
        echo "  请更新 .env: cp .env.example .env && 编辑配置"
    elif [ -n "${VLLM_AGENT_HOST_PORT:-}" ]; then
        echo "  检测到旧变量 VLLM_AGENT_HOST_PORT (v1 格式)"
        echo "  请更新 .env: cp .env.example .env && 编辑配置"
    elif [ -n "${VLLM_HOST_PORT:-}" ]; then
        echo "  检测到 main 单服务变量 (VLLM_HOST_PORT)"
        echo "  可能原因: 从 main 切换后 .env 未更新"
    else
        echo "  缺少关键变量 VLLM_PRIMARY_GPU_IDS"
    fi
    echo ""
    echo "修复: cp .env.example .env && 编辑 API Key 等配置"
    exit 1
fi

# 校验 GPU ID 不重叠
_gpu_overlap() {
    local primary_gpus secondary_gpus overlap
    primary_gpus="$(echo "${VLLM_PRIMARY_GPU_IDS:-0,1}" | tr ',' '\n' | sort)"
    secondary_gpus="$(echo "${VLLM_SECONDARY_GPU_IDS:-2,3}" | tr ',' '\n' | sort)"
    overlap="$(comm -12 <(echo "$primary_gpus") <(echo "$secondary_gpus"))"
    if [ -n "$overlap" ]; then
        echo "错误: GPU ID 重叠: $(echo "$overlap" | tr '\n' ',' | sed 's/,$//')"
        echo "  VLLM_PRIMARY_GPU_IDS=${VLLM_PRIMARY_GPU_IDS:-0,1}"
        echo "  VLLM_SECONDARY_GPU_IDS=${VLLM_SECONDARY_GPU_IDS:-2,3}"
        echo "  每张 GPU 只能分配给一个服务"
        exit 1
    fi
}
_gpu_overlap

# 获取系统 GPU 数量（单次执行内缓存，避免重复调用 nvidia-smi）
_get_gpu_count() {
    if [ -n "${_gpu_count_cache:-}" ]; then
        echo "$_gpu_count_cache"
        return
    fi
    _gpu_count_cache=$(sudo nvidia-smi -L 2>/dev/null | grep -c "^GPU" || echo 0)
    echo "$_gpu_count_cache"
}

# 校验 GPU ID 存在性
_validate_gpu_exists() {
    local gpu_ids="$1"
    local total_gpus
    total_gpus=$(_get_gpu_count)
    if [ "$total_gpus" -eq 0 ]; then
        echo "警告: nvidia-smi 不可用，跳过 GPU 存在性校验"
        return 0
    fi
    for id in $(echo "$gpu_ids" | tr ',' '\n'); do
        if [ "$id" -ge "$total_gpus" ] 2>/dev/null; then
            echo "错误: GPU ${id} 不存在 (系统共 ${total_gpus} 张 GPU, 编号 0-$((total_gpus - 1)))"
            return 1
        fi
    done
}

# NVIDIA 驱动健康检查（start/restart 前置检查，防止掉驱动后盲目启动容器）
# 检查链: 内核模块 → nvidia-smi → 设备节点 → GPU 计数
# 失败时通过 logger 写入 journal，并输出 dmesg 排查指引
_check_nvidia_driver() {
    local check_tag="vllm-driver-check"

    # 1. 内核模块（用 /sys/module 避免 lsmod | grep -q 在 pipefail 下的 SIGPIPE 问题）
    if ! test -d /sys/module/nvidia; then
        echo "错误: NVIDIA 内核模块未加载 — 驱动可能已掉落或未安装"
        echo "  排查: dmesg | grep -iE 'nvidia|NVRM|GPU|nouv'"
        echo "  排查: journalctl -b -t ${check_tag}"
        echo "  修复: sudo modprobe nvidia && dmesg | tail -20"
        logger -t "$check_tag" -p user.err "NVIDIA 内核模块未加载，阻止 vLLM 服务启动"
        return 1
    fi

    # 2. nvidia-smi
    if ! sudo nvidia-smi -L >/dev/null 2>&1; then
        echo "错误: nvidia-smi 执行失败 — 驱动与硬件通信异常"
        echo "  排查: dmesg | grep -iE 'NVRM|Xid|GPU|broken'"
        echo "  排查: nvidia-smi -q 2>&1 | head -20"
        echo "  排查: journalctl -b -t ${check_tag}"
        logger -t "$check_tag" -p user.err "nvidia-smi 失败，驱动异常，阻止 vLLM 服务启动"
        return 1
    fi

    # 3. 设备节点
    if ! ls /dev/nvidia* >/dev/null 2>&1; then
        echo "错误: GPU 设备节点不存在 (/dev/nvidia*) — 驱动可能未正确初始化"
        echo "  排查: ls -la /dev/nvidia*"
        echo "  排查: dmesg | grep -i 'nvidia.*device\|nvidia.*register\|nvidia-uvm'"
        echo "  排查: journalctl -b -t ${check_tag}"
        logger -t "$check_tag" -p user.err "GPU 设备节点缺失，阻止 vLLM 服务启动"
        return 1
    fi

    # 4. GPU 计数
    local gpu_count
    gpu_count=$(_get_gpu_count)
    if [ "$gpu_count" -eq 0 ]; then
        echo "错误: nvidia-smi 未检测到任何 GPU — 驱动可能已掉落"
        echo "  排查: dmesg | grep -iE 'NVRM|GPU lost|GPU fallen off'"
        echo "  排查: nvidia-smi"
        echo "  排查: journalctl -b -t ${check_tag}"
        logger -t "$check_tag" -p user.err "nvidia-smi 未检测到 GPU，阻止 vLLM 服务启动"
        return 1
    fi

    # 通过: 输出驱动信息（-i 0 限制单 GPU 输出，避免 head -1 在 pipefail 下的 SIGPIPE）
    local driver_ver
    driver_ver=$(sudo nvidia-smi -i 0 --query-gpu=driver_version --format=csv,noheader 2>/dev/null || true)
    echo "NVIDIA 驱动检查通过 (v${driver_ver:-unknown}, ${gpu_count}x GPU)"
    logger -t "$check_tag" -p user.info "驱动检查通过: v${driver_ver:-unknown}, ${gpu_count}x GPU"
    return 0
}

# 开机延迟（仅在系统 uptime < 3 分钟时延迟 60s，确保 Docker/NVIDIA 等基础服务稳定）
# 手动 systemctl start 在系统运行一段时间后不延迟
_boot_delay_if_needed() {
    local uptime_s
    uptime_s=$(cut -d' ' -f1 /proc/uptime 2>/dev/null | cut -d. -f1) || return 0
    if [ "${uptime_s:-999}" -lt 180 ]; then
        echo "系统启动中 (uptime ${uptime_s}s)，等待 60s 确保 Docker/NVIDIA 驱动稳定..."
        sleep 60
    fi
}

# 检查指定服务的 GPU 是否全部可用（用于 start all 智能决策）
_service_gpus_available() {
    local svc="$1"
    local gpu_ids
    gpu_ids="$(get_gpu_ids "$svc")"
    # 空 GPU IDs 视为不可用
    [ -n "$gpu_ids" ] || return 1
    local total_gpus
    total_gpus=$(_get_gpu_count)
    [ "$total_gpus" -gt 0 ] || return 1
    for id in $(echo "$gpu_ids" | tr ',' '\n'); do
        if [ "$id" -ge "$total_gpus" ] 2>/dev/null; then
            return 1
        fi
    done
    return 0
}

# 更新 Docker daemon.json 的 shutdown-timeout（保留其他字段）
# 用法: _update_docker_shutdown_timeout <秒数>  (0 = 移除字段)
_update_docker_shutdown_timeout() {
    local timeout="$1"
    local daemon_json="/etc/docker/daemon.json"
    local backup="${daemon_json}.bak"

    if [ ! -f "$daemon_json" ]; then
        if [ "$timeout" != "0" ]; then
            echo "{\"shutdown-timeout\": ${timeout}}" | sudo tee "$daemon_json" > /dev/null
            echo "[OK] 已创建 $daemon_json (shutdown-timeout: ${timeout}s)"
        fi
        return
    fi

    sudo cp "$daemon_json" "$backup"

    if [ "$timeout" = "0" ]; then
        if command -v jq &>/dev/null; then
            if jq 'del(.["shutdown-timeout"])' "$backup" | sudo tee "$daemon_json" > /dev/null; then
                echo "[OK] 已从 $daemon_json 移除 shutdown-timeout"
                sudo rm -f "$backup"
            else
                echo "[错误] 更新失败，备份保留在 $backup"
                return 1
            fi
        else
            echo "[跳过] jq 未安装，需手动编辑 $daemon_json 移除 shutdown-timeout"
        fi
    else
        if command -v jq &>/dev/null; then
            if jq --arg t "$timeout" '.["shutdown-timeout"] = ($t|tonumber)' "$backup" | sudo tee "$daemon_json" > /dev/null; then
                echo "[OK] Docker daemon.json 已更新 (shutdown-timeout: ${timeout}s)"
                sudo rm -f "$backup"
            else
                echo "[错误] 更新失败，备份保留在 $backup"
                return 1
            fi
        else
            echo "[跳过] jq 未安装，需手动编辑 $daemon.json 添加: \"shutdown-timeout\": ${timeout}"
        fi
    fi
}

# 自动清理 .env 中已废弃的变量（仅在写操作时执行）
_cleanup_deprecated_vars() {
    if grep -q '^VLLM_GPU_UTIL=' .env; then
        echo "提示: 已从 .env 移除废弃变量 VLLM_GPU_UTIL（gpu_memory_utilization 现由 profile YAML 控制）"
        sed -i '/^VLLM_GPU_UTIL=/d' .env
    fi
}

# 幂等写入 .env 变量: 存在则 sed 替换，不存在则追加
_set_env_var() {
    local key="$1" value="$2"
    if grep -q "^${key}=" .env; then
        sed -i "s|^${key}=.*|${key}=${value}|" .env
    else
        [ -n "$(tail -1 .env 2>/dev/null)" ] && echo "" >> .env
        echo "${key}=${value}" >> .env
    fi
}

# ---------- 变量初始化 ----------

DEFAULT_IMAGE="vllm-qwen36:rtx3090-sm86"
PROFILES_DIR="./profiles"
PRIMARY_PROFILE="${VLLM_PRIMARY_PROFILE:-agent-fast}"
SECONDARY_PROFILE="${VLLM_SECONDARY_PROFILE:-agent-thinking}"
# 日志目录固定为服务名（不再跟随 profile）
VLLM_PRIMARY_LOG_DIR="./logs/primary"
export VLLM_PRIMARY_LOG_DIR
VLLM_SECONDARY_LOG_DIR="./logs/secondary"
export VLLM_SECONDARY_LOG_DIR

# ---------- API Key 空值警告 ----------
_check_api_keys() {
    local target="${1:-all}"
    if [ "$target" = "all" ] || [ "$target" = "primary" ]; then
        if [ -z "${VLLM_PRIMARY_API_KEY:-}" ]; then
            echo "警告: Primary 服务未配置 API Key (VLLM_PRIMARY_API_KEY 为空)，将开放访问"
        fi
    fi
    if [ "$target" = "all" ] || [ "$target" = "secondary" ]; then
        if [ -z "${VLLM_SECONDARY_API_KEY:-}" ]; then
            echo "警告: Secondary 服务未配置 API Key (VLLM_SECONDARY_API_KEY 为空)，将开放访问"
        fi
    fi
}

# ---------- Profile 校验 ----------

validate_profile() {
    local profile="$1"
    local file="${PROFILES_DIR}/${profile}.yaml"
    if [ ! -f "$file" ]; then
        echo "错误: Profile '${profile}' 不存在 (${file})"
        echo ""
        echo "可用 profiles:"
        ls "${PROFILES_DIR}"/*.yaml 2>/dev/null | xargs -n1 basename | sed 's/\.yaml$//' || echo "  (无)"
        return 1
    fi
}

# ---------- 工具函数 ----------

# 从 profile YAML 中读取顶层字段值（锚定行首，只匹配非缩进字段）
get_profile_field() {
    local profile="$1" field="$2"
    local file="${PROFILES_DIR}/${profile}.yaml"
    [ -f "$file" ] || return 0
    grep -v '^\s*#' "$file" | grep "^${field}:" | head -1 | sed "s/^${field}: *//" || true
}

# 从 profile YAML 中读取嵌套字段值（不锚定行首，匹配任意缩进级别）
# 用于 override_generation_config.* 和 default_chat_template_kwargs.* 下的字段
_get_nested_field() {
    local profile="$1" field="$2"
    local file="${PROFILES_DIR}/${profile}.yaml"
    [ -f "$file" ] || return 0
    grep -v '^\s*#' "$file" | grep "${field}:" | head -1 | sed "s/.*${field}: *//" || true
}

# 获取 profile 的 MTP 信息
_get_mtp_info() {
    local profile="$1"
    local file="${PROFILES_DIR}/${profile}.yaml"
    [ -f "$file" ] || { echo "off"; return; }
    if grep -v '^\s*#' "$file" | grep -q "method: mtp"; then
        local n
        n="$(grep -v '^\s*#' "$file" | grep "num_speculative_tokens:" | head -1 | sed 's/.*num_speculative_tokens: *//')"
        echo "mtp n=${n:-1}"
    else
        echo "off"
    fi
}

# 检测 PYTORCH_CUDA_ALLOC_CONF expandable_segments 与 CUDA Graph 兼容性
# CUDA Graph (cudagraph_mode 非 NO_GRAPH) 启用时，expandable_segments 必须为 False
# 原因: VMM 动态重映射导致 CUDA Graph 捕获的地址失效 → prefill OOM (详见 R12)
_check_expandable_segments() {
    local profile="$1"
    local file="${PROFILES_DIR}/${profile}.yaml"
    [ -f "$file" ] || return 0

    # 解析 cudagraph_mode（跳过注释行）
    local cudagraph_mode
    cudagraph_mode="$(grep -v '^\s*#' "$file" | grep "cudagraph_mode:" | head -1 | sed 's/.*cudagraph_mode: *//' || true)"

    # 无 CUDA Graph 或显式禁用 → 无需检查
    if [ -z "$cudagraph_mode" ] || [ "$cudagraph_mode" = "NO_GRAPH" ]; then
        return 0
    fi

    # 读取 docker-compose.yml 中的 PYTORCH_CUDA_ALLOC_CONF
    local compose_file="${SCRIPT_DIR}/docker-compose.yml"
    if [ ! -f "$compose_file" ]; then
        echo "警告: docker-compose.yml 不存在，跳过 expandable_segments 检查"
        return 0
    fi

    local pytorch_conf
    pytorch_conf="$(grep "PYTORCH_CUDA_ALLOC_CONF:" "$compose_file" | head -1 | sed 's/.*PYTORCH_CUDA_ALLOC_CONF: *//' || true)"

    # 检测不兼容: expandable_segments:True → 自动修复为 False
    if echo "$pytorch_conf" | grep -q "expandable_segments:True"; then
        echo "错误: CUDA Graph (cudagraph_mode=${cudagraph_mode}) 与 expandable_segments:True 不兼容"
        echo "  profile: ${profile} | docker-compose.yml: PYTORCH_CUDA_ALLOC_CONF: ${pytorch_conf}"
        echo "  VMM 动态重映射会导致 CUDA Graph 地址失效 → prefill OOM"
        echo "  自动修复: expandable_segments:True → False"
        sed -i "s/expandable_segments:True/expandable_segments:False/" "$compose_file"
        echo "  ✅ 已修复 docker-compose.yml"
        echo "  提示: docker-compose.yml 已被修改，建议 commit 此修复"
        return 0
    fi

    # 确认已显式设为 False
    if ! echo "$pytorch_conf" | grep -q "expandable_segments:False"; then
        echo "警告: CUDA Graph 已启用但 PYTORCH_CUDA_ALLOC_CONF 未显式设置 expandable_segments:False"
        echo "  当前值: ${pytorch_conf:-未设置}"
        echo "  建议显式设置: PYTORCH_CUDA_ALLOC_CONF: expandable_segments:False (详见 R12)"
    fi

    return 0
}

# 校验单个 cpuset 值的格式和范围，输出展开后的 CPU ID 列表到 stdout
_validate_cpuset_value() {
    local label="$1" cpuset="$2"
    [ -z "$cpuset" ] && return 0

    if ! echo "$cpuset" | grep -qE '^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$'; then
        echo "错误: ${label} cpuset 格式无效: '${cpuset}'" >&2
        echo "  期望格式: 逗号分隔的 CPU 编号或范围，例如 '0-23,48-71'" >&2
        return 1
    fi

    local total_cpus
    total_cpus="$(nproc 2>/dev/null || echo 0)"

    local segment
    while IFS= read -r segment; do
        [ -z "$segment" ] && continue
        if [[ "$segment" == *"-"* ]]; then
            local start="${segment%-*}" end="${segment#*-}"
            if [ "$start" -gt "$end" ] 2>/dev/null; then
                echo "错误: ${label} cpuset 范围无效: '${segment}' (起始 > 结束)" >&2
                return 1
            fi
            if [ "$total_cpus" -gt 0 ] && [ "$((end - start + 1))" -gt "$((total_cpus * 2))" ]; then
                echo "错误: ${label} cpuset 范围 '${segment}' 跨度过大 (系统共 ${total_cpus} 逻辑 CPU)" >&2
                return 1
            fi
            local id
            for id in $(seq "$start" "$end"); do
                if [ "$total_cpus" -gt 0 ] && [ "$id" -ge "$total_cpus" ]; then
                    echo "错误: ${label} cpuset CPU ${id} 超出系统范围 (共 ${total_cpus} 逻辑 CPU, 编号 0-$((total_cpus-1)))" >&2
                    return 1
                fi
                echo "$id"
            done
        else
            if [ "$total_cpus" -gt 0 ] && [ "$segment" -ge "$total_cpus" ] 2>/dev/null; then
                echo "错误: ${label} cpuset CPU ${segment} 超出系统范围 (共 ${total_cpus} 逻辑 CPU, 编号 0-$((total_cpus-1)))" >&2
                return 1
            fi
            echo "$segment"
        fi
    done < <(echo "$cpuset" | tr ',' '\n')
    return 0
}

# 检测 Primary/Secondary cpuset 重叠
_check_cpuset_overlap() {
    local primary_ids="$1" secondary_ids="$2"
    [ -z "$primary_ids" ] || [ -z "$secondary_ids" ] && return 0

    local overlap
    overlap="$(comm -12 <(echo "$primary_ids" | sort) <(echo "$secondary_ids" | sort))"
    if [ -n "$overlap" ]; then
        echo "错误: CPU 隔离范围重叠: $(echo "$overlap" | tr '\n' ',' | sed 's/,$//')"
        echo "  VLLM_PRIMARY_CPUSET=${VLLM_PRIMARY_CPUSET:-}"
        echo "  VLLM_SECONDARY_CPUSET=${VLLM_SECONDARY_CPUSET:-}"
        echo "  每个 CPU 核心只能分配给一个服务"
        return 1
    fi
    return 0
}

# 从展开的 CPU ID 列表中剥离超线程兄弟，仅保留物理核心
# 输入: stdout 展开的 CPU ID（每行一个），来自 _validate_cpuset_value
# 输出: 去除 HT 后的 CPU ID 列表（每行一个）
_strip_ht_siblings() {
    local ids="$1"
    [ -z "$ids" ] && return 0
    while IFS= read -r cpu_id; do
        [ -z "$cpu_id" ] && continue
        local siblings_list
        siblings_list="$(cat "/sys/devices/system/cpu/cpu${cpu_id}/topology/thread_siblings_list" 2>/dev/null)" || { echo "$cpu_id"; continue; }
        # thread_siblings_list 格式: "0,48" 或 "0-48" 等；取编号最小的
        local primary
        primary="$(echo "$siblings_list" | grep -oE '[0-9]+' | sort -n | head -1)"
        # 仅当此 CPU 是 primary 时输出（避免重复）
        if [ "$cpu_id" = "$primary" ]; then
            echo "$cpu_id"
        fi
    done < <(echo "$ids")
}

# 将展开的 CPU ID 列表合并为 cpuset 范围格式
# 输入: CPU ID 列表（每行一个）
# 输出: 范围格式字符串，如 "0-11,24-35"
_compact_cpuset() {
    local ids="$1"
    [ -z "$ids" ] && return 0
    local sorted start end result
    sorted="$(echo "$ids" | sort -n | uniq)"
    start="" end="" result=""
    while IFS= read -r id; do
        [ -z "$id" ] && continue
        if [ -z "$start" ]; then
            start="$id" end="$id"
        elif [ "$id" -eq "$((end + 1))" ] 2>/dev/null; then
            end="$id"
        else
            if [ "$start" = "$end" ]; then
                result="${result:+$result,}$start"
            else
                result="${result:+$result,}$start-$end"
            fi
            start="$id" end="$id"
        fi
    done < <(echo "$sorted")
    # 处理最后一个范围
    if [ -n "$start" ]; then
        if [ "$start" = "$end" ]; then
            result="${result:+$result,}$start"
        else
            result="${result:+$result,}$start-$end"
        fi
    fi
    echo "$result"
}

# cpuset 统一校验入口（格式 + 范围 + 重叠 + 部分配置提示）
_validate_cpuset() {
    local primary_cpuset="${VLLM_PRIMARY_CPUSET:-}"
    local secondary_cpuset="${VLLM_SECONDARY_CPUSET:-}"

    # 双空 = 不隔离，合法状态，仅提示
    if [ -z "$primary_cpuset" ] && [ -z "$secondary_cpuset" ]; then
        echo "提示: CPU 隔离未配置，双服务线程可能竞争导致输出卡顿"
        echo "  自动检测: ./manage.sh detect-topology"
        echo "  一键应用: ./manage.sh apply-cpuset"
        echo ""
        return 0
    fi

    # 格式 + 范围校验
    local primary_ids="" secondary_ids=""
    primary_ids="$(_validate_cpuset_value "Primary" "$primary_cpuset")" || return 1
    secondary_ids="$(_validate_cpuset_value "Secondary" "$secondary_cpuset")" || return 1

    # 部分配置提示
    if [ -z "$primary_cpuset" ] && [ -n "$secondary_cpuset" ]; then
        echo "警告: 仅 Secondary 配置了 cpuset, Primary 未隔离 (VLLM_PRIMARY_CPUSET 为空)"
    elif [ -n "$primary_cpuset" ] && [ -z "$secondary_cpuset" ]; then
        echo "警告: 仅 Primary 配置了 cpuset, Secondary 未隔离 (VLLM_SECONDARY_CPUSET 为空)"
    fi

    # 重叠检测
    _check_cpuset_overlap "$primary_ids" "$secondary_ids" || return 1
    return 0
}

# 生成 docker-compose.override.yml（仅当 cpuset 已配置时）
_generate_cpuset_override() {
    local primary_cpuset secondary_cpuset
    primary_cpuset="${VLLM_PRIMARY_CPUSET:-}"
    secondary_cpuset="${VLLM_SECONDARY_CPUSET:-}"

    # 无 cpuset → 删除 override 文件（如有），使用默认 docker-compose.yml
    if [ -z "$primary_cpuset" ] && [ -z "$secondary_cpuset" ]; then
        rm -f "${SCRIPT_DIR}/docker-compose.override.yml"
        return 0
    fi

    # 幂等: 已存在且内容匹配则跳过重新生成
    local override_file="${SCRIPT_DIR}/docker-compose.override.yml"
    if [ -f "$override_file" ]; then
        local expected_count=0 actual_count
        [ -n "$primary_cpuset" ] && expected_count=$((expected_count + 1))
        [ -n "$secondary_cpuset" ] && expected_count=$((expected_count + 1))
        actual_count="$(grep -c "cpuset:" "$override_file" 2>/dev/null || echo 0)"
        if [ "$actual_count" -eq "$expected_count" ]; then
            local need_regenerate=false
            if [ -n "$primary_cpuset" ]; then
                grep -q "cpuset: \"${primary_cpuset}\"" "$override_file" || need_regenerate=true
            fi
            if [ -n "$secondary_cpuset" ]; then
                grep -q "cpuset: \"${secondary_cpuset}\"" "$override_file" || need_regenerate=true
            fi
            if [ "$need_regenerate" = false ]; then
                return 0
            fi
        fi
    fi

    echo "生成 CPU 隔离覆盖配置..."
    cat > "${SCRIPT_DIR}/docker-compose.override.yml" <<OVERRIDE
# 由 manage.sh 自动生成 — CPU NUMA 隔离覆盖
# 删除此文件可恢复默认（不隔离），或执行: ./manage.sh apply-cpuset 重新生成
services:
OVERRIDE

    if [ -n "$primary_cpuset" ]; then
        echo "  vllm-primary:" >> "${SCRIPT_DIR}/docker-compose.override.yml"
        echo "    cpuset: \"${primary_cpuset}\"" >> "${SCRIPT_DIR}/docker-compose.override.yml"
    fi
    if [ -n "$secondary_cpuset" ]; then
        echo "  vllm-secondary:" >> "${SCRIPT_DIR}/docker-compose.override.yml"
        echo "    cpuset: \"${secondary_cpuset}\"" >> "${SCRIPT_DIR}/docker-compose.override.yml"
    fi

    echo "  cpuset 已生效: Primary=${primary_cpuset:-(默认)}, Secondary=${secondary_cpuset:-(默认)}"
}

# 根据服务名获取显示标签
get_label() {
    case "$1" in
        primary)   echo "primary (${PRIMARY_PROFILE})" ;;
        secondary) echo "secondary (${SECONDARY_PROFILE})" ;;
        *)         echo "$1" ;;
    esac
}

# 根据服务名获取宿主机端口
get_port() {
    case "$1" in
        primary)      echo "${VLLM_PRIMARY_HOST_PORT:-8089}" ;;
        secondary) echo "${VLLM_SECONDARY_HOST_PORT:-8099}" ;;
        *)         echo "未知服务: $1" >&2; return 1 ;;
    esac
}

# 根据服务名获取日志目录
get_log_dir() {
    case "$1" in
        primary)      echo "${VLLM_PRIMARY_LOG_DIR:-./logs/primary}" ;;
        secondary) echo "${VLLM_SECONDARY_LOG_DIR:-./logs/secondary}" ;;
        *)         echo "未知服务: $1" >&2; return 1 ;;
    esac
}

# 根据服务名获取 GPU IDs
get_gpu_ids() {
    case "$1" in
        primary)      echo "${VLLM_PRIMARY_GPU_IDS:-0,1}" ;;
        secondary) echo "${VLLM_SECONDARY_GPU_IDS:-2,3}" ;;
        *)         echo "未知服务: $1" >&2; return 1 ;;
    esac
}

# 根据服务名获取 docker compose 服务名
get_compose_service() {
    case "$1" in
        primary)      echo "vllm-primary" ;;
        secondary) echo "vllm-secondary" ;;
        *)         echo "未知服务: $1" >&2; return 1 ;;
    esac
}

# 等待服务就绪
wait_ready() {
    local svc="$1"
    local port="$2"
    local key_var="VLLM_${svc^^}_API_KEY"
    local key="${!key_var:-}"
    local curl_args=(-s)
    [ -n "$key" ] && curl_args+=(-H "Authorization: Bearer ${key}")
    local label; label="$(get_label "$svc")"
    echo "等待 ${label} 服务就绪 (端口 ${port}, 最长 400s, 首次启动约 5-7 分钟)..."
    for i in $(seq 1 200); do
        if curl --max-time 5 "${curl_args[@]}" "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
            echo "${label} 服务已就绪! ($((i*2))s)"
            echo ""
            echo "快速测试:"
            if [ -n "$key" ]; then
                echo "  curl -H 'Authorization: Bearer ${key}' http://localhost:${port}/v1/models"
            else
                echo "  curl http://localhost:${port}/v1/models"
            fi
            return 0
        fi
        sleep 2
        REM=$((i % 15))
        if [ "$REM" -eq 0 ]; then
            echo "  等待中... ($((i*2))s) - 查看日志: ./manage.sh logs ${svc}"
        fi
    done
    echo "警告: ${label} 启动超时 (400s)，请检查日志: ./manage.sh logs ${svc}"
    return 1
}

# 显示部署信息
print_header() {
    local target="${1:-all}"
    local primary_gpu_util
    primary_gpu_util="$(get_profile_field "${PRIMARY_PROFILE}" "gpu_memory_utilization")"
    local secondary_gpu_util
    secondary_gpu_util="$(get_profile_field "${SECONDARY_PROFILE}" "gpu_memory_utilization")"
    local primary_mtp secondary_mtp
    primary_mtp="$(_get_mtp_info "$PRIMARY_PROFILE")"
    secondary_mtp="$(_get_mtp_info "$SECONDARY_PROFILE")"
    echo "=========================================="
    echo " Qwen3.6-27B-FP8 双服务并行部署"
    echo "=========================================="
    echo " 镜像:     ${DOCKER_IMAGE:-$DEFAULT_IMAGE}"
    if [ "$target" = "all" ] || [ "$target" = "primary" ]; then
        echo ""
        echo " Primary [${PRIMARY_PROFILE}]:"
        echo "   GPU:    ${VLLM_PRIMARY_GPU_IDS:-0,1}"
        echo "   端口:   ${VLLM_PRIMARY_HOST_PORT:-8089}"
        echo "   显存:   ${primary_gpu_util:-0.97}"
        echo "   API Key: $([ -n "${VLLM_PRIMARY_API_KEY:-}" ] && echo "已配置" || echo "未设置(开放访问)")"
        echo "   CPU:    ${VLLM_PRIMARY_CPUSET:-未隔离}"
        echo "   MTP:    ${primary_mtp}"
    fi
    if [ "$target" = "all" ] || [ "$target" = "secondary" ]; then
        echo ""
        echo " Secondary [${SECONDARY_PROFILE}]:"
        echo "   GPU:    ${VLLM_SECONDARY_GPU_IDS:-2,3}"
        echo "   端口:   ${VLLM_SECONDARY_HOST_PORT:-8099}"
        echo "   显存:   ${secondary_gpu_util:-0.97}"
        echo "   API Key: $([ -n "${VLLM_SECONDARY_API_KEY:-}" ] && echo "已配置" || echo "未设置(开放访问)")"
        echo "   CPU:    ${VLLM_SECONDARY_CPUSET:-未隔离}"
        echo "   MTP:    ${secondary_mtp}"
    fi
    echo ""
    echo "=========================================="
}

# 显示服务完整配置（从 .env + profile YAML 读取，不依赖 Docker）
show_config() {
    local svc="$1" profile="$2" gpu_ids="$3" port="$4" key="$5" cpuset="$6"
    local file="${PROFILES_DIR}/${profile}.yaml"

    if [ ! -f "$file" ]; then
        echo "  ⚠ Profile 文件不存在: ${file}"
        return
    fi

    local gpu_util max_len thinking preserve_kvp temp kv_dtype top_p_val presence_p mtp_method mtp_n
    gpu_util="$(get_profile_field "$profile" "gpu_memory_utilization")"
    max_len="$(get_profile_field "$profile" "max_model_len")"
    # 嵌套字段（override_generation_config.* / default_chat_template_kwargs.*）
    temp="$(_get_nested_field "$profile" "temperature")"
    thinking="$(_get_nested_field "$profile" "enable_thinking")"
    preserve_kvp="$(_get_nested_field "$profile" "preserve_thinking")"
    kv_dtype="$(get_profile_field "$profile" "kv_cache_dtype")"
    top_p_val="$(_get_nested_field "$profile" "top_p")"
    presence_p="$(_get_nested_field "$profile" "presence_penalty")"
    mtp_method="$(_get_nested_field "$profile" "method")"
    mtp_n="$(_get_nested_field "$profile" "num_speculative_tokens")"

    # 上下文长度人类可读
    local ctx_human
    case "$max_len" in
        262144) ctx_human="256K" ;;
        155648) ctx_human="152K" ;;
        148480) ctx_human="145K" ;;
        143360) ctx_human="140K" ;;
        139264) ctx_human="136K" ;;
        135168) ctx_human="132K" ;;
        131072) ctx_human="128K" ;;
        *)      ctx_human="${max_len:-?}" ;;
    esac

    echo "  ${svc} [${profile}]:"
    echo "    GPU:        ${gpu_ids}"
    echo "    端口:       ${port}"
    echo "    显存:       ${gpu_util:-0.97}"
    echo "    KV Cache:   ${kv_dtype:-?}"
    echo "    上下文:     ${max_len:-?} (${ctx_human})"
    if [ -n "$mtp_method" ]; then
        echo "    MTP:        ${mtp_method} n=${mtp_n:-?}"
        if [ "${mtp_n:-0}" -ge 2 ] 2>/dev/null && [ "${gpu_util:-0}" = "0.97" ]; then
            echo "    ⚠ n>=2 需 gpu_memory_utilization<=0.96 (当前: ${gpu_util})"
        fi
    else
        echo "    MTP:        off"
    fi
    echo "    思考:       ${thinking:-?}"
    echo "    preserve:   ${preserve_kvp:-?}"
    echo "    温度:       ${temp:-?}"
    echo "    top_p:      ${top_p_val:-?}"
    echo "    penalty:    ${presence_p:-?}"
    echo "    API Key:    $([ -n "$key" ] && echo "已配置" || echo "未设置(开放访问)")"
    echo "    CPU 隔离:   ${cpuset:-未配置}"
    echo "    日志:       ./logs/${svc,,}/vllm.log"
}

# 解析服务选择参数
parse_target() {
    local input="${1:-all}"
    case "$input" in
        primary|secondary) echo "$input" ;;
        all)               echo "all" ;;
        *)
            echo "错误: 未知参数 '${input}'" >&2
            echo "可选: primary, secondary, all" >&2
            return 1
            ;;
    esac
}

# 显示单个服务状态
show_service_status() {
    local svc="$1"
    local compose_svc
    compose_svc="$(get_compose_service "$svc")"
    local port
    port="$(get_port "$svc")"
    local key_var="VLLM_${svc^^}_API_KEY"
    local key="${!key_var:-}"
    local curl_args=(-sf)
    [ -n "$key" ] && curl_args+=(-H "Authorization: Bearer ${key}")

    local label="${svc}"
    [ "$svc" = "primary" ] && label="primary [${PRIMARY_PROFILE}]"
    [ "$svc" = "secondary" ] && label="secondary [${SECONDARY_PROFILE}]"
    echo "--- ${label} 服务 (${compose_svc}) ---"

    # 容器状态
    local container_status
    container_status=$(sudo docker compose ps --status running --format '{{.Name}}' 2>/dev/null | grep -q "^${compose_svc}$" && echo "运行中" || echo "已停止")
    echo "  容器: ${container_status}"

    # API 状态
    if curl --max-time 3 "${curl_args[@]}" "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
        echo "  API:  正常 (http://localhost:${port})"
    else
        echo "  API:  不可达"
    fi
    echo ""
}

# ---------- 启动逻辑 ----------

start_service() {
    local svc="$1"
    # 校验模型路径
    if [ -z "${VLLM_MODEL_PATH:-}" ]; then
        echo "错误: .env 缺少 VLLM_MODEL_PATH（模型路径）"
        echo "  请在 .env 中设置 VLLM_MODEL_PATH 指向模型目录"
        exit 1
    fi
    if [ ! -d "${VLLM_MODEL_PATH}" ]; then
        echo "错误: 模型路径不存在: ${VLLM_MODEL_PATH}"
        echo "  请在 .env 中设置 VLLM_MODEL_PATH 指向模型目录"
        exit 1
    fi
    if [ -z "$(ls -A "${VLLM_MODEL_PATH}" 2>/dev/null)" ]; then
        echo "警告: 模型目录为空: ${VLLM_MODEL_PATH}"
    fi
    if [ -z "${VLLM_MODEL_DIR:-}" ]; then
        echo "错误: .env 缺少 VLLM_MODEL_DIR（模型目录名，用于容器内路径）"
        exit 1
    fi
    local compose_svc
    compose_svc="$(get_compose_service "$svc")"
    local port
    port="$(get_port "$svc")"
    local gpu_ids
    gpu_ids="$(get_gpu_ids "$svc")"
    # 校验 GPU 数量与 profile tensor_parallel_size 匹配
    local active_profile
    active_profile="$(case "$svc" in primary) echo "$PRIMARY_PROFILE" ;; secondary) echo "$SECONDARY_PROFILE" ;; *) echo "" ;; esac)"
    local tp_size
    tp_size="$(get_profile_field "$active_profile" "tensor_parallel_size")"
    local gpu_count
    gpu_count="$(echo "$gpu_ids" | tr ',' '\n' | grep -c .)"
    if [ -n "$tp_size" ] && [ "$tp_size" -ne "$gpu_count" ] 2>/dev/null; then
        echo "错误: ${svc} GPU 数量 (${gpu_count}) 与 ${active_profile}.yaml 的 tensor_parallel_size (${tp_size}) 不匹配"
        exit 1
    fi
    _validate_gpu_exists "$gpu_ids" || exit 1
    # 检测 expandable_segments 与 CUDA Graph 兼容性
    _check_expandable_segments "$active_profile"
    local logdir
    logdir="$(get_log_dir "$svc")"

    # 确保日志目录存在（避免 Docker 以 root:root 创建）
    mkdir -p "$logdir"
    # 确保各服务的 FlashInfer 缓存目录存在（独立 bind mount，避免并发写入）
    mkdir -p ./flashinfer-cache/primary ./flashinfer-cache/secondary

    local label; label="$(get_label "$svc")"
    echo "启动 ${label} 服务 (GPU ${gpu_ids}, 端口 ${port})..."
    _generate_cpuset_override
    sudo docker compose up -d "$compose_svc"
    wait_ready "$svc" "$port"
}

# 条件启动 Secondary 服务：GPU 可用且 profile 有效时启动，否则跳过（非致命）
_start_secondary_if_available() {
    if _service_gpus_available secondary; then
        if validate_profile "$SECONDARY_PROFILE"; then
            start_service secondary
            echo ""
        else
            echo "警告: Secondary profile 无效，跳过 Secondary 服务"
        fi
    else
        echo "提示: Secondary GPU (${VLLM_SECONDARY_GPU_IDS:-2,3}) 不可用，跳过 Secondary 服务"
    fi
}

# 从 nvidia-smi topo -m 输出中解析指定 GPU 列表的 CPU Affinity
# 输出: cpuset 格式字符串（如 "0-23,48-71"），多个 GPU 的 Affinity 逗号拼接
_parse_gpu_affinity() {
    local topo="$1" gpu_count="$2" gpu_ids="$3"
    local cpu_col=$((gpu_count + 2))
    local result="" id affinity
    for id in $(echo "$gpu_ids" | tr ',' '\n'); do
        affinity="$(echo "$topo" | awk -v gpu="GPU${id}" -v col="$cpu_col" '$1 == gpu {print $col}')"
        if [ -n "$affinity" ] && [ "$affinity" != "N/A" ]; then
            [ -n "$result" ] && result+=","
            result+="$affinity"
        fi
    done
    echo "$result"
}

# 本地修复覆盖: 将 local-fixes/ 中的修改文件临时覆盖到 vllm/ 子模块
# 仅在 local-fixes/ 目录存在时生效
# auto-detection: 读取 local-fixes/CHECK 标记，官方已包含修复时自动跳过
_apply_local_fixes() {
    local fixes_dir="${SCRIPT_DIR}/local-fixes"
    if [ ! -d "${fixes_dir}/vllm" ]; then
        echo "提示: 无本地修复 (local-fixes/ 不存在，构建纯官方版本)"
        return 0
    fi
    echo "检查本地修复..."

    # 读取检测标记
    local check_file="${fixes_dir}/CHECK"
    local -a markers=()
    if [ -f "$check_file" ]; then
        while IFS= read -r line; do
            [[ "$line" =~ ^#.*$ ]] && continue
            [[ -z "$line" ]] && continue
            markers+=("$line")
        done < "$check_file"
    fi

    local applied=0 skipped=0
    # 遍历 local-fixes/vllm/ 下所有文件
    while IFS= read -r -d '' fix_file; do
        local rel_path="${fix_file#${fixes_dir}/}"
        local target="${SCRIPT_DIR}/${rel_path}"

        if [ ! -f "$target" ]; then
            echo "  警告: ${rel_path} — 目标文件不存在，跳过"
            skipped=$((skipped + 1))
            continue
        fi

        # auto-detection: 检查标记是否已存在于官方源码
        local marker_found=false
        for marker in "${markers[@]}"; do
            if grep -qF "$marker" "$target" 2>/dev/null; then
                marker_found=true
                break
            fi
        done

        if $marker_found; then
            echo "  [skip] ${rel_path} — 官方已包含修复"
            skipped=$((skipped + 1))
        else
            cp "$fix_file" "$target"
            echo "  [OK]   ${rel_path} — 已覆盖"
            applied=$((applied + 1))
        fi
    done < <(find "${fixes_dir}/vllm" -type f -print0)

    echo "  结果: ${applied} 已覆盖 | ${skipped} 已跳过"
}

# 恢复 vllm/ 子模块到干净状态（git checkout）
_restore_vllm() {
    echo "恢复 vllm/ 子模块到干净状态..."
    cd "${SCRIPT_DIR}/vllm" && git checkout -- .
    echo "  已恢复"
}

# ---------- 命令分发 ----------

CMD="${1:-help}"

case "$CMD" in
    start)
        TARGET="$(parse_target "${2:-all}")" || exit 1
        if [ -n "${3:-}" ]; then
            if [ "$TARGET" = "all" ]; then
                echo "提示: 不再支持临时 profile 参数 '${3}'。请分别设置: ./manage.sh set-profile primary ${3} && ./manage.sh set-profile secondary ${3}"
            else
                echo "提示: 不再支持临时 profile 参数 '${3}'。请使用: ./manage.sh set-profile ${TARGET} ${3} && ./manage.sh restart ${TARGET}"
            fi
        fi
        _cleanup_deprecated_vars

        # 驱动健康检查（在所有 GPU 相关操作之前）
        _check_nvidia_driver || exit 1

        case "$TARGET" in
            all)
                validate_profile "$PRIMARY_PROFILE" || exit 1
                _check_api_keys all
                _validate_cpuset || exit 1
                print_header
                echo ""
                # Primary: 必须可启动
                if ! _service_gpus_available primary; then
                    echo "错误: Primary GPU (${VLLM_PRIMARY_GPU_IDS:-0,1}) 不可用，无法启动任何服务"
                    exit 1
                fi
                start_service primary
                echo ""
                # Secondary: GPU 不可用时跳过（非致命）
                _start_secondary_if_available
                ;;
            primary|secondary)
                active_profile="$(case "$TARGET" in primary) echo "$PRIMARY_PROFILE" ;; secondary) echo "$SECONDARY_PROFILE" ;; esac)"
                validate_profile "$active_profile" || exit 1
                _validate_cpuset || exit 1
                _check_api_keys "$TARGET"
                print_header "$TARGET"
                echo ""
                start_service "$TARGET"
                ;;
        esac
        ;;

    stop)
        TARGET="$(parse_target "${2:-all}")" || exit 1
        if [ -n "${3:-}" ]; then
            echo "提示: 忽略多余参数 '${3}'。用法: ./manage.sh stop [primary|secondary|all]"
        fi
        _cleanup_deprecated_vars
        case "$TARGET" in
            all)
                echo "停止所有服务..."
                sudo docker compose down --timeout 120 --remove-orphans
                echo "所有服务已停止"
                ;;
            primary|secondary)
                compose_svc="$(get_compose_service "$TARGET")"
                echo "停止 ${TARGET} 服务..."
                sudo docker compose stop --timeout 120 "$compose_svc"
                echo "${TARGET} 服务已停止"
                ;;
        esac
        ;;

    restart)
        TARGET="$(parse_target "${2:-all}")" || exit 1
        if [ -n "${3:-}" ]; then
            if [ "$TARGET" = "all" ]; then
                echo "提示: 不再支持临时 profile 参数 '${3}'。请分别设置: ./manage.sh set-profile primary ${3} && ./manage.sh set-profile secondary ${3}"
            else
                echo "提示: 不再支持临时 profile 参数 '${3}'。请使用: ./manage.sh set-profile ${TARGET} ${3} && ./manage.sh restart ${TARGET}"
            fi
        fi
        _cleanup_deprecated_vars

        # 驱动健康检查
        _check_nvidia_driver || exit 1

        case "$TARGET" in
            all)
                validate_profile "$PRIMARY_PROFILE" || exit 1
                echo "停止所有服务..."
                sudo docker compose down --timeout 120 --remove-orphans
                sleep 2
                _check_api_keys all
                _validate_cpuset || exit 1
                print_header
                echo ""
                if ! _service_gpus_available primary; then
                    echo "错误: Primary GPU (${VLLM_PRIMARY_GPU_IDS:-0,1}) 不可用"
                    exit 1
                fi
                start_service primary
                echo ""
                _start_secondary_if_available
                ;;
            primary|secondary)
                active_profile="$(case "$TARGET" in primary) echo "$PRIMARY_PROFILE" ;; secondary) echo "$SECONDARY_PROFILE" ;; esac)"
                validate_profile "$active_profile" || exit 1
                compose_svc="$(get_compose_service "$TARGET")"
                echo "重启 ${TARGET} 服务..."
                sudo docker compose stop --timeout 120 "$compose_svc"
                sleep 2
                _validate_cpuset || exit 1
                _check_api_keys "$TARGET"
                print_header "$TARGET"
                echo ""
                start_service "$TARGET"
                ;;
        esac
        ;;

    config)
        echo "=== 配置预检 ==="
        echo " 镜像: ${DOCKER_IMAGE:-$DEFAULT_IMAGE}"
        # 模型路径校验 (与 start_service 一致)
        if [ -d "${VLLM_MODEL_PATH:-}" ] && [ -n "$(ls -A "${VLLM_MODEL_PATH}" 2>/dev/null)" ]; then
            echo " 模型路径: ${VLLM_MODEL_PATH} (存在)"
        else
            echo " 模型路径: ${VLLM_MODEL_PATH:-未设置} (不存在或为空)"
        fi
        if [ -z "${VLLM_MODEL_DIR:-}" ]; then
            echo " VLLM_MODEL_DIR: 未设置 (启动时会报错)"
        fi
        _check_api_keys all
        echo ""
        echo " 可用 Profile:"
        for f in ${PROFILES_DIR}/*.yaml; do
            [ -f "$f" ] || continue
            name=$(basename "$f" .yaml)
            desc=$(grep -E '^# [a-zA-Z0-9_]' "$f" | head -1 | sed 's/^# //')
            marker=""
            [ "$name" = "${PRIMARY_PROFILE}" ] && marker=" <-- PRIMARY"
            [ "$name" = "${SECONDARY_PROFILE}" ] && marker="${marker:+$marker / }<-- SECONDARY"
            printf "   %-20s %s%s\n" "$name" "${desc:-}" "$marker"
        done
        echo ""
        show_config "Primary" "${PRIMARY_PROFILE}" \
            "${VLLM_PRIMARY_GPU_IDS:-0,1}" \
            "${VLLM_PRIMARY_HOST_PORT:-8089}" \
            "${VLLM_PRIMARY_API_KEY:-}" \
            "${VLLM_PRIMARY_CPUSET:-}"
        echo ""
        show_config "Secondary" "${SECONDARY_PROFILE}" \
            "${VLLM_SECONDARY_GPU_IDS:-2,3}" \
            "${VLLM_SECONDARY_HOST_PORT:-8099}" \
            "${VLLM_SECONDARY_API_KEY:-}" \
            "${VLLM_SECONDARY_CPUSET:-}"
        echo ""
        if [ -n "${VLLM_PRIMARY_CPUSET:-}" ] || [ -n "${VLLM_SECONDARY_CPUSET:-}" ]; then
            echo " CPU 隔离: 已配置"
            [ -n "${VLLM_PRIMARY_CPUSET:-}" ] && echo "  Primary cpuset:   ${VLLM_PRIMARY_CPUSET}"
            [ -n "${VLLM_SECONDARY_CPUSET:-}" ] && echo "  Secondary cpuset: ${VLLM_SECONDARY_CPUSET}"
            if [ -f "${SCRIPT_DIR}/docker-compose.override.yml" ]; then
                echo "  Override 文件: 存在"
            else
                echo "  Override 文件: 不存在 (下次启动时生成)"
            fi
        else
            echo " CPU 隔离: 未配置 (运行 ./manage.sh detect-topology 查看 NUMA 拓扑)"
        fi
        ;;

    status)
        STATUS_TARGET="${2:-}"
        [ "$STATUS_TARGET" = "all" ] && STATUS_TARGET=""
        # 一次性获取运行中的容器列表
        RUNNING_CONTAINERS="$(sudo docker compose ps --status running --format '{{.Name}}' 2>/dev/null || true)"
        PRIMARY_RUNNING=false
        SECONDARY_RUNNING=false
        echo "$RUNNING_CONTAINERS" | grep -q '^vllm-primary$' && PRIMARY_RUNNING=true
        echo "$RUNNING_CONTAINERS" | grep -q '^vllm-secondary$' && SECONDARY_RUNNING=true

        if [ -n "$STATUS_TARGET" ]; then
            case "$STATUS_TARGET" in
                primary|secondary)
                    echo "=== ${STATUS_TARGET} 服务状态 ==="
                    echo ""
                    show_service_status "$STATUS_TARGET"
                    ;;
                *)
                    echo "用法: $0 status [primary|secondary]"
                    exit 1
                    ;;
            esac
        elif [ "$PRIMARY_RUNNING" = true ] && [ "$SECONDARY_RUNNING" = true ]; then
            echo "=== 双服务状态 ==="
            echo ""
            show_service_status primary
            show_service_status secondary
        elif [ "$PRIMARY_RUNNING" = true ]; then
            echo "=== 单服务状态 (仅 Primary) ==="
            echo ""
            show_service_status primary
        elif [ "$SECONDARY_RUNNING" = true ]; then
            echo "=== 单服务状态 (仅 Secondary) ==="
            echo ""
            show_service_status secondary
        else
            echo "=== 服务状态 (均已停止) ==="
            echo ""
            show_service_status primary
            show_service_status secondary
        fi
        echo "=== 容器列表 ==="
        sudo docker compose ps 2>/dev/null
        ;;

    logs)
        if [ -z "${2:-}" ]; then
            echo "用法: $0 logs [primary|secondary]"
            exit 1
        fi
        case "$2" in
            primary|secondary)
                compose_svc="$(get_compose_service "$2")"
                sudo docker compose logs -f --tail 100 "$compose_svc"
                ;;
            *)
                echo "用法: $0 logs [primary|secondary]"
                exit 1
                ;;
        esac
        ;;

    logfiles)
        if [ -z "${2:-}" ]; then
            echo "用法: $0 logfiles [primary|secondary]"
            exit 1
        fi
        case "$2" in
            primary|secondary)
                logdir="$(get_log_dir "$2")"
                if [ ! -f "${logdir}/vllm.log" ]; then
                    echo "日志文件不存在: ${logdir}/vllm.log"
                    echo "$2 服务可能未启动，或日志尚未写入"
                    exit 1
                fi
                tail -f "${logdir}/vllm.log"
                ;;
            *)
                echo "用法: $0 logfiles [primary|secondary]"
                exit 1
                ;;
        esac
        ;;

    build)
        # 构建参数（可通过 .env 覆盖，默认值自动检测）
        # OOM 时降 max_jobs（如 12 核 CPU: VLLM_BUILD_MAX_JOBS=12）
        build_arch="${VLLM_BUILD_CUDA_ARCH:-8.6}"
        build_jobs="${VLLM_BUILD_MAX_JOBS:-$(nproc)}"
        build_nvcc_threads="${VLLM_BUILD_NVCC_THREADS:-$(( $(nproc) > 8 ? 8 : $(nproc) ))}"
        echo "构建 Docker 镜像..."
        echo "  目标: ${DOCKER_IMAGE:-$DEFAULT_IMAGE}"
        echo "  架构: sm_${build_arch} | max_jobs=${build_jobs} | nvcc_threads=${build_nvcc_threads}"
        echo "  Dockerfile: ${SCRIPT_DIR}/Dockerfile (官方 + 可选阿里云源)"
        echo "  预计: ~45 分钟"
        echo ""

        # 前置校验: 编译依赖必须已准备
        ext_deps=(triton cutlass cutlass-fa flash-attn deepgemm flashmla qutlass)
        for dep in "${ext_deps[@]}"; do
            if [ ! -d "${SCRIPT_DIR}/vllm/external-src/${dep}" ]; then
                echo "错误: 缺少编译依赖 '${dep}'"
                echo "提示: 运行 ./manage.sh prepare-src 准备依赖（~422MB，约 5-10 分钟）"
                exit 1
            fi
        done

        # 前置校验: 磁盘空间（FA3 链接需大量临时空间）
        _avail_kb=$(df -k "${SCRIPT_DIR}" | awk 'NR==2{print $4}')
        _avail_gb=$((_avail_kb / 1024 / 1024))
        if [ "$_avail_gb" -lt 50 ]; then
            echo "  [警告] 可用磁盘空间不足: ${_avail_gb}GB < 50GB（FA3 编译链接可能失败）"
            echo "  建议: sudo docker builder prune --all -f 清理构建缓存"
            echo "  或删除旧镜像: sudo docker images | grep vllm"
        else
            echo "  [OK] 磁盘空间: ${_avail_gb}GB >= 50GB"
        fi

        # 关闭代理（Docker build 使用阿里云镜像，代理会导致拉取失败）
        unset http_proxy https_proxy ftp_proxy no_proxy \
              HTTP_PROXY HTTPS_PROXY FTP_PROXY NO_PROXY \
              all_proxy ALL_PROXY 2>/dev/null || true
        # 可选: 用户自定义扩展关闭脚本（如有）
        [[ -f ~/.off_proxy ]] && source ~/.off_proxy

        # 获取 vllm 版本字符串（子模块 .git 是 gitdir 引用，Docker 内不可用）
        _vllm_ver=$(cd "${SCRIPT_DIR}/vllm" && git describe --tags --long --always 2>/dev/null \
            | sed 's/^v//; s/-\([0-9]*\)-\(g[a-f0-9]*\)$/.dev\1+\2/')
        echo "  版本: ${_vllm_ver}"
        echo ""

        # 应用本地修复覆盖（构建前临时覆盖 vllm/ 子模块文件）
        _apply_local_fixes

        # 设置 trap: 无论构建成功或失败，都恢复 vllm/ 干净状态
        trap '_restore_vllm' EXIT

        sudo -E DOCKER_BUILDKIT=1 docker build \
            --network=host \
            --progress=plain \
            --file "${SCRIPT_DIR}/Dockerfile" \
            --target vllm-openai \
            --tag "${DOCKER_IMAGE:-$DEFAULT_IMAGE}" \
            --build-arg torch_cuda_arch_list="${build_arch}" \
            --build-arg max_jobs="${build_jobs}" \
            --build-arg nvcc_threads="${build_nvcc_threads}" \
            --build-arg RUN_WHEEL_CHECK=false \
            --build-arg SETUPTOOLS_SCM_PRETEND_VERSION="${_vllm_ver}" \
            --build-arg USE_CHINA_MIRROR="${VLLM_USE_CHINA_MIRROR:-false}" \
            --build-arg PIP_INDEX_URL="${VLLM_PIP_INDEX_URL:-https://pypi.org/simple/}" \
            --build-arg PIP_EXTRA_INDEX_URL="${VLLM_PIP_INDEX_URL:-https://pypi.org/simple/}" \
            --build-arg UV_INDEX_URL="${VLLM_PIP_INDEX_URL:-https://pypi.org/simple/}" \
            --build-arg UV_EXTRA_INDEX_URL="${VLLM_PIP_INDEX_URL:-https://pypi.org/simple/}" \
            --build-arg http_proxy= \
            --build-arg https_proxy= \
            --build-arg ftp_proxy= \
            "${SCRIPT_DIR}/vllm"
        build_rc=$?

        # 恢复 vllm/ 子模块并清除 trap
        _restore_vllm
        trap - EXIT

        echo ""
        if [ $build_rc -eq 0 ]; then
            echo "构建完成: ${DOCKER_IMAGE:-$DEFAULT_IMAGE}"
        else
            echo "构建失败 (rc=${build_rc})"
            exit $build_rc
        fi
        ;;

    prepare-src)
        # 准备 vLLM 编译外部依赖到 ./vllm/external-src/
        # 7 个依赖共 ~422MB，首次克隆约 5-10 分钟
        # --force: 清除已有依赖后重新克隆（升级 vllm 版本后使用）
        EXT_SRC_DIR="${SCRIPT_DIR}/vllm/external-src"

        if [ "${2:-}" = "--force" ]; then
            echo "强制重新准备编译依赖..."
            if [ -d "${EXT_SRC_DIR}" ]; then
                echo "  清除已有依赖: ${EXT_SRC_DIR}"
                for d in "${EXT_SRC_DIR}"/*/; do
                    [ -d "$d" ] && rm -r "$d"
                done
                rmdir "${EXT_SRC_DIR}" 2>/dev/null || true
            fi
        else
            echo "准备 vLLM 编译外部依赖..."
        fi
        echo "  目标: ${EXT_SRC_DIR}"
        echo "  预计: ~422MB，首次约 5-10 分钟"
        echo ""

        mkdir -p "${EXT_SRC_DIR}"

        clone_dep() {
            local name="$1" url="$2" ref="$3" target="${EXT_SRC_DIR}/$4"
            if [ -d "${target}" ]; then
                echo "  [跳过] ${name} (已存在: ${target})"
                return 0
            fi
            echo "  [克隆] ${name} (${ref})..."
            if ! git clone --depth 1 --branch "${ref}" "${url}" "${target}" 2>/dev/null; then
                # ref 可能是 commit hash，先克隆再 checkout
                if ! git clone --depth 50 "${url}" "${target}" 2>/dev/null; then
                    echo "  [失败] ${name}: git clone 失败"
                    [ -d "${target}" ] && rm -r "${target}"
                    return 1
                fi
                (cd "${target}" && git checkout "${ref}" 2>/dev/null) || true
            fi
            # 清理 .git 目录减小体积
            rm -r "${target}/.git"
        }

        clone_dep "triton"      "https://github.com/triton-lang/triton.git"           "v3.6.0"            "triton"
        clone_dep "cutlass"     "https://github.com/NVIDIA/cutlass.git"                "v4.4.2"            "cutlass"
        clone_dep "cutlass-fa"  "https://github.com/NVIDIA/cutlass.git"                "v3.9.0"            "cutlass-fa"
        clone_dep "flash-attn"  "https://github.com/vllm-project/flash-attention.git"  "f5bc33cfc0"        "flash-attn"
        clone_dep "deepgemm"    "https://github.com/deepseek-ai/DeepGEMM.git"          "891d57b4db"        "deepgemm"
        clone_dep "flashmla"    "https://github.com/vllm-project/FlashMLA.git"         "a6ec2ba7bd"        "flashmla"
        clone_dep "qutlass"     "https://github.com/IST-DASLab/qutlass.git"            "830d2c4537"        "qutlass"

        echo ""
        echo "准备完成。依赖列表:"
        ls -1 "${EXT_SRC_DIR}/" 2>/dev/null | sed 's/^/  /'
        ;;

    pull-base)
        echo "拉取 CUDA 基础镜像（通过 crane）..."
        pull_one_image() {
            local img="$1"
            echo "  拉取: ${img}"
            local local_tag tar_file
            local_tag=$(echo "$img" | tr '/:' '-')
            tar_file="/tmp/${local_tag}.tar"
            if ! crane pull "$img" "$tar_file"; then
                rm -f "$tar_file"
                echo "错误: crane pull 失败: ${img}"
                return 1
            fi
            sudo docker load < "$tar_file"
            rm -f "$tar_file"
            echo "  完成: ${img}"
        }
        pull_one_image "nvidia/cuda:13.0.2-devel-ubuntu22.04"
        pull_one_image "nvidia/cuda:13.0.2-base-ubuntu22.04"
        echo "基础镜像拉取完成"
        ;;

    edit)
        if [ -z "${2:-}" ]; then
            echo "用法: $0 edit [primary|secondary]"
            exit 1
        fi
        edit_target="$2"
        config_file=""
        case "$edit_target" in
            primary)      config_file="${PROFILES_DIR}/${PRIMARY_PROFILE}.yaml" ;;
            secondary) config_file="${PROFILES_DIR}/${SECONDARY_PROFILE}.yaml" ;;
            *)         echo "用法: $0 edit [primary|secondary]"; exit 1 ;;
        esac
        if [ ! -f "$config_file" ]; then
            echo "配置文件不存在: ${config_file}"
            exit 1
        fi

        # 保存编辑前副本，用于变更检测
        edit_tmp=$(mktemp)
        trap 'rm -f "$edit_tmp"' EXIT
        cp "$config_file" "$edit_tmp"

        # 默认 vi（所有 Unix 系统均可用，nano/taller 需额外安装）
        ${EDITOR:-vi} "$config_file"
        edit_rc=$?

        # 编辑器异常退出时提示
        if [ "$edit_rc" -ne 0 ]; then
            echo "编辑器退出异常 (exit code: ${edit_rc})，配置可能未保存"
            rm -f "$edit_tmp"
            trap - EXIT
            exit 1
        fi

        # 变更检测
        if cmp -s "$edit_tmp" "$config_file"; then
            echo "配置未变更: ${config_file}"
        else
            echo "配置已变更:"
            diff -u "$edit_tmp" "$config_file" || true
            echo ""
            echo "需 restart 生效: ./manage.sh restart ${edit_target}"
        fi
        rm -f "$edit_tmp"
        trap - EXIT
        ;;

    set-profile)
        # 语法: set-profile [primary|secondary] <name>
        # 省略服务名时默认 secondary (向后兼容)
        svc_arg="" new_profile="" env_var=""
        _cleanup_deprecated_vars
        if [ $# -ge 3 ]; then
            svc_arg="${2:-}"; new_profile="${3:-}"
        else
            svc_arg="secondary"; new_profile="${2:-}"
            echo "（未指定服务名，默认设置 Secondary。指定方式: set-profile primary <name>）"
        fi
        if [ -z "$new_profile" ]; then
            echo "用法: $0 set-profile [primary|secondary] <profile-name>"
            echo "当前 Primary: ${PRIMARY_PROFILE}"
            echo "当前 Secondary: ${SECONDARY_PROFILE}"
            echo ""
            echo "可用 profiles:"
            ls "${PROFILES_DIR}"/*.yaml 2>/dev/null | xargs -n1 basename | sed 's/\.yaml$//' || echo "  (无)"
            exit 1
        fi
        case "$svc_arg" in
            primary)      env_var="VLLM_PRIMARY_PROFILE" ;;
            secondary) env_var="VLLM_SECONDARY_PROFILE" ;;
            *)         echo "错误: 未知服务 '${svc_arg}'，可选: primary, secondary"; exit 1 ;;
        esac
        if [[ ! "$new_profile" =~ ^[a-zA-Z0-9_-]+$ ]]; then
            echo "错误: Profile 名称仅允许字母、数字、下划线和连字符"
            exit 1
        fi
        validate_profile "$new_profile" || exit 1
        # 清理 .env 中手动设置的 LOG_DIR（日志目录由 manage.sh 固定为服务名）
        sed -i '/^VLLM_PRIMARY_LOG_DIR=/d' .env
        sed -i '/^VLLM_SECONDARY_LOG_DIR=/d' .env
        _set_env_var "$env_var" "$new_profile"
        # 同步内存变量（确保后续输出使用新值）
        case "$svc_arg" in
            primary)   PRIMARY_PROFILE="$new_profile" ;;
            secondary) SECONDARY_PROFILE="$new_profile" ;;
        esac
        echo "已设置 ${svc_arg} profile: ${new_profile}"
        echo "执行以下命令生效: ./manage.sh restart ${svc_arg}"
        ;;

    detect-topology)
        echo "=== NUMA 拓扑自动检测 ==="
        echo ""

        # 检查 nvidia-smi
        if ! command -v nvidia-smi &>/dev/null; then
            echo "错误: nvidia-smi 不可用"
            exit 1
        fi

        # 检查 GPU 数量
        gpu_count="$(_get_gpu_count)"
        if [ "$gpu_count" -eq 0 ]; then
            echo "错误: 未检测到 GPU"
            exit 1
        fi
        echo "GPU 数量: ${gpu_count}"

        # 检查 NUMA 节点数
        numa_count="$(lscpu 2>/dev/null | grep "^NUMA node(s):" | awk '{print $3}')" || numa_count="?"
        echo "NUMA 节点: ${numa_count}"
        echo ""

        # 获取拓扑
        topo="$(nvidia-smi topo -m 2>/dev/null)"
        if [ -z "$topo" ]; then
            echo "错误: nvidia-smi topo -m 执行失败"
            exit 1
        fi

        echo "$topo"
        echo ""

        # 单 NUMA 节点 → 不需要 NUMA 对齐
        if [ "$numa_count" = "1" ]; then
            total_cpus="$(nproc 2>/dev/null || echo "?")"
            echo "单 NUMA 节点，NUMA 对齐无意义"
            echo "如需减少线程竞争，可按核心数均分（共 ${total_cpus} 逻辑 CPU）:"
            half=$((total_cpus / 2))
            echo "  VLLM_PRIMARY_CPUSET=0-$((half - 1))"
            echo "  VLLM_SECONDARY_CPUSET=${half}-$((total_cpus - 1))"
            exit 0
        fi

        # 多 NUMA: 解析各 GPU 的 CPU Affinity
        primary_gpus="${VLLM_PRIMARY_GPU_IDS:-0,1}"
        secondary_gpus="${VLLM_SECONDARY_GPU_IDS:-2,3}"

        primary_cpuset="$(_parse_gpu_affinity "$topo" "$gpu_count" "$primary_gpus")"
        secondary_cpuset="$(_parse_gpu_affinity "$topo" "$gpu_count" "$secondary_gpus")"

        if [ -z "$primary_cpuset" ] && [ -z "$secondary_cpuset" ]; then
            echo "未能从拓扑中解析 CPU Affinity（可能不支持）"
            exit 0
        fi

        # 校验解析结果
        if [ -n "$primary_cpuset" ]; then
            if ! _validate_cpuset_value "Primary" "$primary_cpuset" >/dev/null 2>&1; then
                echo "警告: Primary 解析结果异常: '${primary_cpuset}'，请手动验证"
            fi
        fi
        if [ -n "$secondary_cpuset" ]; then
            if ! _validate_cpuset_value "Secondary" "$secondary_cpuset" >/dev/null 2>&1; then
                echo "警告: Secondary 解析结果异常: '${secondary_cpuset}'，请手动验证"
            fi
        fi

        # 重叠检测
        dt_primary_ids="$(_validate_cpuset_value "Primary" "$primary_cpuset" 2>/dev/null)"
        dt_secondary_ids="$(_validate_cpuset_value "Secondary" "$secondary_cpuset" 2>/dev/null)"
        if ! _check_cpuset_overlap "$dt_primary_ids" "$dt_secondary_ids" 2>/dev/null; then
            echo "警告: Primary/Secondary CPU 隔离范围重叠（两对 GPU 可能在同一 NUMA 节点）"
            echo ""
        fi

        # 剥离超线程兄弟，生成物理核心版本
        primary_noht="" secondary_noht=""
        if [ -d "/sys/devices/system/cpu/cpu0/topology" ]; then
            primary_ids_raw="$(_validate_cpuset_value "Primary" "$primary_cpuset" 2>/dev/null)"
            secondary_ids_raw="$(_validate_cpuset_value "Secondary" "$secondary_cpuset" 2>/dev/null)"
            primary_noht="$(_compact_cpuset "$(_strip_ht_siblings "$primary_ids_raw")")"
            secondary_noht="$(_compact_cpuset "$(_strip_ht_siblings "$secondary_ids_raw")")"
        fi

        echo "推荐 cpuset 配置:"
        if [ -n "$primary_noht" ]; then
            echo "  [推荐] 仅物理核心（推理负载最优，避免 HT 共享 ALU 竞争）:"
            echo "  VLLM_PRIMARY_CPUSET=${primary_noht}"
            echo "  VLLM_SECONDARY_CPUSET=${secondary_noht}"
            echo ""
            echo "  [可选] 含超线程（完整 CPU Affinity）:"
            echo "  VLLM_PRIMARY_CPUSET=${primary_cpuset}"
            echo "  VLLM_SECONDARY_CPUSET=${secondary_cpuset}"
        else
            echo "  VLLM_PRIMARY_CPUSET=${primary_cpuset}"
            echo "  VLLM_SECONDARY_CPUSET=${secondary_cpuset}"
        fi
        echo ""

        # 实际写入的推荐值（优先物理核心）
        recommend_primary="${primary_noht:-$primary_cpuset}"
        recommend_secondary="${secondary_noht:-$secondary_cpuset}"

        # 检测 GPU 对之间 SYS 拓扑，提示设置 NCCL_P2P_DISABLE
        has_sys=false
        p_first="$(echo "${VLLM_PRIMARY_GPU_IDS:-0,1}" | cut -d, -f1)"
        p_second="$(echo "${VLLM_PRIMARY_GPU_IDS:-0,1}" | cut -d, -f2)"
        topo_val="$(echo "$topo" | awk -v gpu="GPU${p_first}" -v col="$((p_second + 2))" '$1 == gpu {print $col}')"
        [ "$topo_val" = "SYS" ] && has_sys=true
        if ! $has_sys; then
            s_first="$(echo "${VLLM_SECONDARY_GPU_IDS:-2,3}" | cut -d, -f1)"
            s_second="$(echo "${VLLM_SECONDARY_GPU_IDS:-2,3}" | cut -d, -f2)"
            topo_val="$(echo "$topo" | awk -v gpu="GPU${s_first}" -v col="$((s_second + 2))" '$1 == gpu {print $col}')"
            [ "$topo_val" = "SYS" ] && has_sys=true
        fi
        if $has_sys; then
            echo "注意: GPU 对之间存在 SYS 拓扑（跨 NUMA），建议设置:"
            echo "  NCCL_P2P_DISABLE=1  (写入 .env)"
            echo ""
        fi

        echo "写入 .env（推荐物理核心）:"
        if grep -q '^VLLM_PRIMARY_CPUSET=' .env 2>/dev/null; then
            echo "  sed -i 's/^VLLM_PRIMARY_CPUSET=.*/VLLM_PRIMARY_CPUSET=${recommend_primary}/' .env"
            echo "  sed -i 's/^VLLM_SECONDARY_CPUSET=.*/VLLM_SECONDARY_CPUSET=${recommend_secondary}/' .env"
        else
            echo "  echo 'VLLM_PRIMARY_CPUSET=${recommend_primary}' >> .env"
            echo "  echo 'VLLM_SECONDARY_CPUSET=${recommend_secondary}' >> .env"
        fi
        echo ""
        echo "一键应用（仅物理核心）:  ./manage.sh apply-cpuset"
        echo "含超线程方案:            ./manage.sh apply-cpuset --with-ht"
        echo "生效:                    ./manage.sh restart all"
        ;;

    apply-cpuset)
        # 自动检测并写入 .env
        # 默认剥离超线程兄弟（推理负载最优），--with-ht 保留超线程
        strip_ht=true
        if [ "${2:-}" = "--with-ht" ]; then
            strip_ht=false
        fi
        if ! command -v nvidia-smi &>/dev/null; then
            echo "错误: nvidia-smi 不可用"
            exit 1
        fi
        gpu_count="$(_get_gpu_count)"
        numa_count="$(lscpu 2>/dev/null | grep "^NUMA node(s):" | awk '{print $3}')" || numa_count="?"
        topo="$(nvidia-smi topo -m 2>/dev/null)"

        if [ "$numa_count" = "1" ] || [ -z "$topo" ]; then
            total_cpus="$(nproc 2>/dev/null || echo "?")"
            echo "单 NUMA 或无法检测拓扑，不自动配置 cpuset (需要 NUMA 对齐才有意义)"
            echo ""
            if [ "$total_cpus" != "?" ] && [ "$total_cpus" -gt 1 ] 2>/dev/null; then
                half=$((total_cpus / 2))
                echo "如需减少线程竞争，可手动按核心数均分 (共 ${total_cpus} 逻辑 CPU):"
                echo "  VLLM_PRIMARY_CPUSET=0-$((half - 1))"
                echo "  VLLM_SECONDARY_CPUSET=${half}-$((total_cpus - 1))"
                echo ""
                echo "手动写入 .env 后执行: ./manage.sh restart all"
            fi
            exit 0
        fi

        primary_gpus="${VLLM_PRIMARY_GPU_IDS:-0,1}"
        secondary_gpus="${VLLM_SECONDARY_GPU_IDS:-2,3}"
        primary_cpuset="$(_parse_gpu_affinity "$topo" "$gpu_count" "$primary_gpus")"
        secondary_cpuset="$(_parse_gpu_affinity "$topo" "$gpu_count" "$secondary_gpus")"

        if [ -z "$primary_cpuset" ] || [ -z "$secondary_cpuset" ]; then
            missing=""
            [ -z "$primary_cpuset" ] && missing="Primary"
            [ -z "$secondary_cpuset" ] && missing="${missing:+$missing 和 }Secondary"
            echo "错误: 未能解析 ${missing} GPU 的 CPU Affinity"
            exit 1
        fi

        # 校验解析结果的格式和范围 + 重叠检测
        primary_ids="$(_validate_cpuset_value "Primary" "$primary_cpuset")" || exit 1
        secondary_ids="$(_validate_cpuset_value "Secondary" "$secondary_cpuset")" || exit 1
        _check_cpuset_overlap "$primary_ids" "$secondary_ids" || exit 1

        # 剥离超线程兄弟，仅保留物理核心（推理负载默认）
        if [ "$strip_ht" = true ]; then
            noht_primary="$(_strip_ht_siblings "$primary_ids")"
            noht_secondary="$(_strip_ht_siblings "$secondary_ids")"
            if [ -n "$noht_primary" ]; then
                primary_cpuset="$(_compact_cpuset "$noht_primary")"
            fi
            if [ -n "$noht_secondary" ]; then
                secondary_cpuset="$(_compact_cpuset "$noht_secondary")"
            fi
            echo "已剥离超线程 (仅物理核心):"
        else
            echo "保留超线程 (--with-ht):"
        fi

        # 写入 .env（幂等: 存在则更新，不存在则追加）
        if ! grep -q '^# CPU NUMA' .env 2>/dev/null; then
            echo "" >> .env
            echo "# CPU NUMA 隔离（由 apply-cpuset 自动生成）" >> .env
        fi
        _set_env_var VLLM_PRIMARY_CPUSET "$primary_cpuset"
        _set_env_var VLLM_SECONDARY_CPUSET "$secondary_cpuset"
        echo "已写入 .env:"
        echo "  VLLM_PRIMARY_CPUSET=${primary_cpuset}"
        echo "  VLLM_SECONDARY_CPUSET=${secondary_cpuset}"
        echo ""
        echo "生效: ./manage.sh restart all"
        ;;

    boot-delay)
        _boot_delay_if_needed
        ;;

    check-driver)
        _check_nvidia_driver || exit 1
        ;;

    enable-boot)
        echo "=== 配置开机自启动 + 关机保护 ==="
        echo ""

        # 前置检查
        if ! command -v docker &>/dev/null; then
            echo "错误: Docker 未安装"; exit 1
        fi
        if ! systemctl is-active --quiet docker.service 2>/dev/null; then
            echo "错误: Docker 服务未运行"; exit 1
        fi

        template_dir="${SCRIPT_DIR}/config/systemd"

        # 安装 nvidia-persistenced 服务（如可用）
        if [ -f /usr/bin/nvidia-persistenced ]; then
            if [ ! -f "$template_dir/nvidia-persistenced.service" ]; then
                echo "警告: 模板文件不存在: $template_dir/nvidia-persistenced.service，跳过"
            else
                sudo mkdir -p /var/run/nvidia-persistenced
                sudo cp "$template_dir/nvidia-persistenced.service" /etc/systemd/system/
                echo "[OK] nvidia-persistenced.service 已安装"
            fi
        else
            echo "[跳过] nvidia-persistenced 未安装，跳过 GPU Persistence Mode"
        fi

        # 安装 vllm 服务（替换 __PROJECT_DIR__ 占位符）
        vllm_template="$template_dir/vllm.service"
        if [ ! -f "$vllm_template" ]; then
            echo "错误: 模板文件不存在: $vllm_template"; exit 1
        fi
        sed "s|__PROJECT_DIR__|${SCRIPT_DIR}|g" "$vllm_template" | sudo tee /etc/systemd/system/vllm.service > /dev/null
        echo "[OK] vllm.service 已安装 (路径: ${SCRIPT_DIR})"

        # 更新 Docker daemon.json（添加 shutdown-timeout）
        _update_docker_shutdown_timeout 120

        # 重载并启用
        sudo systemctl daemon-reload
        if [ -f /etc/systemd/system/nvidia-persistenced.service ]; then
            sudo systemctl enable nvidia-persistenced.service 2>/dev/null
        fi
        sudo systemctl enable vllm.service
        echo "[OK] 服务已启用"

        # 启动 nvidia-persistenced（不启动 vllm，避免与手动运行冲突）
        if [ -f /etc/systemd/system/nvidia-persistenced.service ]; then
            sudo systemctl start nvidia-persistenced.service 2>/dev/null && \
                echo "[OK] nvidia-persistenced 已启动 (Persistence Mode: On)" || \
                echo "[警告] nvidia-persistenced 启动失败（不影响核心功能）"
        fi

        echo ""
        echo "=== 配置完成 ==="
        echo "  vllm.service:              已启用 (下次开机自动启动)"
        echo "  关机/重启保护:             已启用 (关机前自动优雅停止容器)"
        echo "  Docker shutdown-timeout:   120s"
        if [ -f /etc/systemd/system/nvidia-persistenced.service ]; then
            echo "  nvidia-persistenced:       已启用"
        fi
        echo ""
        echo "  注意: Docker daemon.json 已更新，需重启 Docker 生效:"
        echo "    sudo systemctl restart docker"
        echo "    （会短暂中断容器连接，Docker restart policy 会自动恢复容器）"
        echo ""
        echo "  验证: sudo systemctl status vllm.service"
        echo "  移除: ./manage.sh disable-boot"
        ;;

    disable-boot)
        echo "=== 移除开机自启动 + 关机保护 ==="
        echo ""

        # 停止并禁用 vllm 服务
        if [ -f /etc/systemd/system/vllm.service ]; then
            sudo systemctl stop vllm.service 2>/dev/null || true
            sudo systemctl disable vllm.service 2>/dev/null
            sudo rm /etc/systemd/system/vllm.service
            echo "[OK] vllm.service 已移除"
        else
            echo "[跳过] vllm.service 未安装"
        fi

        # 停止并禁用 nvidia-persistenced
        if [ -f /etc/systemd/system/nvidia-persistenced.service ]; then
            sudo systemctl stop nvidia-persistenced.service 2>/dev/null || true
            sudo systemctl disable nvidia-persistenced.service 2>/dev/null
            sudo rm /etc/systemd/system/nvidia-persistenced.service
            echo "[OK] nvidia-persistenced.service 已移除"
        else
            echo "[跳过] nvidia-persistenced.service 未安装"
        fi

        # 恢复 Docker daemon.json（移除 shutdown-timeout）
        _update_docker_shutdown_timeout 0

        # 重载
        sudo systemctl daemon-reload
        echo ""
        echo "=== 已移除 ==="
        echo "  开机自启动:   已禁用"
        echo "  关机保护:     已移除"
        ;;

    help)
        echo "Qwen3.6-27B-FP8 Docker 部署管理脚本 — 双服务并行版"
        echo ""
        echo "用法: $0 <命令> [参数]"
        echo ""
        echo "服务: primary | secondary | all"
        echo ""
        echo "命令:"
        echo "  start [primary|secondary|all]    启动服务 (默认: all)"
        echo "  stop [primary|secondary|all]     停止服务 (默认: all)"
        echo "  restart [primary|secondary|all]  重启服务 (默认: all)"
        echo "  set-profile [primary|secondary] <name>  设置 profile (持久化到 .env)"
        echo "  config                         查看完整配置预检（无需 Docker 运行）"
        echo "  status [primary|secondary]      查看运行状态（不指定时自动检测）"
        echo "  logs [primary|secondary]        查看日志 (Docker json-file, Ctrl+C 退出)"
        echo "  logfiles [primary|secondary]    查看文件日志 (tail -f, 无需 docker)"
        echo "  build                           构建 Docker 镜像"
        echo "  pull-base                       拉取 CUDA 基础镜像"
        echo "  edit [primary|secondary]        编辑服务 vLLM 配置"
        echo "  detect-topology                 检测 NUMA 拓扑，推荐 cpuset 配置"
        echo "  apply-cpuset [--with-ht]        自动检测并写入 cpuset 到 .env（默认仅物理核心）"
        echo "  enable-boot                     一键配置开机自启 + 关机保护"
        echo "  disable-boot                    移除开机自启 + 关机保护"
        echo "  check-driver                    检查 NVIDIA 驱动健康状态"
        echo ""
        echo "服务分配:"
        echo "  primary   GPU ${VLLM_PRIMARY_GPU_IDS:-0,1}  端口 ${VLLM_PRIMARY_HOST_PORT:-8089}  profile: ${PRIMARY_PROFILE}"
        echo "  secondary GPU ${VLLM_SECONDARY_GPU_IDS:-2,3}  端口 ${VLLM_SECONDARY_HOST_PORT:-8099}  profile: ${SECONDARY_PROFILE}"
        echo ""
        echo "Profile 切换:"
        echo "  ./manage.sh set-profile primary code     # 切换 Primary profile 为 code"
        echo "  ./manage.sh restart primary              # 重启生效"
        echo "  ./manage.sh set-profile instruct         # 切换 Secondary profile (省略服务名)"
        echo "  ./manage.sh restart secondary            # 重启生效"
        echo ""
        echo "注: start all 自动检测 GPU 可用性 — 仅启动配置了可用 GPU 的服务"
        echo "    顺序启动 (先 primary 后 secondary)，确保 secondary 命中热缓存"
        echo "    gpu_memory_utilization 由各 profile YAML 控制 (当前: $(if [ -f "${PROFILES_DIR}/${PRIMARY_PROFILE}.yaml" ]; then get_profile_field "${PRIMARY_PROFILE}" "gpu_memory_utilization"; else echo "N/A"; fi))"
        ;;
    *)
        echo "未知命令: ${CMD}"
        echo "运行 './manage.sh help' 查看用法"
        exit 1
        ;;
esac
