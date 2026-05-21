#!/usr/bin/env python3
"""
长上下文 Streaming TTFT 基准测试
  模式 1: 单请求 TTFT 扫描 — 不同上下文长度的 streaming TTFT/TBT/Decode 曲线
  模式 2: 并发 prefill 测试 — N 请求同时占满指定上下文，测 TTFT 退化和 OOM 边界

用法:
  # TTFT 扫描 (6 个上下文长度)
  python3 scripts/longctx_streaming_bench.py \
    --context-tokens 10000,30000,60000,90000,130000,150000 \
    --config-label "bf16+n=1@152K"

  # 并发 prefill (40K tokens × c=4)
  python3 scripts/longctx_streaming_bench.py \
    --context-tokens 40000 --concurrency 4 \
    --config-label "bf16+n=1@152K"
"""

import argparse, json, time, sys, requests, os, statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

API_URL = "http://localhost:8089/v1/chat/completions"
MODEL = "Qwen3.6-27B-FP8"
CHARS_PER_TOKEN = 1.94

CONTEXT_PARAGRAPHS = [
    "软件架构设计中，分层架构是最经典的模式之一。表现层负责用户界面展示与交互逻辑，"
    "业务逻辑层封装核心领域规则和工作流，数据访问层管理持久化存储的读写操作。"
    "严格的层间依赖关系（上层依赖下层，下层不感知上层）使得每一层可以独立演进和测试。"
    "在实践中，依赖注入框架（如 Spring 的 IoC 容器）进一步解耦了层间具体实现。",

    "数据库查询性能优化通常从索引策略入手。B+Tree 索引适合范围查询和排序场景，"
    "Hash 索引在等值查找时性能最优。复合索引的列顺序必须遵循最左前缀匹配原则，"
    "否则索引无法被有效利用。查询执行计划分析（EXPLAIN）是诊断慢查询的核心手段，"
    "重点关注全表扫描、临时表创建和文件排序等高代价操作。",

    "TCP 协议的三次握手建立连接过程：客户端发送 SYN 包，服务端回复 SYN-ACK，"
    "客户端再发送 ACK 确认。这个设计确保了双方都具备发送和接收能力。"
    "四次挥手断开连接则更为复杂，因为 TCP 是全双工协议，每个方向需要独立关闭。"
    "TIME_WAIT 状态持续 2MSL（最大报文段生存时间），确保最后的 ACK 能到达对端。",

    "容器编排平台 Kubernetes 的核心概念包括 Pod、Service、Deployment 和 ConfigMap。"
    "Pod 是最小调度单元，包含一个或多个共享网络和存储的容器。Service 提供稳定的"
    "网络端点，通过标签选择器将流量路由到匹配的 Pod。Deployment 管理 Pod 的声明式"
    "更新，支持滚动发布和回滚。ConfigMap 将配置与镜像解耦，便于环境间迁移。",

    "机器学习的偏差-方差权衡是模型选择的核心考量。高偏差模型（如线性回归）对训练数据"
    "欠拟合，无法捕获复杂模式。高方差模型（如深层决策树）过拟合训练数据的噪声。"
    "正则化技术（L1/L2 范数惩罚）通过约束参数空间来降低方差。交叉验证是评估模型"
    "泛化能力的标准方法，k 折交叉验证在偏差和计算成本之间取得良好平衡。",

    "分布式系统中的 CAP 定理指出，在网络分区发生时，系统只能在一致性和可用性之间"
    "选择其一。CP 系统（如 ZooKeeper）优先保证一致性，在分区时拒绝部分请求。"
    "AP 系统（如 Cassandra）优先保证可用性，允许暂时返回过期数据。最终一致性模型"
    "通过向量时钟和读修复机制在后台逐步收敛数据状态。",

    "Git 分支管理策略中，Git Flow 模型定义了 main、develop、feature、release 和 "
    "hotfix 五种分支类型。main 分支始终保持可发布状态，develop 是集成分支。"
    "feature 分支从 develop 创建，完成后合并回去。release 分支冻结代码进入测试阶段，"
    "hotfix 从 main 创建用于紧急修复。Trunk-Based Development 则主张少量长期分支。",

    "网络安全防护的纵深防御策略包含多个层次：边界防火墙过滤恶意流量，Web 应用防火墙"
    "（WAF）防御 SQL 注入和 XSS 攻击，入侵检测系统（IDS）监控异常行为。"
    "身份认证层使用 OAuth2.0 和 JWT 实现无状态令牌验证。传输层加密（TLS 1.3）"
    "使用 AEAD 密码套件同时保证数据机密性和完整性。",

    "持续集成与持续交付（CI/CD）流水线的核心原则是自动化和快速反馈。代码提交触发"
    "自动化构建和单元测试，静态分析工具（如 SonarQube）检查代码质量。集成测试在"
    "隔离的 staging 环境中运行。制品通过制品仓库（如 Nexus、Artifactory）管理版本。"
    "蓝绿部署和金丝雀发布是常见的零停机部署策略。",

    "微服务架构中，服务间通信分为同步和异步两种模式。同步通信通常使用 REST 或 gRPC，"
    "gRPC 基于 Protocol Buffers 和 HTTP/2 提供高效的二进制序列化和双向流。"
    "异步通信通过消息队列（如 Kafka、RabbitMQ）实现服务解耦。事件溯源模式将状态变更"
    "记录为不可变事件序列，配合 CQRS 实现读写分离。",

    "内存管理中的垃圾回收算法分为标记-清除、复制和标记-整理三类。标记-清除产生内存碎片，"
    "复制算法将存活对象复制到新空间但浪费一半内存，标记-整理在清除后压缩内存。"
    "现代 JVM 的 G1 收集器将堆划分为等大 Region，优先回收垃圾最多的 Region（Garbage "
    "First），在延迟和吞吐之间可预测地权衡。ZGC 则追求亚毫秒级停顿时间。",

    "函数式编程范式的核心概念包括不可变数据、纯函数和高阶函数。不可变数据消除了共享"
    "状态的竞态条件，使并发编程更安全。纯函数的输出仅依赖输入参数，没有副作用，"
    "便于测试和推理。高阶函数（map、filter、reduce）提供强大的数据变换抽象。"
    "Haskell、Clojure 和 Scala 是代表性的函数式编程语言。",
]

# 并发 prefill 质量验证用的 Needle
NEEDLE = "在本次并发测试中，特殊验证码是 NHB-#{idx:03d}。请记住这个验证码。"
NEEDLE_QUESTION = "文本中提到的特殊验证码是什么？只回答验证码本身。"


def get_api_key():
    """从 .env 文件读取 API key"""
    for path in [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
        ".env",
    ]:
        try:
            with open(path) as f:
                for line in f:
                    if line.startswith("VLLM_PRIMARY_API_KEY="):
                        return line.strip().split("=", 1)[1]
        except Exception:
            pass
    return None


def build_filler(target_chars):
    """构建填充文本到指定字符数"""
    filler = ""
    idx = 0
    while len(filler) < target_chars:
        filler += CONTEXT_PARAGRAPHS[idx % len(CONTEXT_PARAGRAPHS)] + "\n\n"
        idx += 1
    return filler


def build_context_with_needle(target_tokens, needle, needle_position=0.5):
    """构建包含 needle 的长上下文，needle 插入到指定位置"""
    target_chars = int(target_tokens * CHARS_PER_TOKEN)
    filler = build_filler(target_chars)
    insert_pos = int(len(filler) * needle_position)
    content = filler[:insert_pos] + "\n" + needle + "\n" + filler[insert_pos:]
    return content


def measure_streaming_ttft(api_url, api_key, content, max_completion_tokens=50,
                           expect_needle=None):
    """发送单个 streaming 请求，测量 TTFT/TBT/Decode"""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_completion_tokens": max_completion_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    token_times = []
    ttft = None
    prompt_tokens = 0
    completion_tokens = 0
    full_text = ""
    error_msg = None

    start = time.time()
    try:
        resp = requests.post(api_url, json=payload, headers=headers, stream=True, timeout=600)
        if resp.status_code != 200:
            try:
                err = resp.json()
                error_msg = str(err.get("error", {}).get("message", ""))[:80]
            except Exception:
                error_msg = f"HTTP {resp.status_code}"
            return {"ok": False, "error": error_msg, "elapsed": time.time() - start}

        for line in resp.iter_lines():
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            line = line.strip()
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            # 提取 usage
            usage = data.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                completion_tokens = usage.get("completion_tokens", completion_tokens)

            choices = data.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content_chunk = delta.get("content", "")
            reasoning_chunk = delta.get("reasoning_content", "")

            text = content_chunk or reasoning_chunk
            if text:
                now = time.time()
                token_times.append(now)
                full_text += text
                if ttft is None:
                    ttft = now - start

        elapsed = time.time() - start
    except requests.exceptions.Timeout:
        return {"ok": False, "error": "超时 (>600s)", "elapsed": time.time() - start}
    except Exception as e:
        return {"ok": False, "error": str(e)[:80], "elapsed": time.time() - start}

    # 计算指标
    decode_tokens = max(completion_tokens, len(token_times))
    if ttft is None:
        ttft = elapsed

    # TBT: 前 10 个 token 的平均间隔
    tbt = None
    if len(token_times) >= 3:
        intervals = [token_times[i+1] - token_times[i] for i in range(min(10, len(token_times)-1))]
        tbt = statistics.mean(intervals) * 1000  # ms

    decode_time = elapsed - ttft if ttft < elapsed else 0.001
    decode_rate = (decode_tokens - 1) / decode_time if decode_time > 0 and decode_tokens > 1 else 0

    # Needle 质量验证
    needle_ok = None
    if expect_needle:
        needle_ok = expect_needle in full_text

    return {
        "ok": True,
        "ttft": round(ttft, 3),
        "tbt_ms": round(tbt, 1) if tbt else None,
        "decode_rate": round(decode_rate, 1),
        "decode_tokens": decode_tokens,
        "prompt_tokens": prompt_tokens,
        "elapsed": round(elapsed, 2),
        "needle_ok": needle_ok,
    }


def run_ttft_sweep(context_tokens_list, api_url, api_key, max_model_len, config_label):
    """模式 1: 不同上下文长度的 streaming TTFT 扫描"""
    print(f"\n{'='*80}")
    print(f"  长上下文 Streaming TTFT 扫描 — {config_label}")
    print(f"  上下文长度: {[f'{t//1000}K' for t in context_tokens_list]}")
    print(f"  max_model_len={max_model_len:,}")
    print(f"{'='*80}")

    results = []
    print(f"\n  {'ctx tok':>10} | {'prompt tok':>10} | {'TTFT(s)':>8} | {'TBT(ms)':>8} | "
          f"{'Decode':>8} | {'elapsed':>8} | 备注")
    print(f"  {'─'*80}")

    for target in context_tokens_list:
        target_chars = int(target * CHARS_PER_TOKEN)
        filler = build_filler(target_chars)
        content = filler + "\n请回答: 1"

        r = measure_streaming_ttft(api_url, api_key, content)
        results.append({"target": target, **r})

        if r["ok"]:
            tbt_str = f"{r['tbt_ms']:.1f}" if r["tbt_ms"] else "—"
            print(f"  {target:>10,} | {r['prompt_tokens']:>10,} | {r['ttft']:>8.3f} | "
                  f"{tbt_str:>8} | {r['decode_rate']:>7.1f}/s | {r['elapsed']:>7.2f}s |")
        else:
            print(f"  {target:>10,} | {'—':>10} | {'—':>8} | {'—':>8} | "
                  f"{'—':>8} | {r['elapsed']:>7.2f}s | {r['error']}")

    # 汇总
    ok_results = [r for r in results if r["ok"]]
    print(f"\n{'─'*80}")
    if ok_results:
        print(f"  扫描完成: {len(ok_results)}/{len(results)} 成功")
        print(f"  TTFT 范围: {min(r['ttft'] for r in ok_results):.2f}s ~ "
              f"{max(r['ttft'] for r in ok_results):.2f}s")
        if any(r["tbt_ms"] for r in ok_results):
            tbts = [r["tbt_ms"] for r in ok_results if r["tbt_ms"]]
            print(f"  TBT 范围: {min(tbts):.1f}ms ~ {max(tbts):.1f}ms")
    else:
        print(f"  扫描失败: 0/{len(results)} 成功")
    print(f"{'─'*80}\n")

    return results


def run_concurrent_prefill(context_tokens, concurrency, api_url, api_key,
                           max_model_len, config_label):
    """模式 2: 并发 prefill 测试"""
    print(f"\n{'='*80}")
    print(f"  并发 Prefill 测试 — {config_label}")
    print(f"  上下文: {context_tokens//1000}K tokens × {concurrency} 请求 "
          f"(总需求: {context_tokens*concurrency//1000}K)")
    print(f"  max_model_len={max_model_len:,}")
    print(f"{'='*80}")

    def send_request(idx):
        needle_code = f"NHB-{idx:03d}"
        needle = f"在本次并发测试中，特殊验证码是 {needle_code}。请记住这个验证码。"
        target_chars = int(context_tokens * CHARS_PER_TOKEN)
        filler = build_filler(target_chars)
        # 把 needle 放在中间位置
        mid = len(filler) // 2
        content = filler[:mid] + "\n" + needle + "\n" + filler[mid:]
        content += f"\n文本中提到的特殊验证码是什么？只回答验证码本身。"

        return (idx, needle_code,
                measure_streaming_ttft(api_url, api_key, content,
                                       expect_needle=needle_code))

    wall_start = time.time()
    per_req = [None] * concurrency
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(send_request, i) for i in range(concurrency)]
        for f in as_completed(futs):
            idx, needle_code, r = f.result()
            per_req[idx] = {"idx": idx, "needle": needle_code, **r}
    wall_elapsed = time.time() - wall_start

    # 输出
    print(f"\n  {'#':>3} | {'prompt tok':>10} | {'TTFT(s)':>8} | {'TBT(ms)':>8} | "
          f"{'Decode':>8} | {'elapsed':>8} | {'Needle':>6} | 备注")
    print(f"  {'─'*80}")

    oom_count = 0
    needle_pass = 0
    ttfts = []
    for r in per_req:
        if r["ok"]:
            tbt_str = f"{r['tbt_ms']:.1f}" if r["tbt_ms"] else "—"
            needle_str = "PASS" if r.get("needle_ok") else "FAIL"
            if r.get("needle_ok"):
                needle_pass += 1
            ttfts.append(r["ttft"])
            print(f"  {r['idx']:>3} | {r['prompt_tokens']:>10,} | {r['ttft']:>8.3f} | "
                  f"{tbt_str:>8} | {r['decode_rate']:>7.1f}/s | {r['elapsed']:>7.2f}s | "
                  f"{needle_str:>6} |")
        else:
            oom_count += 1
            print(f"  {r['idx']:>3} | {'—':>10} | {'—':>8} | {'—':>8} | "
                  f"{'—':>8} | {r['elapsed']:>7.2f}s | {'—':>6} | {r['error']}")

    print(f"\n  {'─'*80}")
    print(f"  Wall: {wall_elapsed:.2f}s | OOM: {oom_count}/{concurrency} | "
          f"Needle通过: {needle_pass}/{concurrency - oom_count}")
    if ttfts:
        print(f"  TTFT: avg={statistics.mean(ttfts):.2f}s min={min(ttfts):.2f}s "
              f"max={max(ttfts):.2f}s")
    print(f"{'─'*80}\n")

    return {
        "concurrency": concurrency,
        "context_tokens": context_tokens,
        "wall_elapsed": round(wall_elapsed, 2),
        "oom_count": oom_count,
        "needle_pass": needle_pass,
        "per_request": per_req,
        "avg_ttft": round(statistics.mean(ttfts), 3) if ttfts else None,
        "max_ttft": round(max(ttfts), 3) if ttfts else None,
    }


def main():
    parser = argparse.ArgumentParser(description="长上下文 Streaming TTFT 基准测试")
    parser.add_argument("--context-tokens", required=True,
                        help="上下文 tokens 数，逗号分隔 (如 10000,60000,150000)")
    parser.add_argument("--concurrency", type=int, default=1, choices=[1, 2, 3, 4],
                        help="并发级别 (默认 1)")
    parser.add_argument("--max-model-len", type=int, default=155648,
                        help="max_model_len (默认 155648=152K)")
    parser.add_argument("--api-url", default="http://localhost:8089/v1/chat/completions",
                        help="API endpoint URL (默认 Primary 8089)")
    parser.add_argument("--config-label", default="",
                        help="配置标签 (如 'bf16+n=1@152K')")
    args = parser.parse_args()

    global API_URL
    API_URL = args.api_url

    api_key = get_api_key()
    if not api_key:
        print("ERROR: 无法获取 API key"); sys.exit(1)

    tokens_list = [int(t.strip()) for t in args.context_tokens.split(",")]

    # 预热
    print("  预热请求...", end=" ", flush=True)
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    try:
        requests.post(API_URL, json={
            "model": MODEL, "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 5, "temperature": 0.0,
        }, headers=headers, timeout=60)
    except Exception:
        pass
    print("完成")

    if args.concurrency == 1 and len(tokens_list) > 1:
        # 模式 1: TTFT 扫描
        run_ttft_sweep(tokens_list, args.api_url, api_key,
                       args.max_model_len, args.config_label)
    elif args.concurrency == 1 and len(tokens_list) == 1:
        # 单请求单上下文 (也是扫描，只是只有一个点)
        run_ttft_sweep(tokens_list, args.api_url, api_key,
                       args.max_model_len, args.config_label)
    else:
        # 模式 2: 并发 prefill
        if len(tokens_list) != 1:
            print("ERROR: 并发模式只支持单一上下文长度"); sys.exit(1)
        run_concurrent_prefill(tokens_list[0], args.concurrency, args.api_url, api_key,
                               args.max_model_len, args.config_label)


if __name__ == "__main__":
    main()
