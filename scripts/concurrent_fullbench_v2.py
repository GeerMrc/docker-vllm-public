#!/usr/bin/env python3
"""
增强版完整并发基准+质量测试 — 4 Phase 结构
  Phase A: 非Streaming 吞吐 (500 tok × 4 req)
  Phase B: Streaming TTFT/Decode (300 tok × c req)
  Phase C: 增强基础质量测试 (Q1-Q8)
  Phase D: 上下文质量测试 (Q9-Q12, Needle-in-Haystack + 跨文档推理)

用法:
  python3 scripts/concurrent_fullbench_v2.py -c 1 --context-label 152K --config-label "bf16+n=1"
  python3 scripts/concurrent_fullbench_v2.py -c 4 --context-label 152K --config-label "bf16+n=2"
  python3 scripts/concurrent_fullbench_v2.py -c 1 --no-context-tests  # 跳过 Phase D
"""

import argparse, json, time, sys, requests, os
from concurrent.futures import ThreadPoolExecutor, as_completed

API_URL = "http://localhost:8089/v1/chat/completions"
MODEL = "Qwen3.6-27B-FP8"

# ═══════════════════════════════════════════════════════════════
# Phase A/B: 吞吐与流式测试 prompts
# ═══════════════════════════════════════════════════════════════

THROUGHPUT_PROMPT = {"role": "user", "content":
    "详细解释深度学习中 Transformer 架构的核心组件，包括自注意力机制、多头注意力、位置编码、"
    "前馈网络和层归一化。对每个组件说明原理和作用。"}

STREAMING_PROMPT = {"role": "user", "content":
    "详细解释 Transformer 架构中自注意力机制的数学原理，包括 Q/K/V 矩阵计算、"
    "缩放点积注意力、多头注意力的并行计算方式。"}

# ═══════════════════════════════════════════════════════════════
# Phase C: 增强基础质量测试 (Q1-Q8)
# ═══════════════════════════════════════════════════════════════

QUALITY_PROMPTS = [
    # ── 原有 Q1-Q4 ──
    {
        "name": "Q1-算术",
        "messages": [{"role": "user", "content": "2+3=? 只回答数字。"}],
        "max_tokens": 50, "temp": 0.0,
        "check": lambda c: "5" in c,
        "desc": "包含5",
    },
    {
        "name": "Q2-事实",
        "messages": [{"role": "user", "content": "Python编程语言的创始人是谁？用一句话回答。"}],
        "max_tokens": 100, "temp": 0.0,
        "check": lambda c: "Guido" in c or "范罗苏姆" in c,
        "desc": "Guido/范罗苏姆",
    },
    {
        "name": "Q3-推理",
        "messages": [
            {"role": "system", "content": "你是代码助手。精确计算。"},
            {"role": "user", "content": "令 a=3, b=7"},
            {"role": "assistant", "content": "a=3, b=7"},
            {"role": "user", "content": "令 c=a*b+2"},
            {"role": "assistant", "content": "c=23"},
            {"role": "user", "content": "令 d=c-a+b"},
            {"role": "assistant", "content": "d=27"},
            {"role": "user", "content": "求 (a+b)*(c-d)"},
        ],
        "max_tokens": 200, "temp": 0.0,
        "check": lambda c: "-40" in c,
        "desc": "结果-40",
    },
    {
        "name": "Q4-工具",
        "messages": [
            {"role": "system", "content": "你是助手，可调用：get_weather(city), calculate(expr)。需要时用JSON。"},
            {"role": "user", "content": "北京天气？帮我算 15*28+367。"},
        ],
        "max_tokens": 300, "temp": 0.0,
        "check": lambda c: ("get_weather" in c or "天气" in c or "beijing" in c.lower()) and ("calculate" in c or str(15*28+367) in c),
        "desc": "天气+计算(787)",
    },
    # ── 新增 Q5-Q8 ──
    {
        "name": "Q5-多步推理",
        "messages": [
            {"role": "system", "content": "你是数学助手，精确计算每一步。"},
            {"role": "user", "content":
                "设 x = 12, y = x * 3 - 5。\n"
                "设 z = y / 3.1 (四舍五入到整数)。\n"
                "如果 z 是偶数，则 w = z * 2，否则 w = z + 7。\n"
                "求 w 的值。请逐步计算并给出最终结果。"},
        ],
        "max_tokens": 300, "temp": 0.0,
        "check": lambda c: "20" in c,  # x=12, y=31, z=10(偶数), w=20
        "desc": "结果w=20",
    },
    {
        "name": "Q6-结构化生成",
        "messages": [
            {"role": "user", "content":
                "请列举分布式系统的5个核心设计原则。要求：\n"
                "1. 每个原则用编号列表（1-5）\n"
                "2. 每个原则先写名称（加粗格式），再用一句话解释\n"
                "3. 最后总结一句话"},
        ],
        "max_tokens": 500, "temp": 0.3,
        "check": lambda c: (
            # 检查编号列表存在 (1-5)
            all(str(i) in c for i in range(1, 6)) and
            # 检查总结存在
            ("总结" in c or "综上" in c or "总之" in c or "总而言之" in c)
        ),
        "desc": "5项编号+总结",
    },
    {
        "name": "Q7-指令遵循",
        "messages": [
            {"role": "system", "content":
                "你是一个回答机器。必须严格遵循以下格式规则：\n"
                "1. 回答以 'ANSWER:' 开头\n"
                "2. 答案后面用 'EXPLANATION:' 给出解释\n"
                "3. 不要包含任何其他格式或多余文字"},
            {"role": "user", "content": "地球到月球的平均距离大约是多少公里？"},
        ],
        "max_tokens": 200, "temp": 0.0,
        "check": lambda c: "ANSWER:" in c and "EXPLANATION:" in c and ("384" in c or "38万" in c),
        "desc": "ANSWER:+EXPLANATION:+384",
    },
    {
        "name": "Q8-复杂工具调用",
        "messages": [
            {"role": "system", "content":
                "你是助手，可调用以下工具：\n"
                "1. search_database(query, limit) - 搜索数据库\n"
                "2. calculate(expression) - 计算数学表达式\n"
                "3. send_email(to, subject, body) - 发送邮件\n"
                "需要时用JSON格式调用工具。"},
            {"role": "user", "content":
                "帮我做三件事：\n"
                "1. 搜索数据库查找 '2024年销售额'，限制10条结果\n"
                "2. 计算 (1520 + 3780) * 0.15\n"
                "3. 把结果发给 boss@company.com，主题 '季度报告'"},
        ],
        "max_tokens": 500, "temp": 0.0,
        "check": lambda c: (
            ("search_database" in c or "2024" in c) and
            ("795" in c or "calculate" in c) and  # (1520+3780)*0.15 = 795
            ("send_email" in c or "boss@company" in c)
        ),
        "desc": "search+calculate(795)+email",
    },
]

# ═══════════════════════════════════════════════════════════════
# Phase D: 上下文质量测试 — 填充文本
# ═══════════════════════════════════════════════════════════════

# 12 段多样化中文技术段落，每段 ~150-200 tokens
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

# Phase D 测试定义
CONTEXT_TESTS = [
    {
        "name": "Q9-Needle(前10%)",
        "needle": "在本次测试中，特殊验证码是 XJ-84729。请记住这个验证码。",
        "question": "文本中提到的特殊验证码是什么？只回答验证码本身。",
        "position": 0.10,
        "check": lambda c: "XJ-84729" in c,
        "desc": "检索XJ-84729",
    },
    {
        "name": "Q10-Needle(中50%)",
        "needle": "项目负责人确认：项目 Zeta 的预算批复为 3847 万元，预计 2026 年 9 月完工。",
        "question": "项目 Zeta 的预算是多少万元？什么时候完工？简短回答。",
        "position": 0.50,
        "check": lambda c: "3847" in c and ("9" in c or "九月" in c or "Sept" in c),
        "desc": "3847万+9月",
    },
    {
        "name": "Q11-Needle(尾90%)",
        "needle": "安全审计报告编号: SEC-AUD-2026-0715，风险等级: 中等，审计人: 李明远。",
        "question": "安全审计报告的编号是什么？风险等级是什么？审计人是谁？简短回答。",
        "position": 0.90,
        "check": lambda c: "0715" in c and ("中等" in c or "medium" in c.lower()) and ("李明远" in c),
        "desc": "SEC-AUD-0715+中等+李明远",
    },
    {
        "name": "Q12-跨文档推理",
        "docs": [
            {"id": "DOC-A", "content": "项目 Phoenix 由 Dr. Sarah Chen 于 2019 年在北京实验室启动，初期团队 12 人。"},
            {"id": "DOC-B", "content": "项目 Dragonfly 于 2021 年在杭州成立，由 Dr. James Liu 领导，专注于自然语言处理。"},
            {"id": "DOC-C", "content": "2023 年，Phoenix 和 Dragonfly 合并为 Phoenix-D 联合项目，整合了双方技术栈。"},
            {"id": "DOC-D", "content": "Dr. Sarah Chen 于 2022 年调任杭州实验室，担任首席研究员。"},
            {"id": "DOC-E", "content": "杭州实验室现有 85 名研究人员，是公司最大的 AI 研究基地。"},
            {"id": "DOC-F", "content": "Phoenix-D 联合项目 2024 年发表论文 17 篇，申请专利 5 项。"},
        ],
        "question": "与 Dragonfly 合并的项目由谁启动？该人后来调到了哪个实验室？该实验室有多少名研究人员？",
        "check": lambda c: ("Sarah" in c or "Chen" in c or "陈" in c) and ("杭州" in c or "Hangzhou" in c.lower()) and "85" in c,
        "desc": "Chen+杭州+85人",
    },
    {
        "name": "Q13-指令召回",
        "instruction": "【重要规则】在回答关于数字的问题时，你必须：(1) 先用中文写出数字，(2) 然后在括号中给出阿拉伯数字。例如：三 (3)。",
        "question": "太阳系有几颗行星？",
        "check": lambda c: ("八" in c or "8" in c) and ("(" in c or "（" in c),  # 八 (8)
        "desc": "中文数字+阿拉伯数字格式",
    },
]


def get_api_key():
    """从 .env 文件读取 API key"""
    try:
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
        with open(env_path) as f:
            for line in f:
                if line.startswith("VLLM_PRIMARY_API_KEY="):
                    return line.strip().split("=", 1)[1]
    except Exception:
        pass
    # fallback: 当前目录 .env
    try:
        with open(".env") as f:
            for line in f:
                if line.startswith("VLLM_PRIMARY_API_KEY="):
                    return line.strip().split("=", 1)[1]
    except Exception:
        pass
    return None


def warmup(api_key):
    """发送一个短请求预热，消除首次请求的延迟毛刺"""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "max_completion_tokens": 5,
        "temperature": 0.0,
    }
    try:
        requests.post(API_URL, json=payload, headers=headers, timeout=60)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════
# Phase A: 非Streaming 吞吐
# ═══════════════════════════════════════════════════════════════

def bench_throughput(concurrency, api_key, n_requests=4):
    payload = {
        "model": MODEL,
        "messages": [THROUGHPUT_PROMPT],
        "max_completion_tokens": 500,
        "temperature": 0.7,
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    wall_start = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(lambda: (time.time(), requests.post(API_URL, json=payload, headers=headers, timeout=300).json(), time.time())) for _ in range(n_requests)]
        for f in as_completed(futs):
            s, data, e = f.result()
            usage = data.get("usage", {})
            results.append({
                "start": s - wall_start, "elapsed": e - s,
                "completion_tokens": usage.get("completion_tokens", 0),
                "prompt_tokens": usage.get("prompt_tokens", 0),
            })

    wall = time.time() - wall_start
    total_completion = sum(r["completion_tokens"] for r in results)
    total_elapsed = sum(r["elapsed"] for r in results)
    avg_latency = total_elapsed / len(results)

    return {
        "wall_s": round(wall, 2),
        "total_completion": total_completion,
        "wall_tok_s": round(total_completion / wall, 1),
        "avg_latency_s": round(avg_latency, 2),
        "per_req_tok_s": round(total_completion / total_elapsed, 1) if total_elapsed > 0 else 0,
    }


# ═══════════════════════════════════════════════════════════════
# Phase B: Streaming TTFT/Decode
# ═══════════════════════════════════════════════════════════════

def bench_streaming(concurrency, api_key):
    payload = {
        "model": MODEL,
        "messages": [STREAMING_PROMPT],
        "max_completion_tokens": 300,
        "temperature": 0.7,
        "stream": True,
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    def do_stream(req_id):
        start = time.time()
        first_token = None
        tokens = 0
        resp = requests.post(API_URL, json=payload, headers=headers, stream=True, timeout=300)
        for line in resp.iter_lines():
            line = (line.decode() if isinstance(line, bytes) else line).strip()
            if not line or not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try:
                data = json.loads(line[6:])
                delta = data.get("choices", [{}])[0].get("delta", {})
                if delta.get("content") or delta.get("reasoning"):
                    if first_token is None:
                        first_token = time.time()
                    tokens += 1
            except Exception:
                pass
        elapsed = time.time() - start
        ttft = first_token - start if first_token else None
        decode_time = elapsed - ttft if ttft else None
        return {"id": req_id, "ttft": ttft, "tokens": tokens, "elapsed": elapsed, "decode_time": decode_time}

    wall_start = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(do_stream, i) for i in range(concurrency)]
        for f in as_completed(futs):
            results.append(f.result())

    wall = time.time() - wall_start
    ttfts = [r["ttft"] for r in results if r["ttft"]]
    decodes = [r["tokens"] / r["decode_time"] for r in results if r["decode_time"] and r["tokens"] > 0]
    total_tokens = sum(r["tokens"] for r in results)

    return {
        "wall_s": round(wall, 2),
        "avg_ttft_s": round(sum(ttfts) / len(ttfts), 3) if ttfts else None,
        "avg_decode_tok_s": round(sum(decodes) / len(decodes), 1) if decodes else None,
        "wall_tok_s": round(total_tokens / wall, 1),
    }


# ═══════════════════════════════════════════════════════════════
# Phase C: 增强基础质量测试
# ═══════════════════════════════════════════════════════════════

def bench_quality(concurrency, api_key):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    def do_quality(pcfg, req_id):
        payload = {
            "model": MODEL,
            "messages": pcfg["messages"],
            "max_completion_tokens": pcfg["max_tokens"],
            "temperature": pcfg["temp"],
        }
        start = time.time()
        resp = requests.post(API_URL, json=payload, headers=headers, timeout=300)
        elapsed = time.time() - start
        data = resp.json()
        msg = data["choices"][0]["message"]
        content = msg.get("content", "") or ""
        usage = data["usage"]
        passed = pcfg["check"](content)
        return {
            "id": req_id, "name": pcfg["name"], "elapsed": round(elapsed, 2),
            "tokens": usage["completion_tokens"], "content": content,
            "passed": passed, "desc": pcfg["desc"],
        }

    tasks = []
    for ci in range(concurrency):
        for pidx, pcfg in enumerate(QUALITY_PROMPTS):
            tasks.append((pcfg, f"slot{ci}-{pidx}"))

    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = {pool.submit(do_quality, cfg, rid): (cfg, rid) for cfg, rid in tasks}
        for f in as_completed(futs):
            results.append(f.result())

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    by_name = {}
    for r in results:
        by_name.setdefault(r["name"], []).append(r)

    per_prompt = {}
    for name, items in by_name.items():
        p = sum(1 for i in items if i["passed"])
        contents = [i["content"].strip() for i in items]
        unique = len(set(contents))
        per_prompt[name] = {
            "pass": f"{p}/{len(items)}",
            "consistent": unique == 1,
            "sample": contents[0][:80] if contents else "",
        }

    return {"total_pass": f"{passed}/{total}", "per_prompt": per_prompt, "details": results}


# ═══════════════════════════════════════════════════════════════
# Phase D: 上下文质量测试
# ═══════════════════════════════════════════════════════════════

def build_context(target_chars, needle, position):
    """构建长上下文: 循环填充段落 + 在指定位置插入 needle"""
    filler = ""
    idx = 0
    while len(filler) < target_chars:
        filler += CONTEXT_PARAGRAPHS[idx % len(CONTEXT_PARAGRAPHS)] + "\n\n"
        idx += 1

    split_pos = int(len(filler) * position)
    return filler[:split_pos] + "\n" + needle + "\n" + filler[split_pos:]


def build_crossdoc_context(target_chars, docs):
    """构建跨文档推理上下文: 填充 + 嵌入多个文档 + 更多填充"""
    # 前半填充
    half = target_chars // 2
    prefix = ""
    idx = 0
    while len(prefix) < half:
        prefix += CONTEXT_PARAGRAPHS[idx % len(CONTEXT_PARAGRAPHS)] + "\n\n"
        idx += 1

    # 文档区
    doc_section = "\n--- 参考资料 ---\n\n"
    for d in docs:
        doc_section += f"[{d['id']}] {d['content']}\n\n"
    doc_section += "--- 资料结束 ---\n\n"

    # 后半填充
    suffix = ""
    while len(prefix) + len(doc_section) + len(suffix) < target_chars:
        suffix += CONTEXT_PARAGRAPHS[idx % len(CONTEXT_PARAGRAPHS)] + "\n\n"
        idx += 1

    return prefix + doc_section + suffix


def build_instruction_recall_context(target_chars, instruction, question):
    """构建指令召回上下文: 开头放指令 + 大量填充 + 末尾提问"""
    prefix = instruction + "\n\n"
    filler = ""
    idx = 0
    while len(filler) < target_chars - len(prefix) - 200:
        filler += CONTEXT_PARAGRAPHS[idx % len(CONTEXT_PARAGRAPHS)] + "\n\n"
        idx += 1
    return prefix + filler + question


def get_context_tokens_for_concurrency(c):
    """根据并发级别计算每个请求的目标上下文 token 数"""
    # 152K at 1.25x concurrency = ~194K total KV token budget
    max_model_len = 155648
    max_concurrency = 1.25
    total_budget = int(max_model_len * max_concurrency)
    # 每请求: 不超过 max_model_len-500, 且总占用不超过 KV 预算
    per_req = min(max_model_len - 500, total_budget // c - 500)
    return max(per_req, 2000)


def bench_context(concurrency, api_key):
    """Phase D: 上下文质量测试"""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    target_tokens = get_context_tokens_for_concurrency(concurrency)
    # 实测比率: 1.94 chars/token (中文技术文本 + Qwen tokenizer)
    # 保守: 用目标的 85% 作为填充，留余量给 needle + question + output
    target_chars = int(target_tokens * 0.85 * 1.94)

    ctx_results = []

    for tcfg in CONTEXT_TESTS:
        # 构建上下文
        if tcfg["name"] == "Q12-跨文档推理":
            context = build_crossdoc_context(target_chars, tcfg["docs"])
        elif tcfg["name"] == "Q13-指令召回":
            context = build_instruction_recall_context(target_chars, tcfg["instruction"], tcfg["question"])
        else:
            context = build_context(target_chars, tcfg["needle"], tcfg["position"])

        # 构建 messages
        if tcfg["name"] == "Q12-跨文档推理":
            user_content = context + "\n\n根据以上资料回答: " + tcfg["question"]
        elif tcfg["name"] == "Q13-指令召回":
            user_content = context
        else:
            user_content = context + "\n\n" + tcfg["question"]

        messages = [{"role": "user", "content": user_content}]

        # 并发发送
        def do_ctx(req_id):
            payload = {
                "model": MODEL,
                "messages": messages,
                "max_completion_tokens": 200,
                "temperature": 0.0,
            }
            start = time.time()
            try:
                resp = requests.post(API_URL, json=payload, headers=headers, timeout=600)
                elapsed = time.time() - start
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                passed = tcfg["check"](content)
                error = None
            except Exception as e:
                elapsed = time.time() - start
                content = f"ERROR: {e}"
                prompt_tokens = 0
                completion_tokens = 0
                passed = False
                error = str(e)

            return {
                "id": req_id, "name": tcfg["name"], "elapsed": round(elapsed, 2),
                "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                "content": content, "passed": passed, "desc": tcfg["desc"],
                "error": error,
            }

        tasks = [(i,) for i in range(concurrency)]
        c_results = []
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = [pool.submit(do_ctx, i) for i in range(concurrency)]
            for f in as_completed(futs):
                c_results.append(f.result())

        ctx_results.extend(c_results)

    # 汇总
    total = len(ctx_results)
    passed = sum(1 for r in ctx_results if r["passed"])
    by_name = {}
    for r in ctx_results:
        by_name.setdefault(r["name"], []).append(r)

    per_test = {}
    for name, items in by_name.items():
        p = sum(1 for i in items if i["passed"])
        contents = [i["content"].strip() for i in items]
        avg_prompt_tok = sum(i["prompt_tokens"] for i in items) / len(items) if items else 0
        unique = len(set(contents))
        per_test[name] = {
            "pass": f"{p}/{len(items)}",
            "consistent": unique == 1,
            "avg_prompt_tokens": int(avg_prompt_tok),
            "sample": contents[0][:100] if contents else "",
        }

    return {
        "total_pass": f"{passed}/{total}",
        "per_test": per_test,
        "target_context_tokens": target_tokens,
        "details": ctx_results,
    }


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="增强版完整并发基准+质量测试 v2")
    parser.add_argument("-c", "--concurrency", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--context-label", default="152K", help="上下文标签 (如 132K, 152K)")
    parser.add_argument("--config-label", default="", help="配置标签 (如 'bf16+n=1')")
    parser.add_argument("--no-context-tests", action="store_true", help="跳过 Phase D 上下文测试")
    args = parser.parse_args()
    c = args.concurrency

    api_key = get_api_key()
    if not api_key:
        print("ERROR: 无法获取 API key"); sys.exit(1)

    config_str = f" | {args.config_label}" if args.config_label else ""
    print(f"\n{'='*80}")
    print(f"  增强版完整基准测试 v2 — {args.context_label} context{config_str} — 并发={c}")
    print(f"{'='*80}")

    # 预热
    print("\n  预热请求...", end=" ", flush=True)
    warmup(api_key)
    print("完成")

    # ── Phase A ──
    print(f"\n{'─'*80}")
    print(f"  A: 非Streaming 吞吐 (500 tok × 4 req)")
    print(f"{'─'*80}")
    tp = bench_throughput(c, api_key)
    print(f"  Wall: {tp['wall_s']}s")
    print(f"  总吞吐: {tp['wall_tok_s']} tok/s")
    print(f"  单请求: {tp['per_req_tok_s']} tok/s")
    print(f"  平均延迟: {tp['avg_latency_s']}s")

    # ── Phase B ──
    print(f"\n{'─'*80}")
    print(f"  B: Streaming TTFT/Decode (300 tok × {c} req)")
    print(f"{'─'*80}")
    st = bench_streaming(c, api_key)
    print(f"  Wall: {st['wall_s']}s")
    print(f"  TTFT: {st['avg_ttft_s']}s")
    print(f"  Decode: {st['avg_decode_tok_s']} tok/s/req")
    print(f"  Wall吞吐: {st['wall_tok_s']} tok/s")

    # ── Phase C ──
    print(f"\n{'─'*80}")
    print(f"  C: 基础质量测试 (Q1-Q8 × {c} 份)")
    print(f"{'─'*80}")
    q = bench_quality(c, api_key)
    print(f"  总通过率: {q['total_pass']}")
    for name, info in q['per_prompt'].items():
        cons = "一致" if info["consistent"] else "不一致"
        print(f"  {name}: {info['pass']} PASS, 一致性: {cons}")
        print(f"    样例: {info['sample'][:60]}...")

    # ── Phase D ──
    ctx = None
    if not args.no_context_tests:
        print(f"\n{'─'*80}")
        print(f"  D: 上下文质量测试 (Q9-Q13 × {c} 份, 目标 ~{get_context_tokens_for_concurrency(c):,} tokens/req)")
        print(f"{'─'*80}")
        ctx = bench_context(c, api_key)
        print(f"  总通过率: {ctx['total_pass']}")
        print(f"  目标上下文: ~{ctx['target_context_tokens']:,} tokens/req")
        for name, info in ctx['per_test'].items():
            cons = "一致" if info["consistent"] else "不一致"
            print(f"  {name}: {info['pass']} PASS, 实际输入: {info['avg_prompt_tokens']:,} tokens, 一致性: {cons}")
            print(f"    样例: {info['sample'][:60]}...")
    else:
        print(f"\n  D: 上下文质量测试 — 已跳过 (--no-context-tests)")

    # ── 汇总 ──
    print(f"\n{'═'*80}")
    ctx_str = f" | 上下文质量={ctx['total_pass']}" if ctx else ""
    print(f"  汇总: {args.config_label} | ctx={args.context_label} | c={c}")
    print(f"    吞吐={tp['wall_tok_s']} tok/s | TTFT={st['avg_ttft_s']}s | Decode={st['avg_decode_tok_s']} tok/s/req")
    print(f"    基础质量={q['total_pass']}{ctx_str}")
    print(f"{'═'*80}\n")
