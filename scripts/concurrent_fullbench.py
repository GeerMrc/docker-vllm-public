#!/usr/bin/env python3
"""
完整并发基准+质量测试 — 覆盖 1/2/4 并发
包含: 非Streaming吞吐、Streaming TTFT/Decode分离、4项质量检查+一致性

用法:
  python3 scripts/concurrent_fullbench.py -c 1
  python3 scripts/concurrent_fullbench.py -c 2
  python3 scripts/concurrent_fullbench.py -c 4
"""

import argparse, json, time, sys, requests
from concurrent.futures import ThreadPoolExecutor, as_completed

API_URL = "http://localhost:8089/v1/chat/completions"

# ── 质量测试 prompts ──
QUALITY_PROMPTS = [
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
]

# ── 吞吐测试 prompt ──
THROUGHPUT_PROMPT = {"role": "user", "content": "详细解释深度学习中 Transformer 架构的核心组件，包括自注意力机制、多头注意力、位置编码、前馈网络和层归一化。对每个组件说明原理和作用。"}
STREAMING_PROMPT = {"role": "user", "content": "详细解释 Transformer 架构中自注意力机制的数学原理，包括 Q/K/V 矩阵计算、缩放点积注意力、多头注意力的并行计算方式。"}


def get_api_key():
    try:
        with open(".env") as f:
            for line in f:
                if line.startswith("VLLM_PRIMARY_API_KEY="):
                    return line.strip().split("=", 1)[1]
    except:
        pass
    return None


# ── A: 非Streaming 吞吐 ──
def bench_throughput(concurrency, api_key, n_requests=4):
    payload = {
        "model": "Qwen3.6-27B-FP8",
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
            results.append({"start": s - wall_start, "elapsed": e - s, "completion_tokens": usage.get("completion_tokens", 0), "prompt_tokens": usage.get("prompt_tokens", 0)})

    wall = time.time() - wall_start
    total_completion = sum(r["completion_tokens"] for r in results)
    avg_latency = sum(r["elapsed"] for r in results) / len(results)

    return {
        "wall_s": round(wall, 2),
        "total_completion": total_completion,
        "wall_tok_s": round(total_completion / wall, 1),
        "avg_latency_s": round(avg_latency, 2),
        "per_req_tok_s": round(total_completion / sum(r["elapsed"] for r in results), 1),
    }


# ── B: Streaming TTFT/Decode ──
def bench_streaming(concurrency, api_key):
    payload = {
        "model": "Qwen3.6-27B-FP8",
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
            except:
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


# ── C: 质量测试 ──
def bench_quality(concurrency, api_key):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    def do_quality(pcfg, req_id):
        payload = {
            "model": "Qwen3.6-27B-FP8",
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

    # 为每个并发 slot 分配同一组 prompt
    tasks = []
    for ci in range(concurrency):
        for pidx, pcfg in enumerate(QUALITY_PROMPTS):
            tasks.append((pcfg, f"slot{ci}-{pidx}"))

    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = {pool.submit(do_quality, cfg, rid): (cfg, rid) for cfg, rid in tasks}
        for f in as_completed(futs):
            results.append(f.result())

    # 汇总
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
        per_prompt[name] = {"pass": f"{p}/{len(items)}", "consistent": unique == 1, "sample": contents[0][:80] if contents else ""}

    return {"total_pass": f"{passed}/{total}", "per_prompt": per_prompt, "details": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--concurrency", type=int, required=True)
    args = parser.parse_args()
    c = args.concurrency
    api_key = get_api_key()
    if not api_key:
        print("ERROR: 无法获取 API key"); sys.exit(1)

    print(f"\n{'='*75}")
    print(f"  完整并发基准 (no-thinking, n=2, 132K) — 并发={c}")
    print(f"{'='*75}")

    # A: 非Streaming
    print(f"\n── A: 非Streaming 吞吐 (500 tok × 4 req) ──")
    tp = bench_throughput(c, api_key)
    print(f"  Wall: {tp['wall_s']}s, 总吞吐: {tp['wall_tok_s']} tok/s, "
          f"单请求: {tp['per_req_tok_s']} tok/s, 平均延迟: {tp['avg_latency_s']}s")

    # B: Streaming
    print(f"\n── B: Streaming TTFT/Decode (300 tok × {c} req) ──")
    st = bench_streaming(c, api_key)
    print(f"  Wall: {st['wall_s']}s, TTFT: {st['avg_ttft_s']}s, "
          f"Decode: {st['avg_decode_tok_s']} tok/s/req, Wall: {st['wall_tok_s']} tok/s")

    # C: 质量
    print(f"\n── C: 质量测试 (4项 × {c} 份) ──")
    q = bench_quality(c, api_key)
    print(f"  总通过率: {q['total_pass']}")
    for name, info in q['per_prompt'].items():
        cons = "一致" if info["consistent"] else "不一致"
        print(f"  {name}: {info['pass']} PASS, 一致性: {cons}")
        print(f"    样例: {info['sample'][:60]}...")

    # 汇总一行
    print(f"\n{'─'*75}")
    print(f"  汇总: c={c} | "
          f"吞吐={tp['wall_tok_s']}tok/s | "
          f"TTFT={st['avg_ttft_s']}s | "
          f"Decode={st['avg_decode_tok_s']}tok/s/req | "
          f"质量={q['total_pass']}")
