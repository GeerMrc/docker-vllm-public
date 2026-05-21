#!/usr/bin/env python3
"""
并发压力测试脚本 — 基于指定 vLLM profile
测量: TTFT (prefill), Decode throughput, Total throughput, Output quality

用法:
  python3 scripts/concurrent_bench.py --concurrency 1
  python3 scripts/concurrent_bench.py --concurrency 2
  python3 scripts/concurrent_bench.py --concurrency 4
"""

import argparse
import json
import threading
import time
import sys
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

API_URL = "http://localhost:8089/v1/chat/completions"

# 测试用 prompt 集 — 覆盖不同场景
PROMPTS = [
    # P0: 长输出 (测量 decode throughput)
    {
        "name": "长文生成",
        "messages": [{"role": "user", "content": "详细解释深度学习中 Transformer 架构的核心组件，包括自注意力机制、多头注意力、位置编码、前馈网络和层归一化。对每个组件说明原理和作用。"}],
        "max_tokens": 500,
        "temp": 0.6,
    },
    # P1: 工具调用 (Agent 场景)
    {
        "name": "工具调用",
        "messages": [
            {"role": "system", "content": "你是助手，可调用：get_weather(city), calculate(expr), search(query)。需要调用时用 JSON 格式。"},
            {"role": "user", "content": "上海天气如何？帮我算 23*17+456。"},
        ],
        "max_tokens": 500,
        "temp": 0.0,
    },
    # P2: 精确推理 (质量基准)
    {
        "name": "精确推理",
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
        "max_tokens": 1500,
        "temp": 0.0,
    },
    # P3: 短输出 (测量 TTFT)
    {
        "name": "短回答",
        "messages": [{"role": "user", "content": "Python 的创始人是谁？一句话回答。"}],
        "max_tokens": 500,
        "temp": 0.0,
    },
]


def send_request(prompt_cfg, api_key, request_id):
    """发送单个请求，返回详细指标"""
    start = time.time()

    payload = {
        "model": "Qwen3.6-27B-FP8",
        "messages": prompt_cfg["messages"],
        "max_completion_tokens": prompt_cfg["max_tokens"],
        "temperature": prompt_cfg["temp"],
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    resp = requests.post(API_URL, json=payload, headers=headers, timeout=300)
    elapsed = time.time() - start

    if resp.status_code != 200:
        return {
            "id": request_id,
            "name": prompt_cfg["name"],
            "status": "error",
            "http_code": resp.status_code,
            "error": resp.text[:200],
            "elapsed": elapsed,
        }

    data = resp.json()
    msg = data["choices"][0]["message"]
    usage = data["usage"]
    content = msg.get("content", "") or ""
    reasoning = msg.get("reasoning", "") or ""

    # 质量检查
    quality = {}
    if prompt_cfg["name"] == "精确推理":
        quality["correct"] = "-40" in content
    elif prompt_cfg["name"] == "短回答":
        quality["correct"] = "Guido" in content or "范罗苏姆" in content
    elif prompt_cfg["name"] == "工具调用":
        quality["has_weather"] = "get_weather" in content or "天气" in content
        quality["has_calc"] = "calculate" in content or str(23*17+456) in content

    return {
        "id": request_id,
        "name": prompt_cfg["name"],
        "status": "ok",
        "http_code": 200,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "elapsed": elapsed,
        "content_preview": (content or "(thinking only)")[:150],
        "quality": quality,
        "thinking_tokens_est": len(reasoning) // 4 if reasoning else 0,
        "content_tokens_est": len(content) // 4 if content else 0,
    }


def run_benchmark(concurrency, api_key, rounds=1):
    """运行并发基准测试"""
    results = []

    # 为每个并发 slot 分配 prompt (轮转)
    all_tasks = []
    for r in range(rounds):
        for i in range(concurrency):
            pidx = i % len(PROMPTS)
            all_tasks.append((PROMPTS[pidx], i + r * concurrency))

    total_requests = len(all_tasks)
    print(f"\n{'='*70}")
    print(f"  并发压力测试: concurrency={concurrency}, 请求数={total_requests}")
    print(f"{'='*70}")

    wall_start = time.time()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(send_request, cfg, api_key, rid): (cfg, rid)
            for cfg, rid in all_tasks
        }
        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            cfg, rid = futures[fut]
            status_mark = "✓" if result["status"] == "ok" else "✗"
            tok_s = ""
            if result["status"] == "ok":
                ct = result["completion_tokens"]
                e = result["elapsed"]
                tok_s = f", {ct/e:.1f} tok/s" if e > 0 else ""
            print(f"  [{status_mark}] req#{result['id']:2d} {result['name']:8s} "
                  f"{result.get('prompt_tokens',0):5d}+{result.get('completion_tokens',0):4d} "
                  f"in {result['elapsed']:.2f}s{tok_s}")

    wall_elapsed = time.time() - wall_start

    # 汇总
    ok_results = [r for r in results if r["status"] == "ok"]
    err_results = [r for r in results if r["status"] != "ok"]

    total_prompt = sum(r.get("prompt_tokens", 0) for r in ok_results)
    total_completion = sum(r.get("completion_tokens", 0) for r in ok_results)
    total_tokens = sum(r.get("total_tokens", 0) for r in ok_results)

    # Per-request 平均延迟
    avg_elapsed = sum(r["elapsed"] for r in ok_results) / len(ok_results) if ok_results else 0
    max_elapsed = max(r["elapsed"] for r in ok_results) if ok_results else 0
    min_elapsed = min(r["elapsed"] for r in ok_results) if ok_results else 0

    # 质量汇总
    quality_ok = 0
    quality_total = 0
    for r in ok_results:
        q = r.get("quality", {})
        for k, v in q.items():
            quality_total += 1
            if v:
                quality_ok += 1

    print(f"\n--- 汇总 (并发={concurrency}) ---")
    print(f"  请求: {len(ok_results)} ok / {len(err_results)} err / {total_requests} total")
    print(f"  Wall time: {wall_elapsed:.2f}s")
    print(f"  Token 总量: prompt={total_prompt}, completion={total_completion}, total={total_tokens}")
    print(f"  总吞吐: {total_completion/wall_elapsed:.1f} completion tok/s (wall)")
    print(f"  平均延迟: {avg_elapsed:.2f}s (min={min_elapsed:.2f}, max={max_elapsed:.2f})")
    print(f"  单请求平均吞吐: {total_completion/sum(r['elapsed'] for r in ok_results):.1f} tok/s" if ok_results else "")
    if quality_total > 0:
        print(f"  质量检查: {quality_ok}/{quality_total} PASS ({quality_ok/quality_total*100:.0f}%)")

    return {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "ok": len(ok_results),
        "err": len(err_results),
        "wall_time": wall_elapsed,
        "total_prompt": total_prompt,
        "total_completion": total_completion,
        "wall_throughput": total_completion / wall_elapsed if wall_elapsed > 0 else 0,
        "avg_latency": avg_elapsed,
        "max_latency": max_elapsed,
        "min_latency": min_elapsed,
        "quality_pass_rate": f"{quality_ok}/{quality_total}" if quality_total > 0 else "N/A",
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", "-c", type=int, required=True)
    parser.add_argument("--rounds", "-r", type=int, default=1, help="轮次 (总请求=concurrency*rounds)")
    parser.add_argument("--api-key", type=str, default=None)
    args = parser.parse_args()

    # 读取 API key
    api_key = args.api_key
    if not api_key:
        try:
            with open(".env") as f:
                for line in f:
                    if line.startswith("VLLM_PRIMARY_API_KEY="):
                        api_key = line.strip().split("=", 1)[1]
                        break
        except:
            pass
    if not api_key:
        print("ERROR: 无法获取 API key", file=sys.stderr)
        sys.exit(1)

    result = run_benchmark(args.concurrency, api_key, args.rounds)
