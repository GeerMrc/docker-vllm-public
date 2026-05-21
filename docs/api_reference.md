# vLLM API 规范性参考文档

基于 vLLM 源码和实际部署环境的完整 API 接口参考。覆盖所有适用于当前 Qwen3.6-27B-FP8 模型部署的端点，包含请求格式、响应格式、认证机制和 curl 示例。

## 版本溯源

| 项目 | 值 |
|------|------|
| vLLM 版本（源码） | `0.21.0` |
| vLLM 版本（运行时） | `0.21.0` |
| 源码仓库 | [vllm-project/vllm](https://github.com/vllm-project/vllm) 官方 (含 local-fixes/ 兼容性补丁) |
| 源码 commit | `ad7125a431` (2026-05-14) |
| Docker 镜像 | `vllm-qwen36:rtx3090-sm86` |
| 模型 | `Qwen3.6-27B-FP8` |
| 模型任务类型 | `generate` |

> 完整构建信息见 [BUILD_INFO](../BUILD_INFO)。`/version` 端点返回运行时版本。

## 部署架构

| 服务 | 容器内端口 | 宿主机端口 | GPU | 默认 Profile |
|------|-----------|-----------|-----|-------------|
| Primary | 8089 | 8089 | 0,1 | agent-fast |
| Secondary | 8089 | 8099 | 2,3 | agent-thinking |

本文档所有示例使用 Primary 服务端口 `8089`。Secondary 服务将端口替换为 `8099` 即可。

## 认证机制

vLLM 通过 `AuthenticationMiddleware` 实现 API 认证。行为如下：

- **当 `VLLM_API_KEY` 已配置时**：所有 `/v1/*` 路径的请求需要 `Authorization: Bearer <token>` 头
- **无 `/v1` 前缀的路径**（如 `/health`、`/ping`、`/version`、`/tokenize`、`/detokenize`、`/metrics`、`/load`）**始终无需认证**
- **未配置 `VLLM_API_KEY` 时**：所有端点均无需认证

API Key 由 `.env` 中 `VLLM_PRIMARY_API_KEY` 和 `VLLM_SECONDARY_API_KEY` 控制（空值=不鉴权）。

```bash
# 认证请求示例
curl -H "Authorization: Bearer YOUR_API_KEY" http://localhost:8089/v1/models

# 无认证请求示例（非 /v1 路径）
curl http://localhost:8089/health
```

## 通用约定

- **Content-Type**: 所有 POST 请求需 `Content-Type: application/json`（音频端点除外，使用 `multipart/form-data`）
- **模型名称**: 请求体中的 `model` 字段为可选，未指定时使用默认模型 `Qwen3.6-27B-FP8`
- **SSE 流式**: 启用 `stream: true` 时，响应以 Server-Sent Events (SSE) 格式返回，每个事件格式为 `data: {JSON}\n\n`，流结束标志为 `data: [DONE]\n\n`

---

## 1. 健康检查与监控

> 所有端点均无需认证。

### GET /health

服务健康检查。Docker Compose healthcheck 使用此端点。

- **200**: 服务正常
- **503**: 服务不可用（引擎尚未就绪或发生故障）
- **响应体**: 空（无 JSON body）

```bash
curl -sf http://localhost:8089/health -o /dev/null -w "%{http_code}"
# 输出: 200
```

### GET /ping

SageMaker 兼容 ping 端点。支持 GET 和 POST 两种方法。

- **200**: 服务存活
- **响应体**: 空

```bash
curl -sf http://localhost:8089/ping -o /dev/null -w "%{http_code}"
# 输出: 200

# POST 方法同样可用
curl -sf -X POST http://localhost:8089/ping -o /dev/null -w "%{http_code}"
# 输出: 200
```

### GET /version

返回 vLLM 服务版本号。

**响应格式**:

```json
{
  "version": "0.21.0"
}
```

```bash
curl -s http://localhost:8089/version | python3 -m json.tool
```

### GET /load

返回当前 GPU 负载指标。

**响应格式**: `application/json`

```bash
curl -s http://localhost:8089/load
```

### GET /metrics

Prometheus 格式的指标端点。包含 GPU 利用率、请求延迟、吞吐量、KV cache 使用率等指标。

**响应格式**: `text/plain; version=0.0.4; charset=utf-8` (Prometheus exposition format)

```bash
curl -s http://localhost:8089/metrics | head -20
```

## 2. 模型列表

### GET /v1/models

列出当前服务加载的模型。**需要认证**（当 API Key 已配置时）。

**请求**:

无需请求体，通过 HTTP GET 方法调用。

**响应格式** (`ModelList`):

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen3.6-27B-FP8",
      "object": "model",
      "created": 1715635200,
      "owned_by": "vllm",
      "root": "/models/Qwen3.6-27B-FP8",
      "parent": null,
      "max_model_len": 143360,
      "permission": [
        {
          "id": "modelperm-xxxxxxxx",
          "object": "model_permission",
          "created": 1715635200,
          "allow_create_engine": false,
          "allow_sampling": true,
          "allow_logprobs": true,
          "allow_search_indices": false,
          "allow_view": true,
          "allow_fine_tuning": false,
          "organization": "*",
          "group": null,
          "is_blocking": false
        }
      ]
    }
  ]
}
```

**响应字段说明**:

| 字段 | 说明 |
|------|------|
| `data[].id` | 模型名称 |
| `data[].max_model_len` | 最大上下文长度（vLLM 扩展），取决于 profile 配置 |
| `data[].owned_by` | 固定为 `"vllm"` |
| `data[].permission` | 模型权限列表 |

```bash
# 带认证
curl -s -H "Authorization: Bearer $VLLM_PRIMARY_API_KEY" \
  http://localhost:8089/v1/models | python3 -m json.tool

# 简洁版（仅显示模型 ID）
curl -s -H "Authorization: Bearer $VLLM_PRIMARY_API_KEY" \
  http://localhost:8089/v1/models | python3 -c "import sys,json; [print(m['id']) for m in json.load(sys.stdin)['data']]"
```

## 3. Chat Completions

### POST /v1/chat/completions

OpenAI 兼容的对话补全接口。支持流式输出、思考模式、Tool Calling 和结构化输出。**需要认证**。

#### 请求参数

**OpenAI 标准参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `messages` | `array` | **必填** | 消息列表，每条消息包含 `role` 和 `content` |
| `model` | `string` | 可选 | 模型名称，默认使用服务加载的模型 |
| `temperature` | `float` | 服务端默认 | 采样温度，范围 [0, ∞)，越高越随机 |
| `top_p` | `float` | 服务端默认 | 核采样概率阈值，范围 (0, 1] |
| `max_completion_tokens` | `int` | 可选 | 最大生成 token 数（推荐使用，替代已废弃的 `max_tokens`） |
| `max_tokens` | `int` | 可选 | 已废弃，请使用 `max_completion_tokens` |
| `n` | `int` | `1` | 生成候选数量 |
| `stream` | `bool` | `false` | 是否启用流式输出 |
| `stream_options` | `object` | 可选 | `{include_usage: bool, continuous_usage_stats: bool}` |
| `stop` | `string\|array` | `[]` | 停止生成的字符串（最多 4 个） |
| `presence_penalty` | `float` | `0.0` | 存在惩罚，范围 [-2, 2] |
| `frequency_penalty` | `float` | `0.0` | 频率惩罚，范围 [-2, 2] |
| `seed` | `int` | 可选 | 随机种子，相同种子 + 参数可复现输出 |
| `logprobs` | `bool` | `false` | 是否返回 log probabilities |
| `top_logprobs` | `int` | `0` | 返回前 N 个 token 的 logprob，范围 [0, 5] |
| `response_format` | `object` | 可选 | 结构化输出格式（见下方说明） |
| `tools` | `array` | 可选 | 工具定义列表（见 Tool Calling 章节） |
| `tool_choice` | `string\|object` | `"auto"` | 工具选择策略：`"none"` / `"auto"` / `"required"` / 具体工具名 |
| `parallel_tool_calls` | `bool` | `true` | 是否允许并行工具调用 |
| `reasoning_effort` | `string` | 可选 | 推理强度：`"none"` / `"low"` / `"medium"` / `"high"` |
| `user` | `string` | 可选 | 用户标识（vLLM 忽略此字段） |

**vLLM 扩展参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `top_k` | `int` | 可选 | Top-K 采样 |
| `min_p` | `float` | 可选 | 最小概率阈值 |
| `repetition_penalty` | `float` | 可选 | 重复惩罚（1.0 = 不惩罚） |
| `min_tokens` | `int` | `0` | 最小生成 token 数 |
| `stop_token_ids` | `array` | `[]` | 按 token ID 停止 |
| `include_stop_str_in_output` | `bool` | `false` | 在输出中包含停止字符串 |
| `ignore_eos` | `bool` | `false` | 忽略 EOS token |
| `skip_special_tokens` | `bool` | `true` | 跳过特殊 token |
| `echo` | `bool` | `false` | 在输出中回显提示词 |
| `add_generation_prompt` | `bool` | `true` | 添加生成提示 |
| `continue_final_message` | `bool` | `false` | 继续最后一条消息（与 `add_generation_prompt` 互斥） |
| `chat_template` | `string` | 可选 | 覆盖 Jinja 聊天模板 |
| `chat_template_kwargs` | `object` | 可选 | 聊天模板额外参数 |
| `structured_outputs` | `object` | 可选 | 结构化输出参数（见下方说明） |
| `priority` | `int` | `0` | 请求优先级（越小越高） |
| `truncate_prompt_tokens` | `int` | 可选 | 截断提示词到指定 token 数（-1 表示自动） |
| `allowed_token_ids` | `array` | 可选 | 限制允许生成的 token ID 列表 |
| `bad_words` | `array` | `[]` | 禁止出现的词列表 |
| `repetition_detection` | `object` | 可选 | 重复检测参数，提前终止重复输出 |

**`response_format` 类型**:

```json
// JSON 模式
{"type": "json_object"}

// JSON Schema 模式
{
  "type": "json_schema",
  "json_schema": {
    "name": "my_schema",
    "strict": true,
    "schema": {
      "type": "object",
      "properties": {"key": {"type": "string"}},
      "required": ["key"]
    }
  }
}

// 纯文本模式
{"type": "text"}
```

**`structured_outputs` 参数**（vLLM 扩展，与 `response_format` 互补）:

```json
{
  "json": "JSON Schema 字符串或对象",
  "regex": "正则表达式字符串",
  "choice": ["选项1", "选项2"],
  "grammar": "BNF 语法字符串",
  "json_object": true
}
```

> 以上字段互斥，只能设置其中一个。

#### 消息格式

`messages` 数组中的每条消息包含 `role` 和 `content`：

```json
[
  {"role": "system", "content": "你是一个有用的助手。"},
  {"role": "user", "content": "解释量子计算的基本概念。"},
  {"role": "assistant", "content": "量子计算利用量子力学原理..."},
  {"role": "user", "content": "能更详细说明吗？"}
]
```

支持的 `role` 值：`system`、`user`、`assistant`、`tool`。

Tool 响应消息格式：

```json
{"role": "tool", "content": "工具返回的结果", "tool_call_id": "chatcmpl-tool-xxxxxx"}
```

#### 响应格式

**非流式响应** (`ChatCompletionResponse`):

```json
{
  "id": "chatcmpl-xxxxxxxx",
  "object": "chat.completion",
  "created": 1715635200,
  "model": "Qwen3.6-27B-FP8",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "生成的文本内容",
        "refusal": null,
        "annotations": null,
        "audio": null,
        "function_call": null,
        "tool_calls": [],
        "reasoning": "思考过程内容（仅思考模式）"
      },
      "logprobs": null,
      "finish_reason": "stop",
      "stop_reason": null,
      "token_ids": null
    }
  ],
  "service_tier": null,
  "system_fingerprint": "vllm-{版本号}-tp{tensor_parallel}-{hash}",
  "usage": {
    "prompt_tokens": 15,
    "total_tokens": 45,
    "completion_tokens": 30,
    "prompt_tokens_details": null
  },
  "prompt_logprobs": null,
  "prompt_token_ids": null,
  "kv_transfer_params": null
}
```

> `prompt_tokens_details` 在 KV cache 命中时为 `{"cached_tokens": N}`，未命中时为 `null`。`system_fingerprint` 包含 vLLM 版本和并行配置信息。

**`finish_reason` 取值**:

| 值 | 说明 |
|------|------|
| `"stop"` | 正常结束或遇到 stop 序列 |
| `"length"` | 达到 `max_completion_tokens` 限制 |
| `"tool_calls"` | 模型决定调用工具 |
| `"content_filter"` | 内容过滤触发 |

**`usage` 字段**:

| 字段 | 说明 |
|------|------|
| `prompt_tokens` | 输入 token 数 |
| `completion_tokens` | 输出 token 数 |
| `total_tokens` | 总 token 数 |
| `prompt_tokens_details.cached_tokens` | KV cache 命中的 token 数（vLLM 扩展） |

#### curl 示例

**基础请求**:

```bash
curl -s http://localhost:8089/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VLLM_PRIMARY_API_KEY" \
  -d '{
    "model": "Qwen3.6-27B-FP8",
    "messages": [
      {"role": "user", "content": "用一句话解释什么是机器学习。"}
    ],
    "max_completion_tokens": 100,
    "temperature": 0.7
  }' | python3 -m json.tool
```

**流式请求**:

```bash
curl -N http://localhost:8089/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VLLM_PRIMARY_API_KEY" \
  -d '{
    "model": "Qwen3.6-27B-FP8",
    "messages": [
      {"role": "user", "content": "从1数到5"}
    ],
    "max_completion_tokens": 100,
    "stream": true
  }'
```

流式响应格式（SSE）：

**非思考模式**（`enable_thinking: false`，如 `agent-fast`/`instruct` profile）：

```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1715635200,"model":"Qwen3.6-27B-FP8","choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}],"prompt_token_ids":null}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1715635200,"model":"Qwen3.6-27B-FP8","choices":[{"index":0,"delta":{"content":"生成"},"logprobs":null,"finish_reason":null,"token_ids":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1715635200,"model":"Qwen3.6-27B-FP8","choices":[{"index":0,"delta":{"content":"的文本"},"logprobs":null,"finish_reason":null,"token_ids":null}]}

...

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1715635200,"model":"Qwen3.6-27B-FP8","choices":[{"index":0,"delta":{},"logprobs":null,"finish_reason":"stop","token_ids":null}]}

data: [DONE]
```

**思考模式**（`enable_thinking: true`，如 `agent-thinking`/`code`/`think` profile）：

```
data: {"id":"chatcmpl-xxx",...,"choices":[{"index":0,"delta":{"role":"assistant","content":""},...}]}

data: {"id":"chatcmpl-xxx",...,"choices":[{"index":0,"delta":{"reasoning":"思考"},"finish_reason":null,"token_ids":null}]}

data: {"id":"chatcmpl-xxx",...,"choices":[{"index":0,"delta":{"reasoning":"过程"},"finish_reason":null,"token_ids":null}]}

...（reasoning 输出完毕后切换到 content）...

data: {"id":"chatcmpl-xxx",...,"choices":[{"index":0,"delta":{"content":"最终"},"finish_reason":null,"token_ids":null}]}

data: {"id":"chatcmpl-xxx",...,"choices":[{"index":0,"delta":{},"finish_reason":"stop","token_ids":null}]}

data: [DONE]
```

> 思考模式下，`delta.reasoning` 先逐步返回思考内容，思考完毕后 `delta.content` 返回正文。

**流式带 usage 统计**（需设置 `stream_options`）:

```bash
curl -N http://localhost:8089/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VLLM_PRIMARY_API_KEY" \
  -d '{
    "model": "Qwen3.6-27B-FP8",
    "messages": [{"role": "user", "content": "你好"}],
    "max_completion_tokens": 50,
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

> 启用 `include_usage` 后，最后一个 chunk 会包含 `usage` 字段。

#### 思考模式

当 profile 设置 `enable_thinking: true` 时（如 `agent-thinking`），模型会在生成回答前进行推理思考。思考内容通过 `reasoning` 字段返回。

**非流式思考模式响应**:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "机器学习是一种人工智能的分支...",
      "reasoning": "用户问了一个关于机器学习的基础问题，我需要简洁明了地解释..."
    }
  }]
}
```

**流式思考模式响应**: 思考内容通过 `delta.reasoning` 字段逐步返回，正文中断内容通过 `delta.content` 返回。

> 当 profile 设置 `preserve_thinking: true` 时，`reasoning` 内容会被保留在最终输出中。设置为 `false` 时，思考内容仅在流式传输中可见，非流式响应中 `reasoning` 为空。

#### Tool Calling

启用 `enable_auto_tool_choice` 和 `tool_call_parser` 的 profile（当前所有 profile 均已启用 `qwen3_coder` 解析器）支持 Tool Calling。

**Tool 定义格式**:

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "获取指定城市的天气信息",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string", "description": "城市名称"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
          },
          "required": ["city"]
        }
      }
    }
  ]
}
```

**Tool Calling 请求示例**:

```bash
curl -s http://localhost:8089/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VLLM_PRIMARY_API_KEY" \
  -d '{
    "model": "Qwen3.6-27B-FP8",
    "messages": [
      {"role": "user", "content": "北京今天天气怎么样？"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "获取指定城市的天气信息",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {"type": "string", "description": "城市名称"}
            },
            "required": ["city"]
          }
        }
      }
    ],
    "tool_choice": "auto"
  }' | python3 -m json.tool
```

**Tool Calling 响应**:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "chatcmpl-tool-xxxxxxxx",
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": "{\"city\": \"北京\"}"
          }
        }
      ]
    },
    "finish_reason": "tool_calls"
  }]
}
```

**多轮 Tool Calling 流程**:

```bash
# 1. 模型返回 tool_calls → 2. 执行工具 → 3. 将结果作为 tool 消息发回
curl -s http://localhost:8089/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VLLM_PRIMARY_API_KEY" \
  -d '{
    "model": "Qwen3.6-27B-FP8",
    "messages": [
      {"role": "user", "content": "北京天气怎么样？"},
      {"role": "assistant", "content": null, "tool_calls": [{"id": "chatcmpl-tool-xxx", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"北京\"}"}}]},
      {"role": "tool", "tool_call_id": "chatcmpl-tool-xxx", "content": "{\"temperature\": 25, \"condition\": \"晴\"}"}
    ]
  }' | python3 -m json.tool
```

#### 结构化输出

**JSON 模式**（确保输出合法 JSON）:

```bash
curl -s http://localhost:8089/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VLLM_PRIMARY_API_KEY" \
  -d '{
    "model": "Qwen3.6-27B-FP8",
    "messages": [{"role": "user", "content": "列出三种编程语言"}],
    "response_format": {"type": "json_object"}
  }'
```

**JSON Schema 模式**（按 Schema 生成）:

```bash
curl -s http://localhost:8089/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VLLM_PRIMARY_API_KEY" \
  -d '{
    "model": "Qwen3.6-27B-FP8",
    "messages": [{"role": "user", "content": "提取：张三，25岁，软件工程师"}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "person_info",
        "strict": true,
        "schema": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "occupation": {"type": "string"}
          },
          "required": ["name", "age", "occupation"],
          "additionalProperties": false
        }
      }
    }
  }'
```

### POST /v1/chat/completions/batch

批量对话补全。在单次请求中发送多组对话，返回每组对话的补全结果。

**与单次请求的区别**:

- `messages` 字段为 `array[array]`（二维数组），每个元素是一组完整的对话
- 不支持流式（`stream` 必须为 `false`）
- 不支持 Tool Calling
- 不支持 Beam Search
- `n` 必须为 1

**响应**: 标准 `ChatCompletionResponse`，`choices` 按索引对应每组对话（0 到 N-1）。

```bash
curl -s http://localhost:8089/v1/chat/completions/batch \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VLLM_PRIMARY_API_KEY" \
  -d '{
    "model": "Qwen3.6-27B-FP8",
    "messages": [
      [{"role": "user", "content": "1+1=?"}],
      [{"role": "user", "content": "2+2=?"}]
    ],
    "max_completion_tokens": 20
  }' | python3 -m json.tool
```

## 4. Text Completions

### POST /v1/completions

OpenAI 兼容的文本补全接口。给定一段提示文本，模型继续生成后续内容。**需要认证**。

#### 请求参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prompt` | `string\|array` | 可选 | 提示文本（字符串或字符串数组） |
| `model` | `string` | 可选 | 模型名称 |
| `max_tokens` | `int` | `16` | 最大生成 token 数 |
| `temperature` | `float` | 服务端默认 | 采样温度 |
| `top_p` | `float` | 服务端默认 | 核采样阈值 |
| `n` | `int` | `1` | 候选数量 |
| `stream` | `bool` | `false` | 是否流式 |
| `stream_options` | `object` | 可选 | 同 Chat Completions |
| `stop` | `string\|array` | `[]` | 停止序列 |
| `echo` | `bool` | `false` | 在输出中回显提示文本 |
| `suffix` | `string` | 可选 | 补全文本后的后缀 |
| `presence_penalty` | `float` | `0.0` | 存在惩罚 |
| `frequency_penalty` | `float` | `0.0` | 频率惩罚 |
| `seed` | `int` | 可选 | 随机种子 |
| `logprobs` | `int` | 可选 | 返回前 N 个 token 的 logprob |
| `logit_bias` | `object` | 可选 | Token ID 到偏置值的映射 |
| `user` | `string` | 可选 | 用户标识 |
| `response_format` | `object` | 可选 | 同 Chat Completions |

**vLLM 扩展参数**: `top_k`, `min_p`, `repetition_penalty`, `length_penalty`, `min_tokens`, `stop_token_ids`, `include_stop_str_in_output`, `ignore_eos`, `skip_special_tokens`, `truncate_prompt_tokens`, `allowed_token_ids`, `structured_outputs`, `priority`, `bad_words`, `repetition_detection` 等。用法同 Chat Completions。

#### 响应格式

```json
{
  "id": "cmpl-xxxxxxxx",
  "object": "text_completion",
  "created": 1715635200,
  "model": "Qwen3.6-27B-FP8",
  "choices": [
    {
      "index": 0,
      "text": "生成的文本内容",
      "logprobs": null,
      "finish_reason": "stop",
      "stop_reason": null,
      "token_ids": null,
      "prompt_logprobs": null,
      "prompt_token_ids": null
    }
  ],
  "service_tier": null,
  "system_fingerprint": "vllm-{版本号}-tp{tensor_parallel}-{hash}",
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 25,
    "completion_tokens": 20,
    "prompt_tokens_details": null
  },
  "kv_transfer_params": null
}
```

#### curl 示例

**基础补全**:

```bash
curl -s http://localhost:8089/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VLLM_PRIMARY_API_KEY" \
  -d '{
    "model": "Qwen3.6-27B-FP8",
    "prompt": "人工智能的未来发展方向包括"
  }' | python3 -m json.tool
```

**流式补全**:

```bash
curl -N http://localhost:8089/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VLLM_PRIMARY_API_KEY" \
  -d '{
    "model": "Qwen3.6-27B-FP8",
    "prompt": "从前有座山",
    "max_tokens": 100,
    "stream": true
  }'
```

**带回显**:

```bash
curl -s http://localhost:8089/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VLLM_PRIMARY_API_KEY" \
  -d '{
    "model": "Qwen3.6-27B-FP8",
    "prompt": "1, 2, 3,",
    "max_tokens": 20,
    "echo": true
  }' | python3 -m json.tool
```

## 5. Responses API

### POST /v1/responses

OpenAI Responses API（较新的接口风格）。将对话输入、工具调用和推理整合为统一的响应对象。**需要认证**。

#### 请求参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `input` | `string\|array` | **必填** | 输入内容：字符串或消息列表 |
| `model` | `string` | 可选 | 模型名称 |
| `instructions` | `string` | 可选 | 系统指令（类似 system message） |
| `temperature` | `float` | 可选 | 采样温度 |
| `top_p` | `float` | 可选 | 核采样阈值 |
| `max_output_tokens` | `int` | 可选 | 最大输出 token 数 |
| `stream` | `bool` | `false` | 是否流式 |
| `tools` | `array` | `[]` | 工具定义列表 |
| `tool_choice` | `string` | `"auto"` | 工具选择策略 |
| `reasoning` | `object` | 可选 | 推理配置，如 `{"effort": "high"}` |
| `text` | `object` | 可选 | 文本输出配置（含结构化输出） |
| `metadata` | `object` | 可选 | 请求元数据 |
| `store` | `bool` | `true` | 是否存储响应 |
| `background` | `bool` | `false` | 是否后台执行 |
| `truncation` | `string` | `"disabled"` | 截断策略：`"auto"` / `"disabled"` |
| `presence_penalty` | `float` | 可选 | 存在惩罚 |
| `frequency_penalty` | `float` | 可选 | 频率惩罚 |

**vLLM 扩展参数**: `top_k`, `repetition_penalty`, `seed`, `stop`, `ignore_eos`, `structured_outputs`, `priority`, `cache_salt`, `vllm_xargs` 等。

#### 响应格式

```json
{
  "id": "resp_xxxxxxxx",
  "object": "response",
  "created_at": 1715635200,
  "model": "Qwen3.6-27B-FP8",
  "status": "completed",
  "output": [
    {
      "type": "reasoning",
      "content": [
        {"type": "reasoning_text", "text": "思考过程内容..."}
      ]
    },
    {
      "type": "message",
      "role": "assistant",
      "content": [
        {"type": "output_text", "text": "生成的文本内容"}
      ]
    }
  ],
  "usage": {
    "input_tokens": 10,
    "output_tokens": 20,
    "total_tokens": 30,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens_details": {"reasoning_tokens": 0}
  },
  "temperature": 0.6,
  "top_p": 0.95,
  "tool_choice": "none",
  "tools": [],
  "parallel_tool_calls": true,
  "truncation": "disabled",
  "background": false,
  "service_tier": "auto"
}
```

> 思考模式启用时，`output` 数组的第一项为 `type: "reasoning"` 的思考内容，第二项为 `type: "message"` 的正式回复。未启用思考模式时仅返回 message 项。`temperature`、`top_p` 等采样参数的值来自服务端 profile 配置（请求中未指定时使用 profile 默认值）。

#### curl 示例

```bash
curl -s http://localhost:8089/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VLLM_PRIMARY_API_KEY" \
  -d '{
    "model": "Qwen3.6-27B-FP8",
    "input": "用三句话解释深度学习"
  }' | python3 -m json.tool
```

### GET /v1/responses/{response_id}

获取指定响应的详情。**需要认证**。

```bash
curl -s -H "Authorization: Bearer $VLLM_PRIMARY_API_KEY" \
  http://localhost:8089/v1/responses/resp_xxxxxxxx | python3 -m json.tool
```

### POST /v1/responses/{response_id}/cancel

取消正在进行的响应。**需要认证**。

```bash
curl -s -X POST -H "Authorization: Bearer $VLLM_PRIMARY_API_KEY" \
  http://localhost:8089/v1/responses/resp_xxxxxxxx/cancel | python3 -m json.tool
```

## 6. Anthropic Messages API

### POST /v1/messages

Anthropic 兼容的 Messages API。允许使用 Anthropic SDK 格式直接请求 vLLM 服务。**需要认证**。

#### 请求参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | `string` | **必填** | 模型名称 |
| `messages` | `array` | **必填** | 消息列表 |
| `max_tokens` | `int` | **必填** | 最大生成 token 数（必须 > 0） |
| `system` | `string\|array` | 可选 | 系统提示词 |
| `temperature` | `float` | 可选 | 采样温度 |
| `top_p` | `float` | 可选 | 核采样阈值 |
| `top_k` | `int` | 可选 | Top-K 采样 |
| `stop_sequences` | `array` | 可选 | 停止序列列表 |
| `stream` | `bool` | `false` | 是否流式 |
| `tools` | `array` | 可选 | 工具定义 |
| `tool_choice` | `object` | 可选 | 工具选择策略 |
| `metadata` | `object` | 可选 | 请求元数据 |

#### 响应格式

```json
{
  "id": "chatcmpl-xxxxxxxx",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "thinking",
      "thinking": "思考过程内容...",
      "signature": "hex_signature"
    },
    {"type": "text", "text": "生成的文本内容"}
  ],
  "model": "Qwen3.6-27B-FP8",
  "stop_reason": "end_turn",
  "usage": {
    "input_tokens": 10,
    "output_tokens": 20
  }
}
```

**`stop_reason` 取值**: `"end_turn"` / `"max_tokens"` / `"stop_sequence"` / `"tool_use"`

#### curl 示例

**基础请求**:

```bash
curl -s http://localhost:8089/v1/messages \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VLLM_PRIMARY_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "Qwen3.6-27B-FP8",
    "max_tokens": 100,
    "messages": [
      {"role": "user", "content": "解释什么是 Transformer 架构"}
    ]
  }' | python3 -m json.tool
```

**带系统提示词**:

```bash
curl -s http://localhost:8089/v1/messages \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VLLM_PRIMARY_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "Qwen3.6-27B-FP8",
    "max_tokens": 100,
    "system": "你是一个简洁的技术助手，回答不超过三句话。",
    "messages": [
      {"role": "user", "content": "什么是 Docker？"}
    ]
  }' | python3 -m json.tool
```

### POST /v1/messages/count_tokens

计算消息的 token 数量。**需要认证**。

**请求参数**: `model`, `messages`, `system`, `tools`, `tool_choice`

**响应格式**:

```json
{
  "input_tokens": 42,
  "context_management": {
    "original_input_tokens": 42
  }
}
```

```bash
curl -s http://localhost:8089/v1/messages/count_tokens \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VLLM_PRIMARY_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "Qwen3.6-27B-FP8",
    "messages": [
      {"role": "user", "content": "你好"}
    ]
  }' | python3 -m json.tool
```

## 7. Tokenization

> 以下端点路径不以 `/v1` 开头，**无需认证**。

### POST /tokenize

将文本或对话消息转换为 token ID 列表。支持纯文本和对话两种输入格式。

**纯文本分词请求** (`TokenizeCompletionRequest`):

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prompt` | `string` | **必填** | 要分词的文本 |
| `model` | `string` | 可选 | 模型名称 |
| `add_special_tokens` | `bool` | `true` | 是否添加特殊 token |
| `return_token_strs` | `bool` | `false` | 是否返回 token 字符串表示 |

**对话分词请求** (`TokenizeChatRequest`):

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `messages` | `array` | **必填** | 消息列表（同 Chat Completions 格式） |
| `model` | `string` | 可选 | 模型名称 |
| `add_generation_prompt` | `bool` | `true` | 是否添加生成提示 |
| `return_token_strs` | `bool` | `false` | 是否返回 token 字符串 |
| `tools` | `array` | 可选 | 工具定义（会影响分词结果） |

**响应格式**:

```json
{
  "count": 5,
  "max_model_len": 143360,
  "tokens": [1234, 5678, 9012, 3456, 7890],
  "token_strs": ["你", "好", "世", "界", "!"]
}
```

```bash
# 纯文本分词
curl -s http://localhost:8089/tokenize \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.6-27B-FP8",
    "prompt": "你好，世界！"
  }' | python3 -m json.tool

# 带 token 字符串
curl -s http://localhost:8089/tokenize \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.6-27B-FP8",
    "prompt": "Hello, world!",
    "return_token_strs": true
  }' | python3 -m json.tool

# 对话分词
curl -s http://localhost:8089/tokenize \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.6-27B-FP8",
    "messages": [
      {"role": "system", "content": "你是助手"},
      {"role": "user", "content": "你好"}
    ]
  }' | python3 -m json.tool
```

### POST /detokenize

将 token ID 列表转换回文本。**无需认证**。

**请求参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `tokens` | `array[int]` | **必填** | token ID 列表（每个值 >= 0） |
| `model` | `string` | 可选 | 模型名称 |

**响应格式**:

```json
{
  "prompt": "你好，世界！"
}
```

```bash
# 先分词再反分词验证
curl -s http://localhost:8089/detokenize \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.6-27B-FP8",
    "tokens": [1234, 5678, 9012]
  }' | python3 -m json.tool
```

## 8. 辅助端点

### POST /v1/chat/completions/render

渲染对话模板，返回格式化后的提示词（不执行推理）。用于调试模板和查看实际发送给模型的输入。**需要认证**。

```bash
curl -s http://localhost:8089/v1/chat/completions/render \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VLLM_PRIMARY_API_KEY" \
  -d '{
    "model": "Qwen3.6-27B-FP8",
    "messages": [
      {"role": "system", "content": "你是助手"},
      {"role": "user", "content": "你好"}
    ]
  }' | python3 -m json.tool
```

### POST /v1/completions/render

渲染文本补全模板。**需要认证**。

```bash
curl -s http://localhost:8089/v1/completions/render \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VLLM_PRIMARY_API_KEY" \
  -d '{
    "model": "Qwen3.6-27B-FP8",
    "prompt": "从前有座山"
  }' | python3 -m json.tool
```

### POST /generative_scoring

生成式评分接口。给定查询和候选项，通过模型评估每个候选项的得分。**无需认证**（路径不含 `/v1`）。

**请求参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `query` | `string\|array[int]` | **必填** | 查询文本或 token ID |
| `items` | `array` | **必填** | 候选项列表（字符串或 token ID 数组） |
| `label_token_ids` | `array[int]` | **必填** | 标签 token ID 列表 |
| `model` | `string` | 可选 | 模型名称 |
| `apply_softmax` | `bool` | `true` | 是否应用 softmax |
| `item_first` | `bool` | `false` | 是否候选项在前 |

```bash
curl -s http://localhost:8089/generative_scoring \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.6-27B-FP8",
    "query": "法国的首都是",
    "items": ["巴黎", "伦敦", "东京"],
    "label_token_ids": [1]
  }' | python3 -m json.tool
```

### POST /inference/v1/generate

脱聚服务（Disaggregated Serving）内部生成接口。直接传入 token ID 进行生成，跳过对话模板处理。**无需认证**。

主要用于 Prefill/Decode 分离架构中的内部服务间通信。普通用户场景不需要直接使用此端点。

```bash
curl -s http://localhost:8089/inference/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "test-001",
    "token_ids": [1, 1234, 5678],
    "sampling_params": {
      "temperature": 0.7,
      "max_tokens": 50
    }
  }'
```

## 9. 错误响应

### 统一错误格式

所有 API 错误均返回以下 JSON 格式：

```json
{
  "error": {
    "message": "错误描述信息",
    "type": "错误类型",
    "param": null,
    "code": 400
  }
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `error.message` | `string` | 人类可读的错误描述 |
| `error.type` | `string` | 错误类型标识（见下表） |
| `error.param` | `string\|null` | 导致错误的参数名（仅验证错误时有值） |
| `error.code` | `int` | HTTP 状态码 |

### 错误类型映射

| HTTP 状态码 | `type` 值 | 典型场景 |
|------------|----------|---------|
| 400 | `"Bad Request"` | 参数格式错误、值越界、无效请求 |
| 401 | `"Unauthorized"` (HTTPException) | 缺少认证头或 API Key 不正确 |
| 404 | `"NotFoundError"` | 模型不存在、端点不存在 |
| 500 | `"InternalServerError"` | 引擎内部错误、生成失败 |
| 501 | `"NotImplementedError"` | 使用了未实现的功能 |
| 503 | (无 body) | 服务未就绪（`/health` 端点） |

### 常见错误示例

**认证失败 (401)**:

```bash
# 无认证头
curl -s http://localhost:8089/v1/models
# {"error":"Unauthorized"}

# 错误的 API Key
curl -s -H "Authorization: Bearer wrong-key" http://localhost:8089/v1/models
# {"error":"Unauthorized"}
```

**参数错误 (400)**:

```bash
# 缺少必填参数
curl -s http://localhost:8089/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VLLM_PRIMARY_API_KEY" \
  -d '{"model": "Qwen3.6-27B-FP8"}'
# {"error":{"message":"...messages is required...","type":"BadRequestError","param":null,"code":400}}
```

**模型不存在 (404)**（假设请求不存在的模型）:

```json
{"error":{"message":"The model `nonexistent-model` does not exist.","type":"NotFoundError","param":null,"code":404}}
```

---

## 附录 A：不适用端点

当前部署的 Qwen3.6-27B-FP8 模型任务类型为 `generate`，以下端点因模型能力限制不可用。如需使用，需部署对应任务类型的模型。

| 端点类别 | 端点路径 | 所需任务类型 | 说明 |
|---------|---------|------------|------|
| 音频转写 | `POST /v1/audio/transcriptions` | `transcription` | 语音转文字 |
| 音频翻译 | `POST /v1/audio/translations` | `transcription` | 语音翻译 |
| 嵌入向量 | `POST /v1/embeddings` | `embed` | 文本嵌入 |
| Cohere 嵌入 | `POST /v2/embed` | `embed` | Cohere 格式嵌入 |
| 分类 | `POST /classify` | `classify` | 文本分类 |
| 评分 | `POST /score`, `POST /v1/score` | `score` | 相关性评分 |
| 重排 | `POST /rerank`, `POST /v1/rerank`, `POST /v2/rerank` | `score` | 搜索结果重排 |
| 通用池化 | `POST /pooling` | `pooling` | 通用池化输出 |
| 实时音频 | `WebSocket /v1/realtime` | `realtime` | 实时音频转写 |

---

## 附录 B：端点速查表

| 方法 | 路径 | 认证 | 用途 |
|------|------|------|------|
| GET | `/health` | 否 | 健康检查 |
| GET | `/ping` | 否 | SageMaker ping |
| POST | `/ping` | 否 | SageMaker ping |
| GET | `/version` | 否 | 版本信息 |
| GET | `/load` | 否 | GPU 负载 |
| GET | `/metrics` | 否 | Prometheus 指标 |
| POST | `/tokenize` | 否 | 文本/对话分词 |
| POST | `/detokenize` | 否 | Token ID 转文本 |
| POST | `/generative_scoring` | 否 | 生成式评分 |
| POST | `/inference/v1/generate` | 否 | 脱聚服务生成 |
| GET | `/v1/models` | 是 | 模型列表 |
| POST | `/v1/chat/completions` | 是 | 对话补全（核心） |
| POST | `/v1/chat/completions/batch` | 是 | 批量对话补全 |
| POST | `/v1/chat/completions/render` | 是 | 渲染对话模板 |
| POST | `/v1/completions` | 是 | 文本补全 |
| POST | `/v1/completions/render` | 是 | 渲染补全模板 |
| POST | `/v1/responses` | 是 | Responses API |
| GET | `/v1/responses/{id}` | 是 | 获取响应 |
| POST | `/v1/responses/{id}/cancel` | 是 | 取消响应 |
| POST | `/v1/messages` | 是 | Anthropic Messages |
| POST | `/v1/messages/count_tokens` | 是 | Anthropic Token 计数 |
