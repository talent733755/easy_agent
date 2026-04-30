# Easy Agent

A Hermes-style intelligent agent built with LangGraph for learning agent orchestration.

## Features

- **Custom StateGraph**: Hand-written ReAct agent loop
- **Tool System**: 8 built-in tools with risk-level classification
- **Four-Layer Memory**: File memory, SQLite FTS5, FAISS vector search
- **Human-in-the-Loop**: Dangerous operation confirmation
- **Multi-Provider**: OpenAI, Anthropic, ZhiPu AI, and custom proxies
- **Self-Evolution**: Simplified nudge engine for auto memory updates

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Set API key
export ZHIPU_API_KEY="your-api-key"

# Run
python main.py --provider zhipu
python main.py --provider openai --model gpt-4o
python main.py --provider anthropic --model claude-sonnet-4-6
```

## Configuration

编辑 `config.yaml` 配置你的 Provider：

```yaml
providers:
  # 内置 Provider
  openai:
    type: openai
    api_key: "${OPENAI_API_KEY}"
    base_url: "https://api.openai.com/v1"
    default_model: "gpt-4o"
    timeout: 60

  # 智谱 AI
  zhipu:
    type: openai_compatible
    api_key: "${ZHIPU_API_KEY}"
    base_url: "https://open.bigmodel.cn/api/paas/v4"
    default_model: "glm-4-plus"

  # 自定义代理（如 codingplan）
  my_proxy:
    type: openai_compatible
    api_key: "${MY_PROXY_API_KEY}"
    base_url: "https://your-proxy-url.com/v1"
    default_model: "gpt-4o"
    timeout: 120

active_provider: "my_proxy"
```

**支持的 Provider 类型：**
- `openai`: OpenAI 官方 API
- `openai_compatible`: 兼容 OpenAI API 的代理服务
- `anthropic`: Anthropic Claude API

## Project Structure

```
easy_agent/
├── main.py              # CLI entry
├── config.yaml          # Default configuration
├── src/
│   ├── state.py         # AgentState definition
│   ├── config.py        # Config loading
│   ├── graph.py         # StateGraph assembly
│   ├── nodes/           # Graph nodes
│   ├── tools/           # Tool implementations
│   ├── memory/          # Memory stores
│   └── providers/       # LLM provider adapters
└── tests/               # Test suite
```

## Commands

| Command | Description |
|---------|-------------|
| `/help` | Show help |
| `/quit` | Exit agent |
| `/memory` | Show MEMORY.md |
| `/profile` | Show USER.md |
| `/clear` | New session |

## Architecture

Inspired by [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent):

```
START → memory_retrieve → agent_node → tool_executor → observe
  → human_gate → nudge_check → memory_save → (loop back)
```

## Test Coverage

```bash
python -m pytest tests/ -v
# 49 tests covering all components
```