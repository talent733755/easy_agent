# Easy Agent

A Hermes-style intelligent agent built with LangGraph for learning agent orchestration.

## Features

- **Custom StateGraph**: Hand-written ReAct agent loop
- **Tool System**: 8 built-in tools with risk-level classification
- **Four-Layer Memory**: File memory, SQLite FTS5, FAISS vector search
- **Human-in-the-Loop**: Dangerous operation confirmation
- **Multi-Provider**: OpenAI, Anthropic, ZhiPu AI
- **Self-Evolution**: Simplified nudge engine for auto memory updates

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Configure
cp config.yaml config.local.yaml
# Edit config.local.yaml with your API keys

# Run
python main.py --provider zhipu
python main.py --provider openai --model gpt-4.1
python main.py --provider anthropic --model claude-sonnet-4-6
```

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