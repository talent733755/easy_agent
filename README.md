# Easy Agent

A Hermes-style intelligent agent built with LangGraph for learning agent orchestration.

## Features

- **Custom StateGraph**: Hand-written ReAct agent loop
- **Tool System**: 8 built-in tools with risk-level classification
- **Four-Layer Memory**: File memory, SQLite FTS5, FAISS vector search
- **Human-in-the-Loop**: Dangerous operation confirmation
- **Multi-Provider**: OpenAI, Anthropic, ZhiPu AI, and custom proxies
- **Self-Evolution**: Simplified nudge engine for auto memory updates
- **Web Interface**: Real-time WebSocket chat with FastAPI backend
- **Beauty Agent**: Industry-specific agent with intent classification, knowledge RAG, and MCP integration

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Set API key
export ZHIPU_API_KEY="your-api-key"

# Run CLI
python main.py --provider zhipu
python main.py --provider openai --model gpt-4o
python main.py --provider anthropic --model claude-sonnet-4-6

# Run Web Interface
python web_app.py
# Open http://localhost:8080 in your browser
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

## Web Interface

Start the web interface for real-time chat:

```bash
# Start web server
python web_app.py --host 0.0.0.0 --port 8080

# With auto-reload for development
python web_app.py --reload
```

### WebSocket API

Connect to `ws://localhost:8080/ws` for real-time chat:

```javascript
// Connect
const ws = new WebSocket('ws://localhost:8080/ws');

// Receive session info
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'session') {
    console.log('Session ID:', data.session_id);
  }
};

// Send message
ws.send(JSON.stringify({type: 'message', content: '你好'}));

// Handle response
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'response') {
    console.log('Agent:', data.content);
  }
};
```

### Slash Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/memory` | Show MEMORY.md content |
| `/profile` | Show USER.md content |
| `/session` | Show current session ID |
| `/clear` | Start a new session |

## Beauty Agent

The Beauty Agent is an industry-specific extension for beauty salon customer service:

### Features

- **Intent Classification**: Automatically classifies queries into customer lookup, knowledge query, mixed, or general
- **Knowledge RAG**: Retrieves relevant information from product, procedure, script, and troubleshooting knowledge bases
- **MCP Integration**: Connects to Customer MCP server for real-time customer data lookup

### Architecture

```
User Query → Intent Classify → [Knowledge Retrieve] → Memory Retrieve → Agent → Response
                              ↘ [MCP Customer]     ↗
```

### Starting MCP Server

The Customer MCP server provides customer data:

```bash
# Start Customer MCP server
cd mcp_servers/customer
python main.py --port 3001
```

### Configuration

Add beauty configuration to `config.yaml`:

```yaml
beauty:
  knowledge_base:
    base_dir: "./knowledge/jiaolifu"
    indexes:
      products:
        path: "products"
        description: "产品知识、成分功效"
      procedures:
        path: "procedures"
        description: "服务流程、操作标准"
      scripts:
        path: "scripts"
        description: "销售话术、接待标准"
      troubleshooting:
        path: "troubleshooting"
        description: "问题处理、应急方案"

  mcp_servers:
    customer:
      url: "http://localhost:3001"
      timeout: 30

  intent_prompt: "./prompts/beauty_intent_v1.txt"

  web:
    host: "0.0.0.0"
    port: 8080
```

### Example Queries

- **Customer Lookup**: "查一下张女士的档案"
- **Knowledge Query**: "推荐适合油性皮肤的护肤产品"
- **Mixed Query**: "张女士上次做了什么项目，推荐适合她的护理流程"
- **General**: "你好"

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
# 70+ tests covering all components including integration tests
```

### Integration Tests

The integration test suite covers:

- Intent classification (customer, knowledge, mixed, general)
- MCP client node (success, not found, timeout, connection error)
- Knowledge retrieval (valid dir, empty dir)
- End-to-end graph flows (customer query, knowledge query, general query)
- Web integration (health endpoint, index page, WebSocket)
- MCP server configuration loading

Run integration tests specifically:

```bash
python -m pytest tests/test_integration.py -v
```