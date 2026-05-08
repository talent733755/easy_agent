# 通用 MCP 服务注册机制实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现通用 MCP 服务注册机制，通过 config.yaml 声明 MCP 服务，graph 自动注册节点和路由，扩展新 MCP 服务无需改 graph 代码。

**Architecture:** 扩展 config 数据类支持 intent/endpoints 字段，创建 MCP 节点工厂，graph 遍历 config 动态注册所有已声明的 MCP 服务节点。

**Tech Stack:** Python, LangGraph, httpx, PyYAML, FastAPI

**设计文档:** `docs/superpowers/specs/2026-05-09-mcp-service-registry-design.md`

---

### Task 1: 扩展 Config 数据类

**Files:**
- Modify: `src/config.py`
- Modify: `config.yaml`

**依赖背景:** 当前 MCPServerConfig 只有 url 和 timeout。需要扩展 intent（意图类型）和 endpoints（端点描述列表），让 graph 知道怎么路由到每个 MCP 服务。

- [ ] **Step 1: 扩展 MCPServerConfig 数据类**

读取 `src/config.py`，找到 `MCPServerConfig`，添加新字段：

```python
@dataclass
class MCPServerConfig:
    url: str
    timeout: int = 30
    intent: str = ""                        # 该 MCP 对应的意图类型
    endpoints: list[dict] = field(default_factory=list)  # 端点描述
```

在 `MCPServerConfig.from_dict()` 中添加字段解析：

```python
@classmethod
def from_dict(cls, data: dict) -> "MCPServerConfig":
    return cls(
        url=data.get("url", ""),
        timeout=data.get("timeout", 30),
        intent=data.get("intent", ""),
        endpoints=data.get("endpoints", []),
    )
```

- [ ] **Step 2: 更新 config.yaml 中 customer 的声明**

在 `config.yaml` 中给 customer 补充 intent 和 endpoints：

```yaml
beauty:
  mcp_servers:
    customer:
      url: "http://localhost:3001"
      timeout: 30
      intent: "query_customer"
      endpoints:
        - name: get_customer
          description: "按姓名查客户信息"
        - name: get_consumption
          description: "按客户ID查消费记录"
        - name: get_service_history
          description: "按客户ID查服务历史"
```

- [ ] **Step 3: 写测试**

在 `tests/test_config.py` 中添加：

```python
class TestMCPServerConfig:
    def test_from_dict_with_endpoints(self):
        data = {
            "url": "http://localhost:3001",
            "timeout": 30,
            "intent": "query_customer",
            "endpoints": [
                {"name": "get_customer", "description": "查客户"},
            ],
        }
        config = MCPServerConfig.from_dict(data)
        assert config.intent == "query_customer"
        assert len(config.endpoints) == 1
        assert config.endpoints[0]["name"] == "get_customer"

    def test_from_dict_defaults(self):
        config = MCPServerConfig.from_dict({"url": "http://localhost:3001"})
        assert config.intent == ""
        assert config.endpoints == []
```

- [ ] **Step 4: 运行测试**

```bash
pytest tests/test_config.py -v
```
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/config.py config.yaml tests/test_config.py
git commit -m "feat: extend MCPServerConfig with intent and endpoints fields

- Add intent field for intent-based routing
- Add endpoints field for MCP endpoint descriptions
- Update customer config with intent and endpoint list

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2: MCP 节点工厂

**Files:**
- Create: `src/nodes/beauty/mcp_service_node.py`
- Create: `tests/test_mcp_service_node.py`

**依赖背景:** 核心扩展点。用一个工厂函数根据 config 声明创建 MCP 调用节点，避免每个 MCP 重复写节点。

- [ ] **Step 1: 创建 MCP 节点工厂**

```python
# src/nodes/beauty/mcp_service_node.py
"""通用 MCP 服务节点工厂：根据 config 创建标准 MCP 调用节点。"""

import httpx
from src.state import AgentState


ENDPOINT_SELECT_PROMPT = """根据用户消息，选择最合适的端点调用。

可用端点：
{endpoint_list}

用户消息：{user_message}

只输出端点名称，不要多余文字。"""


def _extract_last_user_message(state: AgentState) -> str:
    messages = state.get("messages", [])
    for m in reversed(messages):
        if hasattr(m, "content") and not hasattr(m, "tool_call_id"):
            return str(m.content)
    return ""


def _select_endpoint(user_message: str, endpoints: list[dict], model) -> str:
    """用 LLM 选择最合适的端点。"""
    if not endpoints:
        return ""
    if len(endpoints) == 1:
        return endpoints[0]["name"]

    endpoint_list = "\n".join(
        f"- {ep['name']}: {ep.get('description', '')}" for ep in endpoints
    )
    prompt = ENDPOINT_SELECT_PROMPT.format(
        endpoint_list=endpoint_list,
        user_message=user_message[:300],
    )
    try:
        result = model.invoke(prompt)
        content = str(result.content).strip()
        for ep in endpoints:
            if ep["name"] in content:
                return ep["name"]
        return endpoints[0]["name"]
    except Exception:
        return endpoints[0]["name"]


EXTRACT_PARAMS_PROMPT = """从用户消息中提取调用 {endpoint_name} 所需的参数。

端点描述：{endpoint_description}
用户消息：{user_message}

以 JSON 格式输出参数，例如：{{"city": "广州", "date": "2026-05-09"}}
如果没有明确参数，输出空 JSON：{{}}"""


def _extract_params(user_message: str, endpoint_name: str, endpoint_desc: str, model) -> dict:
    """用 LLM 从用户消息中提取调用参数。"""
    prompt = EXTRACT_PARAMS_PROMPT.format(
        endpoint_name=endpoint_name,
        endpoint_description=endpoint_desc,
        user_message=user_message[:300],
    )
    try:
        result = model.invoke(prompt)
        import json
        content = str(result.content).strip()
        # 尝试解析 JSON
        if "{" in content and "}" in content:
            json_str = content[content.index("{"):content.rindex("}") + 1]
            return json.loads(json_str)
        return {}
    except Exception:
        return {}


def create_mcp_service_node(mcp_config: dict, model=None):
    """根据 config 声明创建通用 MCP 调用节点。

    Args:
        mcp_config: dict with keys: url, timeout, intent, endpoints, name
        model: LLM 模型实例（可选，用于端点选择和参数提取）

    Returns:
        标准的 LangGraph node 函数
    """
    url = mcp_config.get("url", "")
    timeout = mcp_config.get("timeout", 30)
    service_name = mcp_config.get("name", "mcp")
    intent = mcp_config.get("intent", "")
    endpoints = mcp_config.get("endpoints", [])

    def mcp_node(state: AgentState) -> dict:
        # 1. 检查意图是否匹配
        state_intent = state.get("intent", "")
        if intent and state_intent not in (intent, "mixed"):
            return {}

        # 2. 提取用户消息
        user_message = _extract_last_user_message(state)
        if not user_message:
            return {}

        # 3. 选择端点
        endpoint_name = _select_endpoint(user_message, endpoints, model) if model else (endpoints[0]["name"] if endpoints else "")

        # 4. 提取参数
        endpoint_desc = ""
        for ep in endpoints:
            if ep["name"] == endpoint_name:
                endpoint_desc = ep.get("description", "")
                break
        params = _extract_params(user_message, endpoint_name, endpoint_desc, model) if model else {}

        # 5. 调用 MCP 端点
        if not url or not endpoint_name:
            return {}

        try:
            response = httpx.post(
                f"{url}/tools/{endpoint_name}",
                json=params,
                timeout=timeout,
            )
            data = response.json()
        except Exception:
            data = {"error": f"调用 {service_name} 服务失败"}

        # 6. 返回结果
        return {
            "customer_context": data if service_name == "customer" else state.get("customer_context", {}),
            "mcp_results": {service_name: data},
        }

    return mcp_node
```

- [ ] **Step 2: 写测试**

```python
# tests/test_mcp_service_node.py
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage
from src.nodes.beauty.mcp_service_node import (
    create_mcp_service_node,
    _select_endpoint,
    _extract_last_user_message,
)


class TestSelectEndpoint:
    def test_single_endpoint_returns_immediately(self):
        endpoints = [{"name": "get_customer", "description": "查客户"}]
        result = _select_endpoint("随便说说", endpoints, MagicMock())
        assert result == "get_customer"

    def test_llm_selects_correct_endpoint(self):
        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content="get_daily_performance")
        endpoints = [
            {"name": "get_customer", "description": "查客户"},
            {"name": "get_daily_performance", "description": "查当日业绩"},
        ]
        result = _select_endpoint("广州今天的业绩", endpoints, mock_model)
        assert result == "get_daily_performance"

    def test_llm_exception_fallback(self):
        mock_model = MagicMock()
        mock_model.invoke.side_effect = Exception("error")
        endpoints = [
            {"name": "get_customer", "description": "查客户"},
            {"name": "get_performance", "description": "查业绩"},
        ]
        result = _select_endpoint("业绩", endpoints, mock_model)
        assert result == "get_customer"  # fallback to first

    def test_empty_endpoints(self):
        result = _select_endpoint("随便说说", [], MagicMock())
        assert result == ""


class TestCreateMCPServiceNode:
    def test_node_skips_wrong_intent(self):
        config = {
            "url": "http://localhost:3001",
            "timeout": 30,
            "intent": "query_customer",
            "endpoints": [{"name": "get_customer", "description": "查客户"}],
            "name": "customer",
        }
        node = create_mcp_service_node(config)
        state = {
            "messages": [HumanMessage(content="天气怎么样")],
            "intent": "general",
            "customer_context": {},
            "mcp_results": {},
        }
        result = node(state)
        assert result == {}

    def test_node_handles_empty_messages(self):
        config = {
            "url": "http://localhost:3001",
            "timeout": 30,
            "intent": "query_customer",
            "endpoints": [{"name": "get_customer", "description": "查客户"}],
            "name": "customer",
        }
        node = create_mcp_service_node(config)
        state = {
            "messages": [],
            "intent": "query_customer",
            "customer_context": {},
            "mcp_results": {},
        }
        result = node(state)
        assert result == {}

    @patch("src.nodes.beauty.mcp_service_node.httpx")
    def test_node_calls_endpoint(self, mock_httpx):
        config = {
            "url": "http://localhost:3001",
            "timeout": 30,
            "intent": "query_customer",
            "endpoints": [{"name": "get_customer", "description": "查客户"}],
            "name": "customer",
        }
        mock_response = MagicMock()
        mock_response.json.return_value = {"name": "张女士", "id": "C001"}
        mock_httpx.post.return_value = mock_response

        node = create_mcp_service_node(config)
        state = {
            "messages": [HumanMessage(content="查询张女士的信息")],
            "intent": "query_customer",
            "customer_context": {},
            "mcp_results": {},
        }
        result = node(state)
        assert result["customer_context"]["name"] == "张女士"
        assert "customer" in result["mcp_results"]

    @patch("src.nodes.beauty.mcp_service_node.httpx")
    def test_node_handles_http_error(self, mock_httpx):
        config = {
            "url": "http://localhost:3001",
            "timeout": 30,
            "intent": "query_customer",
            "endpoints": [{"name": "get_customer", "description": "查客户"}],
            "name": "customer",
        }
        mock_httpx.post.side_effect = Exception("Connection refused")

        node = create_mcp_service_node(config)
        state = {
            "messages": [HumanMessage(content="查询张女士的信息")],
            "intent": "query_customer",
            "customer_context": {},
            "mcp_results": {},
        }
        result = node(state)
        assert "error" in result["mcp_results"]["customer"]
```

- [ ] **Step 3: 运行测试**

```bash
pytest tests/test_mcp_service_node.py -v
```
Expected: PASS

- [ ] **Step 4: 更新 beauty __init__.py 导出**

在 `src/nodes/beauty/__init__.py` 中添加：

```python
from src.nodes.beauty.mcp_service_node import create_mcp_service_node
```

并在 `__all__` 中添加 `"create_mcp_service_node"`。

- [ ] **Step 5: 提交**

```bash
git add src/nodes/beauty/mcp_service_node.py tests/test_mcp_service_node.py src/nodes/beauty/__init__.py
git commit -m "feat: add MCP service node factory

- create_mcp_service_node: factory function from config dict
- LLM-based endpoint selection with fallback
- LLM-based parameter extraction from user message
- Intent filtering: only triggers on matching intent
- Error handling: never blocks agent on MCP failure

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Graph 动态注册

**Files:**
- Modify: `src/graph.py`

**依赖背景:** graph.py 中目前硬编码了 customer MCP 节点。需要改为遍历 config 自动注册所有已声明的 MCP 服务。

- [ ] **Step 1: 修改 graph.py 的 MCP 节点注册逻辑**

在 `build_graph()` 函数中，找到以下硬编码代码：

```python
# 原始硬编码代码（需替换）：
beauty_config = config.beauty
if beauty_config:
    mcp_config = beauty_config.mcp_servers.get("customer")
    if mcp_config:
        builder.add_node("mcp_customer", lambda s: mcp_customer_node(s, mcp_url=mcp_config.url, timeout=mcp_config.timeout))
    else:
        builder.add_node("mcp_customer", lambda s: {"customer_context": {}, "mcp_results": {}})
else:
    builder.add_node("mcp_customer", lambda s: {"customer_context": {}, "mcp_results": {}})
```

替换为动态注册逻辑：

```python
# 新的动态注册代码：
beauty_config = config.beauty
mcp_configs = beauty_config.mcp_servers if beauty_config else {}

# 用 factory 动态注册所有 MCP 服务
registered_mcp_nodes = []
from src.nodes.beauty.mcp_service_node import create_mcp_service_node

for service_name, svc_config in mcp_configs.items():
    svc_dict = {
        "url": svc_config.url,
        "timeout": svc_config.timeout,
        "intent": svc_config.intent,
        "endpoints": svc_config.endpoints,
        "name": service_name,
    }
    node_fn = create_mcp_service_node(svc_dict, model=model)
    builder.add_node(f"mcp_{service_name}", node_fn)
    registered_mcp_nodes.append(service_name)

# 兜底：如果没有配置任何 MCP 服务
if "customer" not in registered_mcp_nodes:
    builder.add_node("mcp_customer", lambda s: {"customer_context": {}, "mcp_results": {}})
```

- [ ] **Step 2: 修改 _route_intent 支持动态路由**

将 `_route_intent()` 从硬编码改为从 config 读取：

```python
def _build_mcp_intent_routes(mcp_configs: dict) -> dict:
    """从 MCP config 构建意图到节点名的路由映射。"""
    routes = {}
    for name, svc_config in mcp_configs.items():
        intent = svc_config.intent if isinstance(svc_config, dict) else getattr(svc_config, "intent", "")
        if intent:
            routes[intent] = f"mcp_{name}"
    return routes


def _route_intent(state: AgentState, mcp_routes: dict = None) -> str:
    """Route based on intent classification.

    Routes:
        - intent mapped in mcp_routes -> corresponding mcp node
        - "knowledge_query" -> knowledge_retrieve
        - "mixed" -> knowledge_retrieve (fan-out to all MCP nodes)
        - "general"/unknown -> skip -> memory_retrieve
    """
    intent = state.get("intent", "general")

    if intent == "knowledge_query":
        return "knowledge"
    elif intent == "mixed":
        return "knowledge"  # Fan-out handles MCP nodes
    elif intent in (mcp_routes or {}):
        return mcp_routes[intent]
    else:
        return "skip"
```

- [ ] **Step 3: 构建动态 fan-out 边**

在 `build_graph()` 中，替代硬编码的 `builder.add_edge("intent_classify", "mcp_customer")`：

```python
# 添加 fan-out 边：mixed 意图时同时走所有 MCP + knowledge
for service_name in registered_mcp_nodes:
    builder.add_edge("intent_classify", f"mcp_{service_name}")

# 动态构建条件路由
mcp_routes = {}
for name, svc_config in mcp_configs.items():
    intent = getattr(svc_config, "intent", "")
    if intent:
        mcp_routes[intent] = f"mcp_{name}"

builder.add_conditional_edges("intent_classify", lambda s: _route_intent(s, mcp_routes), {
    "knowledge": "knowledge_retrieve",
    **{f"mcp_{name}": f"mcp_{name}" for name in registered_mcp_nodes},
    "skip": "memory_retrieve",
})
```

- [ ] **Step 4: 动态收敛边**

将所有 MCP 节点都汇聚到 `memory_retrieve`：

```python
builder.add_edge("knowledge_retrieve", "memory_retrieve")
for service_name in registered_mcp_nodes:
    builder.add_edge(f"mcp_{service_name}", "memory_retrieve")
```

- [ ] **Step 5: 验证图构建**

```bash
python -c "from src.graph import build_graph; g = build_graph(); print('Graph built successfully')"
```
Expected: `Graph built successfully`

- [ ] **Step 6: 运行测试**

```bash
pytest tests/test_graph.py -v
```
Expected: PASS

- [ ] **Step 7: 提交**

```bash
git add src/graph.py
git commit -m "feat: dynamic MCP service registration in graph

- Iterate config.beauty.mcp_servers to auto-register MCP nodes
- Dynamic intent routing from config
- Dynamic fan-out edges for parallel MCP execution
- Backward compatible: customer MCP still works

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 4: 意图扩展

**Files:**
- Modify: `src/nodes/beauty/intent_node.py`

**依赖背景:** 当前意图分类只支持 4 种硬编码意图。需要让 intent_node 能够从 config 中读取已注册的 MCP 服务列表，将 MCP 服务的 intent 值也纳入分类选项。

- [ ] **Step 1: 扩展意图分类 prompt**

读取 `src/nodes/beauty/intent_node.py`，找到 `intent_classify_node` 函数，修改 LLM prompt 中的意图类型列表：

在 prompt 模板的可用意图列表中，动态注入 config 中所有 MCP 服务的 intent 值：

```python
def intent_classify_node(state: AgentState, config=None) -> dict:
    # ... existing code ...

    # 动态构建可用意图列表
    intent_options = ["knowledge_query", "mixed", "general"]
    if config and config.beauty:
        for svc_config in config.beauty.mcp_servers.values():
            intent = getattr(svc_config, "intent", "")
            if intent and intent not in intent_options:
                intent_options.append(intent)

    # 在 prompt 中使用 intent_options
```

- [ ] **Step 2: 更新规则匹配 fallback**

在 `_rule_based_classify` 中，从 config 读取所有 MCP 服务的 intent 关键词，而不是硬编码：

```python
# _rule_based_classify 的关键词映射应从 config 扩展
# 现有：
#   "query_customer": ["客户", "会员", "消费", "充值"]
# 新增（从 config 读取）
```

- [ ] **Step 3: 运行测试**

```bash
pytest tests/ -v
```
Expected: PASS

- [ ] **Step 4: 提交**

```bash
git add src/nodes/beauty/intent_node.py
git commit -m "feat: expand intent classification to support dynamic MCP services

- LLM prompt dynamically lists all MCP intent types from config
- Rule-based classify supports new intent keywords

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 5: 集成测试

**Files:**
- Create: `tests/test_mcp_registry_integration.py`

**依赖背景:** 验证完整流程：config 声明多个 MCP → graph 自动注册 → 意图路由到正确节点。

- [ ] **Step 1: 创建集成测试**

```python
# tests/test_mcp_registry_integration.py
"""MCP 服务注册机制集成测试。"""
import pytest
from unittest.mock import MagicMock, patch
from src.config import MCPServerConfig
from src.nodes.beauty.mcp_service_node import create_mcp_service_node


class TestMCPRegistryIntegration:
    def test_multiple_mcp_configs(self):
        """验证多个 MCP 服务配置可以同时创建节点。"""
        configs = [
            {
                "url": "http://localhost:3001",
                "timeout": 30,
                "intent": "query_customer",
                "endpoints": [{"name": "get_customer", "description": "查客户"}],
                "name": "customer",
            },
            {
                "url": "http://localhost:3002",
                "timeout": 30,
                "intent": "query_performance",
                "endpoints": [{"name": "get_daily_performance", "description": "查当日业绩"}],
                "name": "performance",
            },
        ]
        for config in configs:
            node = create_mcp_service_node(config)
            assert callable(node)

    def test_config_dataclass_with_new_fields(self):
        """验证 MCPServerConfig 新字段正常工作。"""
        config = MCPServerConfig.from_dict({
            "url": "http://localhost:3001",
            "timeout": 30,
            "intent": "query_customer",
            "endpoints": [
                {"name": "get_customer", "description": "查客户"},
                {"name": "get_consumption", "description": "查消费"},
            ],
        })
        assert config.intent == "query_customer"
        assert len(config.endpoints) == 2

    def test_empty_mcp_config_no_crash(self):
        """验证空 MCP 配置不崩溃。"""
        config = {
            "url": "",
            "timeout": 30,
            "intent": "query_test",
            "endpoints": [],
            "name": "empty",
        }
        node = create_mcp_service_node(config)
        state = {
            "messages": [MagicMock(content="test")],
            "intent": "query_test",
            "customer_context": {},
            "mcp_results": {},
        }
        result = node(state)
        assert result == {}
```

- [ ] **Step 2: 运行全部测试**

```bash
pytest tests/ -v
```
Expected: 新测试 PASS

- [ ] **Step 3: 提交**

```bash
git add tests/test_mcp_registry_integration.py
git commit -m "test: add MCP service registry integration tests

- Multiple MCP configs create nodes without crash
- MCPServerConfig new fields work correctly
- Empty MCP config graceful handling

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## 验收标准检查清单

- [ ] config.yaml 中声明 1 个 MCP 服务 → graph 自动创建节点
- [ ] 声明 2+ 个 MCP 服务 → graph 自动注册所有节点，路由自动扩展
- [ ] 不声明任何 MCP 服务 → 图正常构建，跳过所有 MCP 节点
- [ ] 现有 customer MCP 功能不受影响
- [ ] 新 MCP 节点调用时，LLM 自动选择端点和提取参数

---

## 扩展新 MCP 服务的步骤（后续指南）

1. 在 config.yaml 的 `beauty.mcp_servers` 下加配置（url、timeout、intent、endpoints）
2. 在 intent_node.py 的规则匹配中补充关键词（可选）
3. 重启服务，graph 自动识别新 MCP 节点
