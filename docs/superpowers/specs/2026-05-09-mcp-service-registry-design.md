# 通用 MCP 服务注册机制设计

**日期:** 2026-05-09
**状态:** 已批准

---

## 背景

当前系统只有一个 MCP 服务（customer），硬编码在 graph.py 中。需要设计一个通用的 MCP 服务注册机制，方便后续扩展任意业务 MCP（业绩查询、消耗查询、预测等），无需每次改 graph。

**核心需求：** 加新 MCP 服务只需在 config.yaml 声明配置，代码零改动或最小改动。

---

## 设计目标

1. **配置驱动** — 在 config.yaml 中声明 MCP 服务（URL、意图、端点）
2. **节点工厂** — 统一的节点创建工厂，避免每个 MCP 重复写节点
3. **动态注册** — graph.py 遍历 config 自动注册节点和路由
4. **只读 MySQL** — 所有 MCP 服务通过已有的业务接口对接，Agent 不直接碰数据库

---

## 架构概览

```
config.yaml (声明所有 MCP 服务)
    │
    ▼
graph.py build_graph() (遍历配置，动态注册)
    │
    ├── mcp_customer (现有)
    ├── mcp_performance (未来新增)
    ├── mcp_consume (未来新增)
    └── ...

每个 MCP 节点 = create_mcp_service_node(config) 工厂创建
```

---

## config.yaml 配置格式

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

    # --- 以下为未来扩展示例 ---

    # performance:
    #   url: "http://localhost:3002"
    #   timeout: 30
    #   intent: "query_performance"
    #   endpoints:
    #     - name: get_daily_performance
    #       description: "按城市查当日业绩"
    #     - name: get_daily_consumption
    #       description: "按城市查当日消耗"

    # member:
    #   url: "http://localhost:3003"
    #   timeout: 30
    #   intent: "query_member"
    #   endpoints:
    #     - name: get_member_detail
    #       description: "按会员号查详细信息"
```

---

## 核心组件

### 1. 节点工厂：create_mcp_service_node

**文件：** `src/nodes/beauty/mcp_service_node.py`

```python
def create_mcp_service_node(mcp_config: dict):
    """根据 config 声明创建通用 MCP 调用节点。"""
    url = mcp_config["url"]
    timeout = mcp_config.get("timeout", 30)
    service_name = mcp_config.get("name", "mcp")

    def mcp_node(state: AgentState) -> dict:
        # 1. 从 state 中提取触发意图
        # 2. 根据 config 中的 endpoints 列表，用 LLM 选择要调用的端点
        # 3. 从对话中提取参数
        # 4. HTTP POST 调用 MCP 端点
        # 5. 返回结果写入 state

        return {"customer_context": result, "mcp_results": {service_name: result}}

    return mcp_node
```

**端点选择逻辑：**
- LLM 根据用户消息 + endpoints.description 列表，选择最合适的端点
- 参数从对话中由 LLM 提取（如城市名、日期、会员号）

### 2. Graph 动态注册

**文件：** `src/graph.py`

在 `build_graph()` 中：

```python
# 遍历所有配置的 MCP 服务，动态注册节点
mcp_configs = config.beauty.mcp_servers if config.beauty else {}
for service_name, svc_config in mcp_configs.items():
    svc_config["name"] = service_name  # 注入服务名
    node_fn = create_mcp_service_node(svc_config)
    builder.add_node(f"mcp_{service_name}", lambda s, fn=node_fn: fn(s))
```

**路由扩展：**
- `_route_intent()` 函数从 config 中读取已注册的 MCP 服务意图，返回对应的节点名
- 新增意图时只需修改 config 中的 `intent` 字段

### 3. 意图路由扩展

**文件：** `src/graph.py` 中的 `_route_intent()`

```python
# 动态构建意图 → 节点名的映射
def _build_intent_routes(mcp_configs: dict) -> dict:
    """从 config 构建意图到 MCP 节点的路由映射。"""
    routes = {}
    for name, config in mcp_configs.items():
        intent = config.get("intent")
        if intent:
            routes[intent] = f"mcp_{name}"
    return routes

# _route_intent 使用动态映射
INTENT_ROUTES = _build_intent_routes(mcp_configs)
```

---

## 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `src/nodes/beauty/mcp_service_node.py` | 新建 | MCP 节点工厂 |
| `src/nodes/beauty/__init__.py` | 修改 | 导出 create_mcp_service_node |
| `src/graph.py` | 修改 | 动态注册 MCP 节点 |
| `config.yaml` | 修改 | 添加已有 customer 的 endpoints 描述 |
| `tests/test_mcp_service_node.py` | 新建 | 工厂 + 节点测试 |

---

## 向后兼容

- 现有的 `mcp_customer_node` 和 `mcp_customer` 路由**保持不变**
- 新机制与旧节点**共存**，不破坏现有功能
- 可以逐步将旧节点迁移到新工厂模式

---

## 验收标准

1. config.yaml 中声明 1 个 MCP 服务 → graph 自动创建节点，不需要改 graph.py
2. 声明 2+ 个 MCP 服务 → graph 自动注册所有节点，路由自动扩展
3. 不声明任何 MCP 服务 → 图正常构建，跳过所有 MCP 节点
4. 现有 customer MCP 功能不受影响
5. 新 MCP 节点调用时，LLM 自动选择端点和提取参数

---

## 扩展新 MCP 服务的完整步骤（后续指南）

1. 写 MCP 服务（HTTP 端点，接受参数返回 JSON）
2. 在 config.yaml 的 `beauty.mcp_servers` 下加配置（url、timeout、intent、endpoints）
3. 可选：在 intent_node 的 prompt 中补充意图关键词
4. 重启服务，graph 自动识别
